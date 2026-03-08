import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# Загружается файл из папки data на рабочем столе
file_path = 'data/Кто_ты_в_потоке_—_Опрос_Ответы_Ответы_на_форму_1.csv'
df = pd.read_csv(file_path)

# Убирается колонка с временем, она не нужна для модели
raw_data = df.iloc[:, 1:]

# Функция очистки: берется первая цифра из ответов (например, "3 = Иногда")
# Если цифры нет (как в ответе "70/30"), ставится среднее значение 3
def clean_to_int(text):
    text = str(text).strip()
    if text and text[0].isdigit():
        return int(text[0])
    return 3

# Создается чистый числовой датасет
df_numeric = raw_data.applymap(clean_to_int)

# Масштабирование: приводятся все ответы к единому масштабу
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# 2. ПОИСК ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ (МЕТОД ЛОКТЯ)

inertia = []
K = range(1, 11)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42, n_init=10)
    k_means.fit(X_scaled)
    inertia.append(k_means.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Число кластеров')
plt.ylabel('Инерция (ошибка)')
plt.title('Метод локтя: находится изгиб графика')
plt.grid(True)
plt.show()
 
# 3. ОБУЧЕНИЕ МОДЕЛЕЙ (K-MEANS И ИЕРАРХИЧЕСКАЯ)

# На основе "локтя" выбирается 3 кластера (оптимально для 59 человек)
n_clusters = 3

# Модель 1: K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Модель 2: Иерархическая кластеризация
hierarch = AgglomerativeClustering(n_clusters=n_clusters)
df['Hierarch_Cluster'] = hierarch.fit_predict(X_scaled)

# Метрика силуэта (качество разделения)
score = silhouette_score(X_scaled, df['KMeans_Cluster'])
print(f"Качество кластеризации (Silhouette Score): {score:.2f}")

# 4. ВИЗУАЛИЗАЦИЯ ДЛЯ ПРЕЗЕНТАЦИИ

# Сжимаются данные до 2D через PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 7))
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['KMeans_Cluster'], cmap='viridis', s=100)
plt.colorbar(scatter, label='Номер кластера')

# Пояснения прямо на график
plt.title('Результат кластеризации студентов (K-Means)')
plt.xlabel('Главная компонента 1 (Ось X): Влияет на стиль обучения')
plt.ylabel('Главная компонента 2 (Ось Y): Влияет на привычки/ИИ')
plt.grid(True, alpha=0.3)
plt.show()
 
# Дендрограмма (Иерархическая структура)
plt.figure(figsize=(10, 7))
link = linkage(X_scaled, method='ward')
dendrogram(link)
plt.title('Дендрограмма: как студенты объединяются в группы')
plt.xlabel('Индексы студентов (всего 59)')
plt.ylabel('Расстояние сходства')
plt.show()

# Сохраняется результат в новый файл для Researcher-а
df.to_csv('data/clustered_students.csv', index=False)