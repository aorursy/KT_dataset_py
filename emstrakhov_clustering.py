import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.cluster import AgglomerativeClustering, KMeans

from sklearn.preprocessing import StandardScaler
planets = pd.read_csv('../input/unsupervised-learning-data/planets.csv', 

                      index_col='Name')

planets.head(10)
# Признаки имеют разные масштабы -> требуется препроцессинг

X = StandardScaler().fit_transform(planets)



print(X)
# Применяем иерархическую кластеризацию с построением полного дерева

merging = linkage(X, method='single')



# Строим дендрограмму

dendrogram(merging, labels=planets.index, leaf_font_size=10)

plt.show()
# Кластеризация методом k-средних на 3 кластера

kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

labels = kmeans.predict(X)



# Вывод результатов в виде словаря

print(dict(zip(planets.index, labels)))
countries = pd.read_csv('../input/unsupervised-learning-data/countries.tsv', 

                        sep='\t', index_col='Country')

countries.head()
# Препроцессинг: меняем запятые на точки

countries.iloc[:, :3] = countries.iloc[:, :3].apply(lambda x: x.str.replace(',', '.').astype(float), 

                                                    axis=1)



# Препроцессинг: стандартизация

X = StandardScaler().fit_transform(countries)

print(X)
# Применяем иерархическую кластеризацию с построением полного дерева

merging = linkage(X, method='single')



# Строим дендрограмму

dendrogram(merging, labels=countries.index, leaf_rotation=90, leaf_font_size=8)

plt.show()
# "Разрежем" дендрограмму на высоте 1.75, получим три кластера

clusters = fcluster(merging, 1.75, criterion='distance')

print(clusters)
# Добавим полученный столбец к датафрейму

countries['label'] = clusters

print(countries.sort_values('label')['label'])
# Кластеризуем данные методом k-средних

kmeans = KMeans(n_clusters=3)

clusters_kmeans = kmeans.fit_predict(X)

countries['label_kmeans'] = clusters_kmeans

print(countries.sort_values('label_kmeans')['label_kmeans'])
# Подберём наилучшее количество кластеров ("правило локтя")



crit = []

for k in range(2, 8):

    kmeans = KMeans(n_clusters=k, random_state=15)

    kmeans.fit(X)

    crit.append(kmeans.inertia_)

    

plt.plot(range(2,8), crit)

plt.show()