# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

drinks = pd.read_csv("../input/alcohol-consumption/drinks.csv")
drinks.head()
# Графические параметры

import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12,28)
# Нормализация признаков

X = drinks.iloc[:, 1:5].values # выделяем только числовые колонки

X_norm = (X - X.mean(axis=0)) / X.std(axis=0) # вычитаем среднее и делим на стандартное отклонение
# Проверка

X_norm.mean(axis=0)
# Проверка

X_norm.std(axis=0)
drinks['beer_servings'] = X_norm[:, 0]

drinks['spirit_servings'] = X_norm[:, 1]

drinks['wine_servings'] = X_norm[:, 2]

drinks['total_litres_of_pure_alcohol'] = X_norm[:, 3]



numerical_features = drinks.columns.values[1:5]

numerical_features
drinks[numerical_features].corr()
# Видно, что признак total достаточно сильно коррелирует с остальными. Попробуем обучить линейную регрессию



# Выделяем матрицу признаков X и целевую переменную y

X = drinks[numerical_features].drop('total_litres_of_pure_alcohol', axis=1)

y = drinks['total_litres_of_pure_alcohol']



# Делим на train и validation (75 на 25)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=15)



# Обучаем простую линейную регрессию без регуляризации

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_valid)



# Считаем качество модели (коэффициент детерминации)

from sklearn.metrics import r2_score

r2_score(y_valid, y_pred)
# Качество достаточно высокое. Посмотрим на коэффициенты модели

linreg.coef_
# Судя по всему, пиво -- наиболее важный признак
# Попробуем обучить дерево и визуализировать его

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_valid)



r2_score(y_pred, y_valid)
# Качество чуть лучше. А что с главными признаками?

# Визуализация

from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot')

print(open('tree.dot').read()) 

# Далее скопировать полученный текст на сайт https://dreampuf.github.io/GraphvizOnline/ и сгенерировать граф

# Вставить картинку в блокнот: ![](ссылка)
# Попробуем иерархическую кластеризацию по 4 признакам

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

Z = linkage(X, method='average', metric='euclidean')

dend = dendrogram(Z, orientation='left', color_threshold=0.0, labels=drinks['country'].values)
label = fcluster(Z, 2.2, criterion='distance')

np.unique(label)
drinks['label'] = label

for i, group in drinks.groupby('label')['country']:

    print('=' * 10)

    print('cluster {}'.format(i))

    print(group.values)
# Попробуем метод k-means



X = drinks[numerical_features].values



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='random', n_init=10, random_state=15)

kmeans.fit(X)



label_kmeans = kmeans.labels_

drinks['label_kmeans'] = label_kmeans



for i, group in drinks.groupby('label_kmeans')['country']:

    print('=' * 10)

    print('cluster {}'.format(i))

    print(group.values)
# Подберём наилучшее количество кластеров



crit = []

for k in range(2, 8):

    kmeans = KMeans(n_clusters=k, random_state=15)

    kmeans.fit(X)

    crit.append(kmeans.inertia_)

    

plt.plot(range(2,8), crit);
# Из графика можем сделать вывод, что данные достаточно плохо кластеризуются. 

# Однако, возможно, оптимально будет выделить три кластера
# Попробуем понизить размерность, а затем визуализировать наши результаты



from sklearn.decomposition import PCA

pca = PCA(n_components=4)

pca.fit(X)



Z = pca.transform(X)



pca.explained_variance_ratio_
plt.bar(x=['Component '+str(i) for i in range(1, 5)], height=pca.explained_variance_ratio_);
pca.explained_variance_ratio_.cumsum()
# Первые две компоненты объясняют более 88 % дисперсии исходных данных. 

# Сделаем преобразование исходного 4-мерного пространства признаков в 2-мерное пространство главных компонент



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X)



Z = pca.transform(X)
Z
# Визуализируем результаты с метками кластеров по агломеративному алгоритму



plt.plot(Z[label == 1, 0], Z[label == 1, 1], 'bo', label='Cluster 1')

plt.plot(Z[label == 2, 0], Z[label == 2, 1], 'wo', label='France')

plt.plot(Z[label == 3, 0], Z[label == 3, 1], 'ro', label='Cluster 3')

plt.plot(Z[label == 4, 0], Z[label == 4, 1], 'yo', label='Cluster 4')

plt.plot(Z[label == 5, 0], Z[label == 5, 1], 'mo', label='Cluster 5')

plt.legend(loc=0);
# Визуализируем результаты с метками кластеров по методу k-средних



plt.plot(Z[label_kmeans == 0, 0], Z[label_kmeans == 0, 1], 'ro', label='Cluster 0')

plt.plot(Z[label_kmeans == 1, 0], Z[label_kmeans == 1, 1], 'yo', label='Cluster 1')

plt.plot(Z[label_kmeans == 2, 0], Z[label_kmeans == 2, 1], 'mo', label='Cluster 2')

plt.plot(Z[label_kmeans == 3, 0], Z[label_kmeans == 3, 1], 'bo', label='Cluster 3')

plt.legend(loc=0);
# Попробуем сократить размерность с помощью t-SNE



from sklearn.manifold import TSNE

tsne = TSNE(random_state=15)



X_tsne = tsne.fit_transform(X)



plt.figure(figsize=(12, 10))

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, 

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('nipy_spectral', 5))

plt.colorbar();
plt.figure(figsize=(12, 10))

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_kmeans, 

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('nipy_spectral', 5))

plt.colorbar();
fix, ax = plt.subplots(1, 2)

ax[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, 

              edgecolor='none', alpha=0.7, s=40,

              cmap=plt.cm.get_cmap('nipy_spectral', 5))

ax[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_kmeans, 

              edgecolor='none', alpha=0.7, s=40,

              cmap=plt.cm.get_cmap('nipy_spectral', 5));