import pandas as pd 

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler





data = pd.read_csv("../input/cs178-wine-quality/winequality-white.csv", sep=";") 

data.head()
import plotly.express as px

fig = px.pie(data, values='quality', names='quality')

fig.show()
# Guardamos la label como "etiquetas originales" si no lo está ya

if 'quality' in data.columns:

    data['original_labels'] = data['quality']

    del data['quality']
data.info()
data.describe().T
sns.kdeplot(data['fixed acidity'], shade=True);
data['fixed acidity'] = np.log1p(data['fixed acidity'])

data[['fixed acidity']] = MinMaxScaler().fit_transform(data[['fixed acidity']])

sns.distplot(data['fixed acidity'], fit = stats.norm)

print(data['fixed acidity'].skew(), data['fixed acidity'].kurt())
sns.kdeplot(data['volatile acidity'], shade=True);
data['volatile acidity'] = np.log1p(data['volatile acidity'])

data[['volatile acidity']] = MinMaxScaler().fit_transform(data[['volatile acidity']])

sns.distplot(data['volatile acidity'], fit = stats.norm)

print(data['volatile acidity'].skew(), data['volatile acidity'].kurt())
sns.kdeplot(data['citric acid'], shade=True);
data['citric acid'] = np.log1p(data['citric acid'])

data[['citric acid']] = MinMaxScaler().fit_transform(data[['citric acid']])

sns.distplot(data['citric acid'], fit = stats.norm)

print(data['citric acid'].skew(), data['citric acid'].kurt())
sns.kdeplot(data['residual sugar'], shade=True);
data['residual sugar'] = np.log1p(data['residual sugar'])

data[['residual sugar']] = MinMaxScaler().fit_transform(data[['residual sugar']])

sns.distplot(data['residual sugar'], fit = stats.norm)

print(data['residual sugar'].skew(), data['residual sugar'].kurt())
sns.kdeplot(data['chlorides'], shade=True);
data['chlorides'] = np.log1p(data['chlorides'])

data[['chlorides']] = MinMaxScaler().fit_transform(data[['chlorides']])

sns.distplot(data['chlorides'], fit = stats.norm)

print(data['chlorides'].skew(), data['chlorides'].kurt())
sns.kdeplot(data['free sulfur dioxide'], shade=True);
data['free sulfur dioxide'] = np.log1p(data['free sulfur dioxide'])

data[['free sulfur dioxide']] = MinMaxScaler().fit_transform(data[['free sulfur dioxide']])

sns.distplot(data['free sulfur dioxide'], fit = stats.norm)

print(data['free sulfur dioxide'].skew(), data['free sulfur dioxide'].kurt())
sns.kdeplot(data['total sulfur dioxide'], shade=True);
data['total sulfur dioxide'] = np.log1p(data['total sulfur dioxide'])

data[['total sulfur dioxide']] = MinMaxScaler().fit_transform(data[['total sulfur dioxide']])

sns.distplot(data['total sulfur dioxide'], fit = stats.norm)

print(data['total sulfur dioxide'].skew(), data['total sulfur dioxide'].kurt())
sns.kdeplot(data['density'], shade=True);
data['density'] = np.log1p(data['density'])

data[['density']] = MinMaxScaler().fit_transform(data[['density']])

sns.distplot(data['density'], fit = stats.norm)

print(data['density'].skew(), data['density'].kurt())
sns.kdeplot(data['pH'], shade=True);
data['pH'] = np.log1p(data['pH'])

data[['pH']] = MinMaxScaler().fit_transform(data[['pH']])

sns.distplot(data['pH'], fit = stats.norm)

print(data['pH'].skew(), data['pH'].kurt())
sns.kdeplot(data['sulphates'], shade=True);
data['sulphates'] = np.log1p(data['sulphates'])

data[['sulphates']] = MinMaxScaler().fit_transform(data[['sulphates']])

sns.distplot(data['sulphates'], fit = stats.norm)

print(data['sulphates'].skew(), data['sulphates'].kurt())
sns.kdeplot(data['alcohol'], shade=True);
data['alcohol'] = np.log1p(data['alcohol'])

data[['alcohol']] = MinMaxScaler().fit_transform(data[['alcohol']])

sns.distplot(data['alcohol'], fit = stats.norm)

print(data['alcohol'].skew(), data['alcohol'].kurt())
import seaborn as sns; 

sns.set(style="ticks", color_codes=True)

sns.pairplot(data, kind="reg")
corrmat = data.corr(method='spearman')

sns.clustermap(corrmat, cmap="YlGnBu", linewidths=0.1);
from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin_min



np.random.seed(42)

data_k_means = data.copy()
%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

score = [kmeans[i].fit(data_k_means).score(data_k_means) for i in range(len(kmeans))]

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
kmeans = KMeans(n_clusters=5).fit(data_k_means)

centroids = kmeans.cluster_centers_

print(centroids)
data_k_means['pred_cluster_labels'] = (kmeans.labels_ + 1).tolist()

data_k_means.head(3)
fig = px.pie(data_k_means, values='pred_cluster_labels', names='pred_cluster_labels')

fig.show()
from sklearn import metrics

print(f"Homogeneity score: {metrics.homogeneity_score(data_k_means['original_labels'], data_k_means['pred_cluster_labels'])}")

print(f"Completeness score: {metrics.completeness_score(data_k_means['original_labels'], data_k_means['pred_cluster_labels'])}")

print(f"Adjusted rand score: {metrics.adjusted_rand_score(data_k_means['original_labels'], data_k_means['pred_cluster_labels'])}")
print(f"Silhouette score: {metrics.silhouette_score(data_k_means, data_k_means['pred_cluster_labels'] ,metric='euclidean', sample_size=4898)}")
def select_k (data, k_range):

    data_ = data.copy()

    scores = dict()

    for k in k_range:    

        kmeans = KMeans(n_clusters=k).fit(data_)

        labels = (kmeans.labels_ + 1).tolist()

        data_[f'pred_{k}_cluster_labels'] = (kmeans.labels_ + 1).tolist()

        residual_error = 0

        print(f'# k = {k} =>')

        homogeneity_residual_error = (1 - metrics.homogeneity_score(data_['original_labels'], labels))

        print(f"- Homogeneity score: {homogeneity_residual_error}")

        completeness_residual_error = (1 - metrics.completeness_score(data_['original_labels'], labels))

        print(f"- Completeness score: {completeness_residual_error}")

        adjusted_rand_residual_error = (1 - metrics.adjusted_rand_score(data_['original_labels'], labels))

        print(f"- Adjusted rand score: {adjusted_rand_residual_error}")

        silhouette_residual_error = (1 - metrics.silhouette_score(data, labels, metric='euclidean', sample_size=data.shape[0]))

        print(f"- Silhouette score: {silhouette_residual_error}")

        scores[f'k_{k}'] = silhouette_residual_error + adjusted_rand_residual_error + completeness_residual_error + homogeneity_residual_error

        print(f'Total error = {scores[f"k_{k}"]}')

    lists = sorted(scores.items()) 

    x, y = zip(*lists) 

    plt.plot(x, y)

    plt.show()

    return scores, data_



scores_k_means, data_k_means = select_k(data_k_means, range(2,10))

# Quitamos el que ya había de k = 5

if 'pred_cluster_labels' in data.columns:

    del data['pred_cluster_labels']

data_k_means.head()
d = pd.concat([data_k_means['fixed acidity'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="fixed acidity", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['citric acid'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="citric acid", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['citric acid'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="citric acid", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['chlorides'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="chlorides", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['free sulfur dioxide'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="free sulfur dioxide", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['total sulfur dioxide'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="total sulfur dioxide", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['density'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="density", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['pH'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="pH", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['sulphates'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="sulphates", data=d)

plt.xticks(rotation=90);
d = pd.concat([data_k_means['alcohol'], data_k_means['pred_5_cluster_labels']], axis=1)

f, ax = plt.subplots()

fig = sns.boxplot(x='pred_5_cluster_labels', y="alcohol", data=d)

plt.xticks(rotation=90);
from scipy.cluster.hierarchy import dendrogram, linkage

data_aggl_hier = data.copy()



# Creamos la matriz de linkage, pasándo como parámetros los datos y el criterio para escoger la distancia mínima

H_cluster = linkage(data_aggl_hier,'ward') # ward es el caso c) Distancia entre los centroides de dos grupos.



# Preparamos la representación gráfica

plt.title('Hierarchical Clustering Dendrogram (truncated)')

plt.xlabel('Clústeres')

plt.ylabel('Distancia')



# Representamos la algomeración en forma de digrama en árbol

dendrogram(

    H_cluster, # jerarquía de distancias mínimas y clústeres calculado

    leaf_rotation=90., # Diseño representación

    leaf_font_size=12., # Diseño represetanción

    show_contracted=True, 

    orientation='right'

)

plt.show()
from sklearn.cluster import AgglomerativeClustering



cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

labels = cluster.fit_predict(data_aggl_hier)
data_aggl_hier['pred_cluster_labels'] = (labels + 1).tolist()

data_aggl_hier.head(3)
fig = px.pie(data_aggl_hier, values='pred_cluster_labels', names='pred_cluster_labels')

fig.show()
def select_n (data, n_range):

    data_ = data.copy()

    scores = dict()

    for n in n_range:    

        cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')

        labels = cluster.fit_predict(data_aggl_hier)

        data_[f'pred_{n}_cluster_labels'] = (labels + 1).tolist()

        residual_error = 0

        print(f'# n = {n} =>')

        homogeneity_residual_error = (1 - metrics.homogeneity_score(data_['original_labels'], labels))

        print(f"- Homogeneity score: {homogeneity_residual_error}")

        completeness_residual_error = (1 - metrics.completeness_score(data_['original_labels'], labels))

        print(f"- Completeness score: {completeness_residual_error}")

        adjusted_rand_residual_error = (1 - metrics.adjusted_rand_score(data_['original_labels'], labels))

        print(f"- Adjusted rand score: {adjusted_rand_residual_error}")

        silhouette_residual_error = (1 - metrics.silhouette_score(data, labels, metric='euclidean', sample_size=data.shape[0]))

        print(f"- Silhouette score: {silhouette_residual_error}")

        scores[f'n_{n}'] = silhouette_residual_error + adjusted_rand_residual_error + completeness_residual_error + homogeneity_residual_error

        print(f'Total error = {scores[f"n_{n}"]}')

    lists = sorted(scores.items()) 

    x, y = zip(*lists) 

    plt.plot(x, y)

    plt.show()

    return scores, data_



scores_aggl_hier, data_aggl_hier = select_n(data_aggl_hier, range(2,15))

# Quitamos el que ya había de n = 5

if 'pred_cluster_labels' in data.columns:

    del data['pred_cluster_labels']

data_aggl_hier.head()
algorithms = ['K-Means', 'Agglomeration']

scores = [scores_k_means['k_5'], scores_aggl_hier['n_5']]

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(algorithms, scores)

ax.set_ylabel('Error en la agrupación')

plt.show()
from sklearn.cluster import MeanShift, estimate_bandwidth



data_mean_shif = data.copy()



# The following bandwidth can be automatically detected using

bandwidth = estimate_bandwidth(data_mean_shif, quantile=0.1)

mean_shif = MeanShift(bandwidth).fit(data_mean_shif)



data_mean_shif['pred_cluster_labels'] = (mean_shif.labels_ + 1).tolist()

data_mean_shif.head()
fig = px.pie(data_mean_shif, values='pred_cluster_labels', names='pred_cluster_labels')

fig.show()
def select_q (data, q_range):

    data_ = data.copy()

    scores = dict()

    for q in q_range:    

        bandwidth = estimate_bandwidth(data_mean_shif, quantile=q)

        mean_shif = MeanShift(bandwidth).fit(data_mean_shif)

        labels = (mean_shif.labels_ + 1).tolist()

        data_['pred_cluster_labels'] = labels

        residual_error = 0

        print(f'# q = {q} =>')

        homogeneity_residual_error = (1 - metrics.homogeneity_score(data_['original_labels'], labels))

        print(f"- Homogeneity score: {homogeneity_residual_error}")

        completeness_residual_error = (1 - metrics.completeness_score(data_['original_labels'], labels))

        print(f"- Completeness score: {completeness_residual_error}")

        adjusted_rand_residual_error = (1 - metrics.adjusted_rand_score(data_['original_labels'], labels))

        print(f"- Adjusted rand score: {adjusted_rand_residual_error}")

        silhouette_residual_error = (1 - metrics.silhouette_score(data, labels, metric='euclidean', sample_size=data.shape[0]))

        print(f"- Silhouette score: {silhouette_residual_error}")

        scores[f'q_{q}'] = silhouette_residual_error + adjusted_rand_residual_error + completeness_residual_error + homogeneity_residual_error

        print(f'Total error = {scores[f"q_{q}"]}')

    lists = sorted(scores.items()) 

    x, y = zip(*lists) 

    plt.plot(x, y)

    plt.show()

    return scores, data_



scores_mean_shif, data_mean_shif = select_q(data_mean_shif, [0.1,0.2,0.3,0.4])

# Quitamos el que ya había de q = 0.1

if 'pred_cluster_labels' in data.columns:

    del data['pred_cluster_labels']

data_mean_shif.head()
algorithms = ['K-Means', 'Agglomeration', 'Mean Shift']

scores = [scores_k_means['k_5'], scores_aggl_hier['n_5'], scores_mean_shif['q_0.2']]

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(algorithms, scores)

ax.set_ylabel('Error en la agrupación')

plt.show()