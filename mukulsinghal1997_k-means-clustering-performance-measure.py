#importing necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

from sklearn.cluster import KMeans
df = pd.read_csv("../input/wine-pca/Wine.csv")
df.head()
df.Customer_Segment.unique()
df.drop(labels = "Customer_Segment", axis = 1, inplace = True)
df.isnull().any().sum()
df.shape
fig, ax = plt.subplots(5,3, figsize=(14,12))

axes_ = [axes_row for axes in ax for axes_row in axes]

for i,c in enumerate(df.columns):

    sns.distplot(df[c], ax = axes_[i], color = 'orange')

    plt.tight_layout()
fig, ax = plt.subplots(5,3, figsize=(14,12))

axes_ = [axes_row for axes in ax for axes_row in axes]

for i,c in enumerate(df.columns):

    sns.boxplot(df[c], ax = axes_[i], color = 'skyblue')

    plt.tight_layout()
fig, ax = plt.subplots(5,3, figsize=(14,12))

axes_ = [axes_row for axes in ax for axes_row in axes]

for i,c in enumerate(df.columns):

    sns.scatterplot(x = "Alcohol", y = df[c], data = df, ax = axes_[i])

    plt.tight_layout()
corr = df.corr()
plt.figure(figsize=(10,8))

sns.heatmap(corr, annot = True, cmap="Blues")
threshold = 0.6
corr[corr > threshold]
col = ["Proline", "Flavanoids", "Proanthocyanins", "OD280"]
df_new = df
df_new.drop(labels = col, axis = 1, inplace = True)
from sklearn.preprocessing import StandardScaler
se = StandardScaler()
col_n = df_new.columns
se_ = se.fit_transform(df_new)
df_ = pd.DataFrame(se_, columns = col_n)
np.random.seed(42)
clusters = range(2, 16)
inertia = []

sil_score = []

calinski_score =  []

davies_score=  []
for i in clusters:

    kmeans_mod = KMeans(n_clusters = i, init = "k-means++", n_jobs = -1).fit(df_)

    inertia.append(kmeans_mod.inertia_)

    s_score = silhouette_score(df_, kmeans_mod.labels_)

    cal_score = calinski_harabasz_score(df_, kmeans_mod.labels_)

    dav_score = davies_bouldin_score(df_, kmeans_mod.labels_)

    sil_score.append(s_score)

    calinski_score.append(cal_score)

    davies_score.append(dav_score)
plt.figure(figsize=(12,5))

plt.subplot(121)

plt.plot(clusters, inertia, marker = 'o', linestyle = '--')

plt.xlabel("Clusters")

plt.ylabel("Inertia")

plt.title("Clusters v/s Inertia")



plt.subplot(122)

plt.plot(clusters, sil_score, marker = 'o', linestyle = '--', color = 'r')

plt.xlabel("Clusters")

plt.ylabel("Silhouette Score")

plt.title("Clusters v/s Silhouette Score")



plt.tight_layout()
plt.figure(figsize=(12,6))



plt.subplot(121)

plt.plot(clusters, calinski_score, marker = 'o', linestyle = '--', color = 'g')

plt.xlabel("Clusters")

plt.ylabel("Calsinki Score")

plt.title("Clusters v/s Calinski Score")



plt.subplot(122)

plt.plot(clusters, davies_score, marker = 'o', linestyle = '--', color = 'orange')

plt.xlabel("Clusters")

plt.ylabel("Davies Score")

plt.title("Clusters v/s Davies Score")



plt.tight_layout()
k_model = KMeans(n_clusters = 3, init = "k-means++", n_jobs = -1 )
label_predict = k_model.fit_predict(df_)
centers = k_model.cluster_centers_
label_df = pd.DataFrame(label_predict, columns = ["Label"])
df_ = pd.concat([df_, label_df], axis = 1)
df_.head()
mapping = {0: 1, 1: 2, 2: 3}
df_["Label"] = df_["Label"].map(mapping)
df_['Label'].value_counts()
df_['Label'].value_counts().plot(kind="bar")