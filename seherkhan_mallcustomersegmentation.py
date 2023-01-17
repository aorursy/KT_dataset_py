import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from sklearn.preprocessing import StandardScaler

import seaborn as sns; sns.set()  # for plot styling

%matplotlib inline

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')

from sklearn.cluster import KMeans

import os

import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.metrics import silhouette_score





#print(os.listdir("../input"))
dataset=pd.read_csv('../input/Mall_Customers.csv')

pd.set_option('display.max_columns', 10)

print(dataset.keys())

print(len(dataset))

print(dataset.head())
dataset.describe().transpose()
print(dataset['Gender'].unique())

dataset['Gender_code'] = np.where(dataset['Gender']=='Male', 1,0)
scaler = StandardScaler()

df = pd.DataFrame(scaler.fit_transform(dataset[["Age","Annual Income (k$)","Spending Score (1-100)"]]))

df.columns = ["age","income","spending"]

df.insert(0, "gender", dataset["Gender_code"])

df.head()
# Histograms

plot_gender = sns.distplot(df["gender"], label="gender",color="grey")

plot_age = sns.distplot(df["age"], label="age",color="blue")

plot_income = sns.distplot(df["income"], label="income",color="lightgreen")

plot_spend = sns.distplot(df["spending"], label="spend",color="orange")

plt.xlabel('')

plt.legend()

plt.show()
# Violin plot

f, axes = plt.subplots(2,2, figsize=(12,6), sharex=True, sharey=True)

v1 = sns.violinplot(data=df, x="gender", color="gray",ax=axes[0,0])

v2 = sns.violinplot(data=df, x="age", color="skyblue",ax=axes[0,1])

v3 = sns.violinplot(data=df, x="income",color="lightgreen", ax=axes[1,0])

v4 = sns.violinplot(data=df, x="spending",color="pink", ax=axes[1,1])
wcss = []

k_s = [i*i for i in range(1,8)]

print(k_s)

for i in k_s:

    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)

    km.fit(df)

    wcss.append(km.inertia_)

plt.plot(k_s,wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()
wcss = []

k_s = [4,7,9]

print(k_s)

for i in k_s:

    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)

    km.fit(df)

    wcss.append(km.inertia_)

plt.plot(k_s,wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()
wcss = []

k_s = [4,6,7]

print(k_s)

for i in k_s:

    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)

    km.fit(df)

    wcss.append(km.inertia_)

plt.plot(k_s,wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()
wcss = []

k_s = [4,5,6]

print(k_s)

for i in k_s:

    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)

    km.fit(df)

    wcss.append(km.inertia_)

plt.plot(k_s,wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()
km2 = KMeans(n_clusters=2,init='k-means++', max_iter=300, n_init=10, random_state=0)

labels = km2.fit_predict(df[["gender","age","income","spending"]])

df2 = df

df2["labels"] = labels

sns.pairplot(df2,hue="labels",vars=["gender","age","income","spending"])

ss = silhouette_score(df2[["gender","age","income","spending"]], df2["labels"], metric="euclidean")

print(f"Silhouette Score = {ss:.3f}")
km3 = KMeans(n_clusters=3,init='k-means++', max_iter=300, n_init=10, random_state=0)

labels = km3.fit_predict(df[["age","income","spending"]])

df3 = df

df3["labels"] = labels

sns.pairplot(df3,hue="labels",vars=["age","income","spending"])

ss = silhouette_score(df3[["age","income","spending"]], df3["labels"], metric="euclidean")

print(f"Silhouette Score = {ss:.3f}")
km4 = KMeans(n_clusters=4,init='k-means++', max_iter=300, n_init=10, random_state=0)

labels = km4.fit_predict(df[["age","income","spending"]])

df4 = df

df4["labels"] = labels

sns.pairplot(df4,hue="labels",vars=["age","income","spending"])

ss = silhouette_score(df4[["age","income","spending"]], df4["labels"], metric="euclidean")

print(f"Silhouette Score = {ss:.3f}")
km5 = KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10, random_state=0)

labels = km5.fit_predict(df[["age","income","spending"]])

df5 = df

df5["labels"] = labels

sns.pairplot(df5,hue="labels",vars=["age","income","spending"])

ss = silhouette_score(df5[["age","income","spending"]], df5["labels"], metric="euclidean")

print(f"Silhouette Score = {ss:.3f}")

sns.pairplot(df5,hue="gender",vars=["age","income","spending"])
dendrogram = dendrogram(linkage(df[["gender","age","income","spending"]], method='ward'))
ward = AgglomerativeClustering(n_clusters=7,linkage='ward').fit(df)

df_ward = df

df_ward["labels"]=label

label = ward.labels_

# print(np.unique(df_ward["labels"]))



ss = silhouette_score(df_ward[["gender","age","income","spending"]], df_ward["labels"], metric="euclidean")

print(f"Silhouette Score = {ss:.3f}")



sns.pairplot(df_ward,hue="labels",vars=["gender","age","income","spending"])