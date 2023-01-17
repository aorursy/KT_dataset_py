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

        file = os.path.join(dirname, filename)

        print(file)



# Any results you write to the current directory are saved as output.
ds = pd.read_csv(file)

print(ds.columns)

print(ds.info())

print(ds.isnull().sum())

print(ds.describe())
print(ds["Gender"].unique())



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

ds["enc_gender"] = le.fit_transform(ds["Gender"])

'''Get gender integer label mapping'''

mapping = {l: i for i, l in enumerate(le.classes_)}

print(mapping)

print(ds.head())



'''Drop previous Gender column and rename other columns for better access'''

ds.drop(["Gender"], axis=1, inplace=True)

ds.rename(columns={'Age': 'age', 'Annual Income (k$)': 'annual_income_k', 'Spending Score (1-100)':'spending_score','CustomerID':'customer_id' }, inplace=True)

print(ds.head())

import seaborn as sns

import matplotlib.pyplot as plt



f = plt.figure(figsize=(8,5))

ax = f.add_subplot(121)

sns.distplot(ds["enc_gender"], kde=False)



ax = f.add_subplot(122)

sns.distplot(ds["age"])



f = plt.figure(figsize=(8,5))



ax = f.add_subplot(121)

sns.distplot(ds["annual_income_k"])



ax = f.add_subplot(122)

sns.distplot(ds["spending_score"])
f, ax = plt.subplots(figsize=(8, 6))

corr = ds.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),square=True, ax=ax)
sns.pairplot(ds)
sns.relplot(x="age", y="spending_score", hue="enc_gender", data=ds)

sns.relplot(x="annual_income_k", y="spending_score", hue="enc_gender", data=ds)
ds.drop(["customer_id"], axis=1, inplace=True)
ds.head()
from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt





distortions = []

silhouette_sc = []

K = range(1,10)

for k in K:

    model = KMeans(n_clusters=k, random_state=15)

    model.fit(ds)

    distortions.append(sum(np.min(cdist(ds, model.cluster_centers_, 'euclidean'), axis=1)) / ds.shape[0])

    if k>=2:

        silhouette_sc.append(metrics.silhouette_score(ds, model.labels_))



plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()



plt.plot(range(2,10), silhouette_sc, 'bx-')

plt.xlabel('k')

plt.ylabel('Silhouette Score')

plt.title('Silhouette Score')

plt.show()



model = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=15)

y_pred = model.fit_predict(ds)

ds['cluster'] = y_pred

ds.head()
sns.scatterplot(x="annual_income_k", y="spending_score",hue="cluster", legend="full", data=ds)

sns.scatterplot(x=model.cluster_centers_[:,1], y=model.cluster_centers_[:,2], color='r')
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

ds_features = ds.drop(["cluster"], axis=1)

ds_reduced = pca.fit_transform(ds_features)

ds2 = pd.DataFrame(ds_reduced, columns=["C1", "C2"])

ds2["clusters"] = pd.DataFrame(y_pred, columns=["clusters"])

sns.scatterplot(x="C1", y="C2",hue="clusters", legend="full", data=ds2)
model = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=15)

y_pred = model.fit_predict(ds)

ds['cluster'] = y_pred

ds.head()
sns.scatterplot(x="annual_income_k", y="spending_score",hue="cluster", legend="full", data=ds)

sns.scatterplot(x=model.cluster_centers_[:,1], y=model.cluster_centers_[:,2], color='r')
model.cluster_centers_
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

ds_features = ds.drop(["cluster"], axis=1)

ds_reduced = pca.fit_transform(ds_features)
ds_reduced


ds2 = pd.DataFrame(ds_reduced, columns=["C1", "C2"])

ds2["clusters"] = pd.DataFrame(y_pred, columns=["clusters"])

ds2
sns.scatterplot(x="C1", y="C2",hue="clusters", legend="full", data=ds2)