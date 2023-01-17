import numpy as np

import pandas as pd

import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv("../input/compresive_strength_concrete.csv")

display(data)
data.columns = ['cimento', 'escoria', 'cinzas', 'agua', 'superplastificante', 'ag_grosso', 'ag_fino', 'idade', 'resistencia']
data.describe()
df_numeric = data[['cimento', 'escoria', 'cinzas', 'agua', 'superplastificante', 'ag_grosso', 'ag_fino', 'idade', 'resistencia']]
df_numeric.head()
df_numeric.isnull().sum()

df_numeric.dropna(inplace=True)
df_numeric['resistencia'].describe()
df_numeric['resistencia'].quantile(np.arange(.40,1,0.01))
df_numeric.shape
from sklearn import preprocessing
minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('resistencia',axis=1))
df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])
df_numeric_scaled.head()
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]
pl.plot(Nc,score)

pl.xlabel('Número de Clusters')

pl.title('Erro Quadrático por número de Grupos (k)')

pl.show()
kmeans = KMeans(n_clusters=5)

kmeans.fit(df_numeric_scaled)
len(kmeans.labels_)
df_numeric['cluster'] = kmeans.labels_
plt.figure(figsize=(12,7))

axis = sns.barplot(x=np.arange(0,5,1),y=df_numeric.groupby(['cluster']).count()['resistencia'].values)

x=axis.set_xlabel("Clusters")

x=axis.set_ylabel("Score")
df_numeric.groupby(['cluster']).mean()
size_array = list(df_numeric.groupby(['cluster']).count()['resistencia'].values)
size_array
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[2])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[1])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[-1])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[-2])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[0])].sample(5)