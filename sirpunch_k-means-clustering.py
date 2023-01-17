import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/movies_metadata.csv")
df.drop(df.index[19730],inplace=True)
df.drop(df.index[29502],inplace=True)
df.drop(df.index[35585],inplace=True)
df_numeric = df[['budget','popularity','revenue','runtime','vote_average','vote_count','title']]
df_numeric.head()
df_numeric.isnull().sum()
df_numeric.dropna(inplace=True)
df_numeric['vote_count'].describe()
df_numeric['vote_count'].quantile(np.arange(.74,1,0.01))
df_numeric = df_numeric[df_numeric['vote_count']>30]
df_numeric.shape
from sklearn import preprocessing
minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title',axis=1))
df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])
df_numeric_scaled.head()
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_numeric_scaled)
len(kmeans.labels_)
df_numeric['cluster'] = kmeans.labels_
df_numeric.head()
plt.figure(figsize=(12,7))
axis = sns.barplot(x=np.arange(0,5,1),y=df_numeric.groupby(['cluster']).count()['budget'].values)
x=axis.set_xlabel("Cluster Number")
x=axis.set_ylabel("Number of movies")
df_numeric.groupby(['cluster']).mean()
size_array = list(df_numeric.groupby(['cluster']).count()['budget'].values)
size_array
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[0])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[1])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[-1])].sample(5)