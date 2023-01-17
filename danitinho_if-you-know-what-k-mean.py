import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

import seaborn as sns



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/BO_2016.csv')
df[['ID_DELEGACIA','MES','RUBRICA']].head()
df[['ID_DELEGACIA','MES','RUBRICA']].describe()
sns.pairplot(data=df[['ID_DELEGACIA','MES']])
clusters = 10



kmeans = KMeans(n_clusters=clusters)

kmeans = kmeans.fit(df[['ID_DELEGACIA','MES']])

labels = kmeans.predict(df[['ID_DELEGACIA','MES']])

C_center = kmeans.cluster_centers_

print(labels,"\n",C_center)
dfGroup = pd.concat([df[['ID_DELEGACIA','MES']],pd.DataFrame(labels, columns= ['Group'])], axis=1, join='inner')

dfGroup.head()
dfGroup.groupby("Group").aggregate("mean").plot.bar()