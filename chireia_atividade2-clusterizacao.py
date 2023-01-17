import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans
df = pd.read_csv('../input/Iris.csv')

df.head()
df.describe()
df.plot.scatter(x='PetalLengthCm', y='PetalWidthCm')
df.plot.hist(y='PetalLengthCm')

df.plot.hist(y='PetalWidthCm')
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df[['PetalLengthCm','PetalWidthCm']].values)

labels = kmeans.predict(df[['PetalLengthCm','PetalWidthCm']].values)

C = kmeans.cluster_centers_

print(labels,C)
dfPetalGrupo = pd.concat([df,pd.DataFrame(labels, columns = ['Grupo'])], axis=1, join='inner')

cores = dfPetalGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfPetalGrupo.plot.scatter(x='PetalLengthCm',y='PetalWidthCm', c=cores)
cores = dfPetalGrupo.Species.map({'Iris-setosa':'b','Iris-versicolor':'r','Iris-virginica':'y'})

dfPetalGrupo.plot.scatter(x='PetalLengthCm',y='PetalWidthCm', c=cores)