import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

import seaborn as sns #Biblioteca de visualização poderosa



import os

print(os.listdir("../input"))


df = pd.read_csv('../input/zoo.csv')

df.drop(["animal_name"], axis = 1, inplace = True)  #remove a coluna animal_name

df.head()
df.describe()

sns.pairplot(data=df)
inertia = []

for i in range(1,10):

    kmeans = KMeans(n_clusters=i, random_state=1234)

    kmeans.fit(df)

    inertia.append((i,kmeans.inertia_,))

    

plt.plot([w[0] for w in inertia],[w[1] for w in inertia], marker="X")
clusters = 7



kmeans = KMeans(n_clusters=clusters)

kmeans = kmeans.fit(df)

labels = kmeans.predict(df)

C_center = kmeans.cluster_centers_

print(labels,"\n",C_center)
dfGroup = pd.concat([df,pd.DataFrame(labels, columns= ['Group'])], axis=1, join='inner')

dfGroup.head()
dfGroup.groupby("Group").aggregate("mean").plot.bar()
sns.pairplot(data=dfGroup, hue='Group')