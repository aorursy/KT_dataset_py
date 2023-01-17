import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as snsA

from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
df=pd.read_csv("vehicle.csv")
df.info()
newdf=df.dropna()
newdf['class'].value_counts()
newdf.info()
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

for i in newdf.columns:

    newdf[i]=sc.fit_transform(newdf[[i]])
newdf.head()
distortion=np.array(range(1,15))
distortion
newdf1=newdf.drop('class',axis=1)
#finding the best n_clusters value

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

cluster_range=range(1,15)

cluster_inertia=[]

for num_clusters in cluster_range:

    model=KMeans(num_clusters)

    model.fit(newdf1)

    cluster_inertia.append(model.inertia_)

plt.figure(figsize=(12,6))

plt.plot(cluster_range,cluster_inertia,marker='o')

plt.xlabel("no of clusters")

plt.ylabel("interia")
#n_clusters=2

k3=KMeans(n_clusters=3,n_init=15,random_state=2)

k3.fit(newdf1)
cent=pd.DataFrame(k3.cluster_centers_,columns=newdf1.columns)

cent
pd.DataFrame(k3.labels_)[0].value_counts()
newdf[newdf['class']=='car']['compactness'].mean()


newdf[newdf['class']=='van']['compactness'].mean()

newdf[newdf['class']=='bus']['compactness'].mean()
import seaborn as sns
sns.pairplot(df,hue='class',palette='husl')
df_with_label=newdf1.copy(deep=True)

df_with_label['labels']=k3.labels_
df_with_label