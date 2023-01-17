from sklearn.cluster import KMeans

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('../input/banknote-authentication/1fXr31hcEemkYxLyQ1aU1g_50fc36ee697c4b158fe26ade3ec3bc24_Banknote-authentication-dataset-.csv')

df.head()
plt.scatter(df['V1'],df['V2'])
V1 = df['V1']

V2 = df['V2']

V1.mean()

V1 = df['V1']

V2 = df['V2']

V2.mean()

V1 = df['V1']

V2 = df['V2']

V1.std()

V1 = df['V1']

V2 = df['V2']

V2.std()

km = KMeans (n_clusters=3)
y_predicted = km.fit_predict(df[['V1','V2']])

y_predicted

df['cluster']= y_predicted

df.head()

km.cluster_centers_
df1 = df[df.cluster==0]

df2 = df[df.cluster==1]

df3 = df[df.cluster==2]

plt.scatter(df1.V1,df1['V2'], color ='yellow')

plt.scatter(df2.V1,df2['V2'], color ='cyan')

plt.scatter(df3.V1,df3['V2'], color ='red')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.show()

km = KMeans (n_clusters=2)
y_predicted = km.fit_predict(df[['V1','V2']])

y_predicted

df['cluster']= y_predicted

df.head

km.cluster_centers_

df1 = df[df.cluster==0]

df2 = df[df.cluster==1]

plt.scatter(df1.V1,df1['V2'], color ='red')

plt.scatter(df2.V1,df2['V2'], color ='green')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.show()
