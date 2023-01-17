from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
%matplotlib inline


'../input/students2020/studentdata1.csv'
df = pd.read_csv('../input/students2020/studentdata1.csv') 
df.head()
plt.scatter(df['Points'], df['Time'])
km = KMeans(n_clusters=4)
km
y_predicted = km.fit_predict(df[['Time', 'Points']])
y_predicted
df['cluster'] = y_predicted
df.head()
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]

plt.scatter(df1['Time'], df1['Points'], color='green')
plt.scatter(df2['Time'], df2['Points'], color='red')
plt.scatter(df3['Time'], df3['Points'], color='black')
plt.scatter(df4['Time'], df4['Points'], color='orange')

plt.xlabel('Time')
plt.ylabel('Points')
plt.legend()
df[['Points', 'Time']].to_numpy
scaler = MinMaxScaler()
scaler.fit(df[['Points', 'Time']])
array_points = df[['Points', 'Time']].to_numpy().reshape(-1, 2)
data = scaler.transform(array_points)
data
dfdata = pd.DataFrame(data)
dfdata.columns=['Points', 'Time']
km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(dfdata[['Time', 'Points']])
dfdata['cluster'] = y_predicted
km.cluster_centers_
df1 = dfdata[dfdata.cluster==0]
df2 = dfdata[dfdata.cluster==1]
df3 = dfdata[dfdata.cluster==2]
df4 = dfdata[dfdata.cluster==3]

plt.scatter(df1['Time'], df1['Points'], color='green')
plt.scatter(df2['Time'], df2['Points'], color='red')
plt.scatter(df3['Time'], df3['Points'], color='black')
plt.scatter(df4['Time'], df4['Points'], color='orange')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
