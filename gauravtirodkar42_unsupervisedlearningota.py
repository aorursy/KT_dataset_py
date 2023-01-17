import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import sklearn
!unzip /kaggle/input/expedia-personalized-sort/data.zip
train = pd.read_csv('train.csv')
train.head()
train.isnull().sum()
df = train.loc[train['prop_id'] == 104517]

df = df.loc[df['visitor_location_country_id'] == 219]

df = df.loc[df['srch_room_count'] == 1]

df = df[['date_time', 'price_usd', 'srch_booking_window', 'srch_saturday_night_bool']]
df.head()
df['price_usd'].describe()
df = df.loc[df['price_usd']<5584]
df.plot(x='date_time',y='price_usd',figsize=(20,5))
plt.xlabel('Date')
plt.ylabel('Price')
plt.title("Time Series Graph Of Price Of Room Based on Date-Time")
a = df.loc[df['srch_saturday_night_bool']==0,'price_usd']
b = df.loc[df['srch_saturday_night_bool']==1,'price_usd']

plt.figure(figsize=(20,10))
plt.hist(a,bins=50, alpha=0.5,label='Saturday Night')
plt.hist(b,bins=50, alpha=0.5,label='Non-Saturday Night')
plt.legend(loc='upper right')
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Price Comparison between Non-Sat and Saturday')
from sklearn.cluster import KMeans
train_data = df[['price_usd','srch_booking_window','srch_saturday_night_bool']]
n_cluster = range(5,25)
kmeans = [KMeans(n_clusters=i).fit(train_data) for i in n_cluster]
scores = [kmeans[i].score(train_data) for i in range(len(kmeans))]

fig,ax = plt.subplots(figsize=(20,6))
ax.plot(n_cluster,scores)
plt.xlabel('Number Of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
X = train_data.reset_index(drop=True)
km = KMeans(n_clusters = 15)
km.fit(X)
km.predict(X)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(12,12))
ax = Axes3D(fig, rect = [0,0,0.95,1],elev=45, azim=140)
labels = km.labels_
ax.scatter(X.iloc[:,0],X.iloc[:,1],X.iloc[:,2],c=labels.astype(np.float), edgecolor='r')
ax.set_xlabel('Price in USD')
ax.set_ylabel('Search booking window')
ax.set_zlabel('Search saturday night bool')
plt.title('KMeans CLustering for Anamoly Detecttion')
del train
import os
os.remove('train.csv')

