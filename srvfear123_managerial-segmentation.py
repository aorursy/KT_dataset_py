import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

retail = pd.read_csv("/kaggle/input/purchases.txt",sep='\t',names=['custid','amount','date'],parse_dates=[2])
retail.head()
retail.info()
retail.shape
retail.isnull().sum()
import datetime as dt

retail['year'] = retail['date'].dt.year
retail.head()
retail.describe()
retail.amount.hist()
plt.boxplot(x = 'amount', data=retail)
plt.show()
yearwise = retail.groupby('year').agg({'year':'count','amount':['sum','mean']})
display(yearwise.head())

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.bar(yearwise.index, yearwise['year','count'])
plt.ylabel("No. of transactions")

plt.subplot(1,3,2)
plt.bar(yearwise.index, yearwise['amount','mean'])
plt.ylabel("Avg. amount")

plt.subplot(1,3,3)
plt.bar(yearwise.index, yearwise['amount','sum'])
plt.ylabel("sum of amount")
plt.tight_layout()
plt.show()
abv = retail.groupby('custid').agg({'amount':'mean'})
abv.head()
abv.describe()
retail.date.describe()
now = pd.to_datetime("2016-01-01")
now
retail['date2'] = retail['date']

dict = {'date':lambda x:(now-x.max()).days,'date2':lambda x:(now-x.min()).days, 'custid':'count', 'amount':'mean'}
rffm = retail.groupby('custid').agg(dict)

rffm.rename(columns={'date':'recency','date2':'firstpurchase','custid':'frequency','amount':'monetary'}, inplace=True)
rffm.head()
rffm.info()
rffm.describe()
cols = ['recency', 'frequency', 'monetary']

rfm = rffm.loc[:,cols]
rfm.head()
rfm.hist()
plt.tight_layout()
plt.show()
rfm_log = rfm.copy()
rfm_log['monetary'] = rfm_log['monetary'].apply(np.log)
rfm_log.head()
rfm_log.hist()
plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_log_scaled = scaler.fit_transform(rfm_log)
rfm_log_scaled[:5]

rfm_scaled = scaler.fit_transform(rfm)
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples,silhouette_score
from mpl_toolkits.mplot3d import Axes3D

k = 4
kmeans = KMeans(n_clusters=k, n_init=40, random_state=1234)
kmeans.fit(rfm_scaled)
cluster_labels = kmeans.labels_
rfm_km = rfm_log.assign(K_Cluster = cluster_labels)
rfm_km.head()
rfm_km.groupby('K_Cluster').agg({'recency':'mean','frequency':'mean','monetary':['mean','count']})
from mpl_toolkits.mplot3d import Axes3D

cmap = cm.get_cmap("Accent")
colors=cmap(cluster_labels.astype(float)/k)

fig=plt.figure(figsize=(12,10))
ax=fig.add_subplot(111,projection='3d')
x=rfm_km.recency
y=rfm_km.frequency
z=rfm_km.monetary

ax.scatter(x,y,z,alpha=0.3, c=rfm_km.K_Cluster)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.show()


N = len(rfm)
n = 1841
k = round(N/n)
num = np.random.randint(1,k,)
print(num)

rfm_subset = rfm.iloc[num:N:10,:]
len(rfm_subset)
rfm_subset.head()
rfm_subset.hist()
plt.show()
rfm_subset_log = rfm_subset.copy()
rfm_subset_log['monetary'] = rfm_subset_log['monetary'].apply(np.log)
rfm_subset_log.head()
rfm_subset_log.describe()
rfm_subset_log.hist()
plt.tight_layout()
plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_subset_log_scaled = scaler.fit_transform(rfm_subset_log)
rfm_subset_log_scaled[:5]
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10,7))
plt.title('Dendrograms')
dend = shc.dendrogram(shc.linkage(rfm_subset_log_scaled, method='ward'))
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(rfm_subset_log_scaled, method='ward'))
plt.axhline(y=30, color='k', linestyle='--')
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(rfm_subset_log_scaled)
cluster_labels = cluster.labels_
rfm_subset_log.loc[:,'HC_cluster'] = cluster_labels
rfm_subset_log.head()
rfm_subset_log.groupby('HC_cluster').agg({'recency':'mean','frequency':'mean','monetary':['mean','count']})
#plt.figure(figsize=(10, 7))  
#sns.scatterplot(x=rfm_subset.recency, y=rfm_subset.frequency, hue=rfm_subset.HC_cluster, alpha=0.5, palette='viridis')

from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(12,10))
ax=fig.add_subplot(111,projection='3d')
x=rfm_subset_log.recency
y=rfm_subset_log.frequency
z=rfm_subset_log.monetary

ax.scatter(x,y,z,c=rfm_subset_log.HC_cluster)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.show()
