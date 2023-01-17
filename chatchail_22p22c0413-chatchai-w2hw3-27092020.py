#3. (Clustering) ใช้ Kaggle เพื่อทำการวิเคราะห์ข้อมูลการเข้าพักห้องพักของ Airbnb ที่นิวยอร์กซิตี (New York City Airbnb Open Data) https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
#3.4 ทำการสร้างตัวจัดกลุ่มข้อมูลแบบ hierarchical clustering (https://en.wikipedia.org/wiki/Complete-linkage_clustering)
#3.6 ให้แสดงผลเป็นกราฟ dendrogram
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import folium
from folium.plugins import MarkerCluster

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.tail()
df.info()
df['neighbourhood_group'].value_counts()
# shared room is the least popular amongst all types
fig = plt.subplots(figsize = (10,6))
sns.countplot( data = df, x = 'room_type', hue = 'neighbourhood_group')
# manhattan price spreaded more
plt.figure(figsize = (14,8))
lower_bar = df[df['price'] < 500]
sns.violinplot(data=lower_bar, x = 'neighbourhood_group', y = 'price')
# map of AirBNB listed as per neightbourhood_group
# to validate data correctness
plt.figure(figsize=(10,10))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='neighbourhood_group',s=40)
# correlation
# most unrelated
df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(df.corr(), cmap='coolwarm')
# there are some room available more than 365 days a year
# so, better to remove them in this case
df['availability_365'].value_counts().sort_values
df = df[df['availability_365']<366]
# some rooms are super luxury
# so, we only target lower than 2000 in this case
df = df[df['price']<2000]
# 48k rows might be too big
# thus, i randomly select 5k rows
df = df.sample(2000)
# here are three clustering factors
data = df[['price','number_of_reviews','availability_365','latitude','longitude']]
data = np.array(data)
data
### Create Dendrogram
fig = plt.figure(figsize=(20,16))
# ward = minimum variance
dendrogram = sch.dendrogram(sch.linkage(data[:,:3], method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Listing')
plt.ylabel('Euclidean Distance')
plt.Text(0, 0.5, 'Euclidean Distance')
#Chooses 3 clusters as from Dendogram
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage='ward')
y_hc = hc.fit_predict(data)
y_hc
X1 = np.array(data[y_hc==0])[:,0]
Y1 = np.array(data[y_hc==0])[:,2]
X2 = np.array(data[y_hc==1])[:,0]
Y2 = np.array(data[y_hc==1])[:,2]
X3 = np.array(data[y_hc==2])[:,0]
Y3 = np.array(data[y_hc==2])[:,2]
fig = plt.figure(figsize=(14,10))
plt.scatter(X1,Y1,s=15,c='green',label='Cluster1')
plt.scatter(X2,Y2,s=15,c='blue',label='Cluster2')
plt.scatter(X3,Y3,s=15,c='red',label='Cluster3')
plt.title('3 Clusters')
plt.xlabel('Price')
plt.ylabel('availability_365')
plt.legend()
plt.grid()
## the red ones are expensive with low number of reviews (maybe unreliable?)
## the green ones are cheap and often not available
## the blue ones are cheap, often available throughout the year
X1 = np.array(data[y_hc==0])[:,3]
Y1 = np.array(data[y_hc==0])[:,4]
X2 = np.array(data[y_hc==1])[:,3]
Y2 = np.array(data[y_hc==1])[:,4]
X3 = np.array(data[y_hc==2])[:,3]
Y3 = np.array(data[y_hc==2])[:,4]
# plot with lat lon
fig = plt.figure(figsize=(12,8))
plt.scatter(X1,Y1,s=35,c='green',label='Cluster1')
plt.scatter(X2,Y2,s=35,c='blue',label='Cluster2')
plt.scatter(X3,Y3,s=35,c='red',label='Cluster3')
plt.title('3 Clusters')
plt.xlabel('lat')
plt.ylabel('lon')
plt.legend()
plt.grid()
# using real map from folium
cluster_map = folium.Map(location=[40.74870,-73.88470],tiles='OpenStreetMap',zoom_start=10.5) 

for i in range(len(y_hc)):
    lat = data[i,3]
    long = data[i,4]
    radius = 2

    if y_hc[i] == 0:
        folium.CircleMarker(location=[lat, long],radius=radius,fill=True,color='green').add_to(cluster_map)
    elif y_hc[i] == 1:
        folium.CircleMarker(location=[lat, long],radius=radius,fill=True,color='blue').add_to(cluster_map)
    else:
        folium.CircleMarker(location=[lat, long],radius=radius,fill=True,color='red').add_to(cluster_map)

cluster_map