import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/cars.csv',na_values=' ') #To replace empty objects with NaN use attribute na_values=' '
data.head()
data.shape
sns.heatmap(data=data.notnull(),cmap='rainbow') #To visuvalize NaN values in the entire dataset
x=data.iloc[:,0:7].values
#Handling Missing Values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,[2,4]])
x[:,[2,4]]=imputer.transform(x[:,[2,4]])
from sklearn.cluster import KMeans
ssc=[]
for i in range(1,10):
    km = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    km.fit(x)
    ssc.append(km.inertia_)
plt.plot(range(1,10),ssc)
plt.show()
    
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means_pred=kmeans.fit_predict(x)
plt.scatter(x[y_means_pred==0,0],x[y_means_pred==0,1],s=100,c='red',label='US')
plt.scatter(x[y_means_pred==1,0],x[y_means_pred==1,1],s=100,c='blue',label='Japan')
plt.scatter(x[y_means_pred==2,0],x[y_means_pred==2,1],s=100,c='green',label='Europe')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centers')
plt.legend()
plt.xlabel('Clusters of car brands')