import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
mall=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
mall.head(5)
mall.isnull().sum()
mall.shape
mall.describe().T
mall.Gender.value_counts()
mall.dtypes
#find relation ship among the data sets

#percentage of mail and feail

sns.countplot(mall['Age'],palette = 'rainbow')

plt.title('Distrbution of the age',fontsize=15)

label=['Male','Female']

values=mall['Gender'].value_counts().values

colors=['blue','green']

fig,ax1=plt.subplots()

plt.axis('off')

explode = [0, 0.2]

ax1.pie(values,labels=label,shadow=True,startangle=90,autopct ='%.2f%%',explode = explode,colors=colors)

plt.title('Gender', fontsize =10)

plt.legend()

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(x='Gender',y='Age',data=mall)

plt.title('relation between age and gender')

plt.legend()
import warnings

warnings.filterwarnings('ignore')



sns.set(style='whitegrid')

plt.rcParams['figure.figsize'] = (14, 10)



plt.subplot(1,2,1)

sns.distplot(mall['Annual Income (k$)'],bins=100,color='y',vertical=True,rug=True)

plt.ylabel('anual income')

plt.xlabel('count')

plt.title('distrbution of annual income',fontsize=15)

plt.show()


sns.set(style='whitegrid')

plt.subplot(1,2,2)

sns.distplot(mall['Age'],bins=100,color='b',vertical=True,rug=True)

plt.ylabel('age of the person')

plt.xlabel('count')

plt.title('age of the persion',fontsize=15)

plt.legend()

plt.show()


sns.set(style='whitegrid')

plt.subplot(1,2,2)

sns.distplot(mall['Spending Score (1-100)'],color='g',bins=100,vertical=True,rug=True)

plt.ylabel('spending of the persion')

plt.xlabel('count')

plt.legend()

plt.show()
plt.rcParams['figure.figsize']=(14,10)

sns.countplot(mall['Annual Income (k$)'],palette='viridis_r')

#relation

plt.figure(figsize=(15,15))

sns.relplot(y='Age',x='Annual Income (k$)',hue='Gender',style='Gender',data=mall,s=100,label='Gender',palette='twilight_shifted')
sns.countplot(mall['Spending Score (1-100)'], palette='autumn')
sns.pairplot(mall.drop('CustomerID',axis=1))
sns.heatmap(mall.corr(),annot=True,cmap='mako')
sns.relplot(x='Annual Income (k$)',y='Spending Score (1-100)',hue='Gender',s=200,data=mall)

x=mall.iloc[:,[3,4]].values
#here we need standerd scaler to the scale the data

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

scale.fit(x)

scaled_x=scale.fit_transform(x)

scaled_x
inertia=[]

from sklearn.cluster import KMeans

for k in range(1,11):

    kmm=KMeans(n_clusters=k,random_state=0,max_iter=300,n_init = 10,init ='k-means++')

    kmm.fit(scaled_x)

    inertia.append(kmm.inertia_)



plt.plot(range(1,11), inertia)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('inertia')

plt.show()
print(kmm.cluster_centers_)

print(kmm.labels_)
from sklearn.cluster import KMeans

kmm=KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means=kmm.fit_predict(scaled_x)

print(y_means)



plt.scatter(scaled_x[y_means == 0, 0], scaled_x[y_means == 0, 1], s = 100, c = 'green', label = 'lower-class')

plt.scatter(scaled_x[y_means==1, 0], scaled_x[y_means==1, 1], s=100, c='g', label ='lowe-middle-class')

plt.scatter(scaled_x[y_means==2, 0], scaled_x[y_means==2, 1], s=100, c='r', label ='middle-class')

plt.scatter(scaled_x[y_means==3, 0], scaled_x[y_means==3, 1], s=100, c='y', label ='upper-class')

plt.scatter(scaled_x[y_means==4, 0], scaled_x[y_means==4, 1], s=100, c='orange', label ='high-class')

plt.scatter(kmm.cluster_centers_[:,0], kmm.cluster_centers_[:, 1], s = 200, c = 'cyan' , label = 'centeroid')

plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 15)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.show()

plt.show()
#age vs spending 

sns.relplot(x='Age',y='Spending Score (1-100)',hue='Gender',s=200,data=mall)
mall.head(1)

X = mall.iloc[:, [2, 4]].values

X
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

scale.fit(X)

scalex=scale.fit_transform(X)

scalex
inertia=[]

from sklearn.cluster import KMeans

for k in range(1,11):

    km=KMeans(n_clusters=k,random_state=0,max_iter=300,n_init = 10,init ='k-means++')

    km.fit(scalex)

    inertia.append(km.inertia_)



plt.plot(range(1,11), inertia)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('inertia')

plt.show()
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(scalex)

plt.scatter(scalex[y_means == 0, 0], scalex[y_means == 0, 1], s = 100, c = 'green', label = 'usualcustomer')

plt.scatter(scalex[y_means == 1, 0], scalex[y_means == 1, 1], s = 100, c = 'b', label = 'perority customer')

plt.scatter(scalex[y_means == 2, 0], scalex[y_means == 2, 1], s = 100, c = 'r', label = 'main customer')

plt.scatter(scalex[y_means == 3, 0], scalex[y_means == 3, 1], s = 100, c = 'y', label = 'maincustomer')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200, c = 'cyan' , label = 'centeroid')

plt.style.use('fivethirtyeight')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.grid()

plt.show()
import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(scalex, method = 'complete'))

plt.title('Dendrogam', fontsize = 15)

plt.xlabel('Customers')

plt.ylabel('Ecuclidean Distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(scaled_x)

plt.scatter(scalex[y_hc == 0, 0], scalex[y_hc == 0, 1], s = 100, c = 'green', label = 'usualcustomer')

plt.scatter(scalex[y_hc == 1, 0], scalex[y_hc == 1, 1], s = 100, c = 'b', label = 'perority customer')

plt.scatter(scalex[y_hc == 2, 0], scalex[y_hc == 2, 1], s = 100, c = 'r', label = 'main customer')

plt.scatter(scalex[y_hc == 3, 0], scalex[y_hc == 3, 1], s = 100, c = 'y', label = 'maincustomer')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200, c = 'cyan' , label = 'centeroid')

plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 15)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.grid()

plt.show()
from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=3,min_samples=4)

#Fitting the model



model=dbscan.fit(scaled_x)



labels=model.labels_
from sklearn import metrics



#identifying the points which makes up our core points

sample_cores=np.zeros_like(labels,dtype=bool)



sample_cores[dbscan.core_sample_indices_]=True
n_clusters=len(set(labels))- (1 if -1 in labels else 0)

n_clusters