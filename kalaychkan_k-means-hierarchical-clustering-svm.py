# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.cluster import KMeans  #K Means

from sklearn.cluster import AgglomerativeClustering #Hierarchical Clustering

import scipy.cluster.hierarchy as sch

from sklearn.svm import SVC #SVM

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

data.isnull()

data.isnull().sum()
data.info() #Null data yok
data.describe()
print('Total number of customer:',len(data))
data.head()
plt.style.use('fivethirtyeight')

plt.figure(1 , figsize = (15 , 6))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(data[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1 , figsize = (15 , 5))

sns.countplot(y = 'Gender' , data = data)

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = data)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
plt.figure(1 , figsize = (15 , 6))

for gender in ['Male' , 'Female']:

    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,

                data = data[data['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 

plt.title('Annual Income vs Spending Score')

plt.legend()

plt.show()
'''K-MEANS'''

X1 = data[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)



algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

clusters=algorithm.predict(X1)

labels2 = algorithm.labels_

centroids2 = algorithm.cluster_centers_

h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z2 = Z2.reshape(xx.shape)

plt.imshow(Z2 , interpolation='nearest', 

    extent=(xx.min(), xx.max(), yy.min(), yy.max()),

    cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = data , c = labels2 , 

               s = 200 )

plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')

plt.show()
data['clusters']=clusters

data.tail()
'''Hierarchical Clustering Algorithm'''

plt.figure(figsize = (25,10))

dendrogram = sch.dendrogram(sch.linkage(X1, method = 'ward'))



plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distances')

plt.show()
agg_clustering = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

agg_preds = agg_clustering.fit_predict(X1)



point_size = 20

colors = ['red', 'blue', 'green', 'cyan', 'magenta']





plt.figure(figsize = (15 , 7))

for i in range(5):

    plt.scatter(X1[agg_preds == i,0], X1[agg_preds == i,1], s = point_size, c = colors[i])



plt.title('Clusters of Clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc = 'best')

plt.show()
'''SVM cluster'''

cluster_0=data[data['clusters']==0]

print(cluster_0.describe())

print('Total number of customer:',len(cluster_0))
cluster_1=data[data['clusters']==1]

print(cluster_1.describe())

print('Total number of customer:',len(cluster_1))
cluster_2=data[data['clusters']==2]

print(cluster_2.describe())

print('Total number of customer:',len(cluster_2))

cluster_3=data[data['clusters']==3]

print(cluster_3.describe())

print('Total number of customer:',len(cluster_3))
cluster_4=data[data['clusters']==4]

print(cluster_4.describe())

print('Total number of customer:',len(cluster_4))
sex_mapping = {"Female": 0, "Male": 1}

def mymap(x, mapping): return mapping[x]

data['Gender'] = data['Gender'].apply(mymap, mapping = sex_mapping)

X = data.drop(columns=['clusters']).values 

y = data['clusters'].values

scaler=StandardScaler()

X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state = 0, stratify=y)

print ("length_of_train: " ,len(X_train))

print ("length_of_validation: " ,len(X_test))



svc_rbf=SVC(kernel='rbf',gamma='auto')

svc_rbf.fit(X_train,y_train)



print("svc_rbf accuracy:", (svc_rbf.score(X_test, y_test))*100)

print("Score: ",svc_rbf.score(X_train, y_train))



sex_mapping = { 0:"Female",  1: "Male"}

def mymap(x, mapping): return mapping[x]

data['Gender'] = data['Gender'].apply(mymap, mapping = sex_mapping)
