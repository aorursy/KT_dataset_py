# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kmodes.kmodes import KModes
#?KModes
# reproduce results on FDA 2017 Drug inspection data set
#x = np.genfromtxt('../input/FDA2017_drug_all.csv', dtype=int, delimiter=',', names=True, filling_values=0)
#kmodes_drug2017 = KModes(n_clusters=15, init='Drug', verbose=1)
#clusters=kmodes_drug2017.fit_predict(x)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train=pd.read_csv("../input/fda-2017-drug-inspection-observation/FDA2017_drug_all.csv",header=0,engine='python',index_col=0, skipfooter=100)
restaurant=pd.read_csv("../input/nyc-restaurant-inspection-violation/NYC_RESTAURANT_INS_CLUSTERING.csv",header=0,engine='python',index_col=0, nrows=500)
print(train.describe())
print(train.columns.values)
train.isna().head()
print(train.isna().sum())
train.fillna(0, inplace=True)
restaurant.fillna(0,inplace=True)
print(train.isna().sum())
train.info()
restaurant.info()
#kmeans = KMeans(n_clusters=15)
X=np.array(train)
Y=np.array(restaurant)
#kmeans.fit(X)
# Using the elbow method to find the optimal number of clusters
wcssX=[]
wcssY=[]
distancesX=[]
distancesY=[]
j=100
import math as math
from math import sqrt
class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])
    
    def distance_to_line(self, x,y):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff*self.x - x_diff*self.y + p2.x*p1.y - p2.y*p1.x)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den 
for i in range (1,j):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init = 10, random_state =20)
    kmeans.fit(X)
    wcssX.append(kmeans.inertia_)
for i in range (1,j):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init = 10, random_state =20)
    kmeans.fit(Y)
    wcssY.append(kmeans.inertia_)
print(wcssX)
print(wcssY)
#p1 = Point(x_init=1,y_init=wcss[0])
#print(p1)
#p2 = Point(x_init=30,y_init=wcss[28])
#print(p2)
for k in range (1,j):
    p1 = Point(x_init=0,y_init=wcssX[0])
    p2 = Point(x_init=j-1,y_init=wcssX[j-2])
    p = Point(x_init=k-1,y_init=wcssX[k-2])
    distancesX.append(p.distance_to_line(p1,p2))
for k in range (1,j):
    p1 = Point(x_init=0,y_init=wcssY[0])
    p2 = Point(x_init=j-1,y_init=wcssY[j-2])
    p = Point(x_init=k-1,y_init=wcssY[k-2])
    distancesY.append(p.distance_to_line(p1,p2))
print(distancesY)
print("The maximum distance is ",max(distancesY),"at {}th clustering".format(distancesY.index(max(distancesY))))
plt.figure(figsize=(15,10))
plt.plot(range(1,j), wcssX)
plt.plot(range(1,j),distancesX)
plt.title("The elbow method_FDA pharmacy company inspection")
plt.xlabel("The number of clusters")
plt.ylabel("WCSS")
plt.legend(['wcss', 'distance'], loc='upper right')
plt.show()
plt.figure(figsize=(15,10))
plt.plot(range(1,j), wcssY)
plt.plot(range(1,j),distancesY)
plt.title("The elbow method_NYC restaurant hygiene inspection")
plt.xlabel("The number of clusters")
plt.ylabel("WCSS")
plt.legend(['wcss', 'distance'], loc='upper right')
plt.show()
kmeans = KMeans(n_clusters = 18, init='k-means++', max_iter = 300, n_init = 10, random_state =20)
a=kmeans.fit(Y).labels_
print(a)
type(a)
#np.shape(a)
b=np.unique(a)
for n in b:
    print("Clustering {}".format(n)+" has {} restaurants,".format(a.tolist().count(n)))
#type(restaurant)
#restaurant.values()
#restaurant.as_matrix()
#restaurant.info()
#c.info()
#np.shape(a)
#a.tolist.info()
#a.reshape(500,1)
#np.shape(a)
#print(a)
#type(Y)
#np.shape(Y)
#z=np.append(Y, a, axis=0)
z=np.column_stack((restaurant.as_matrix(),a))
#print(z.tolist())
print(z)
pd.DataFrame(z).info()
score=pd.DataFrame(z).groupby([69])
score.describe().head(18)
