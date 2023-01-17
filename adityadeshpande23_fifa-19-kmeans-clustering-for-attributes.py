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
data = pd.read_csv('../input/data.csv')

data.shape

data.columns
data.head()
data.shape
data.describe()
# Change column name of the 1st column

data.rename(columns={'Unnamed: 0':'Sl_num'},inplace=True)
# Check if the name changed

data.columns
#Get required coulmn numbers

data.columns.get_loc("Crossing")
data.columns.get_loc("GKReflexes")
# Find index of all missing values for a particular column

data[data['Dribbling'].isnull()].index.tolist()
data[data['GKReflexes'].isnull()].index.tolist()
# Drop all rows with empty / missing values 

data.drop(data.index[[13236,13237,13238,13239,13240,13241,13242,13243,13244,13245,13246,13247,13248,13249,13250,13251,13252,

13253,13254,13255,13256,13257,13258,13259,13260,13261,13262,13263,13264,13265,13266,13267,13268,13269,13270,13271,13272,13273,

13274,13275,13276,13277,13278,13279,13280,13281,13282,13283]], inplace= True)
type(data)
# Check if delete was successful

data[data['GKReflexes'].isnull()].index.tolist()
#Check for missing values again

data.isnull().sum()
#Select all attribute columns that affect the overall rating of the player

dt1=data.iloc[:,54:88]
dt1.head()
dt1.describe()
## Scale the data, using pandas

def scale(x):

    return (x-np.mean(x))/np.std(x)

data_scaled=dt1.apply(scale,axis=0)
data_scaled.head()
## Scale the data using sklearn

import sklearn.preprocessing as preprocessing

dat_scaled=preprocessing.scale(dt1,axis=0)
print (dat_scaled)

print ("Type of output is "+str(type(dat_scaled)))

print ("Shape of the object is "+str(dat_scaled.shape))
## Create a cluster model

import sklearn.cluster as cluster
kmeans=cluster.KMeans(n_clusters=3,init="k-means++")

kmeans=kmeans.fit(dat_scaled)
lab=kmeans.labels_

lab=list(lab)
kmeans.cluster_centers_
from scipy.spatial.distance import cdist

#np.min(cdist(dat_scaled, kmeans.cluster_centers_, 'euclidean'),axis=1)
## Elbow method



K=range(1,20)

wss = []



for k in K:

    kmeans = cluster.KMeans(n_clusters=k,init="k-means++")

    kmeans.fit(dat_scaled)

    wss.append(sum(np.min(cdist(dat_scaled, kmeans.cluster_centers_, 'euclidean'), 

                                      axis=1)) / dat_scaled.shape[0])
import matplotlib.pyplot as plt

plt.plot(K, wss, 'bx')

plt.xlabel('k')

plt.ylabel('Average distortion')

plt.title('Selecting k with the Elbow Method')

plt.show()
import sklearn.metrics as metrics

labels=cluster.KMeans(n_clusters=8,random_state=200).fit(dat_scaled).labels_
metrics.silhouette_score(dat_scaled,labels,metric="euclidean",sample_size=10000,random_state=200)
for i in range(7,13):

    labels=cluster.KMeans(n_clusters=i,random_state=200).fit(dat_scaled).labels_

    print ("Silhoutte score for k= "+str(i)+" is "+str(metrics.silhouette_score(dat_scaled,labels,metric="euclidean",

                                 sample_size=10000,random_state=200)))
## Let's look for profiles for 8,9,10 clusters

kmeans=cluster.KMeans(n_clusters=8,random_state=200).fit(dat_scaled)
#Population mean

colmeans=dt1.mean()
type(colmeans)
#Population standard deviation

std=dt1.std(axis=0)
type(std)
#Group Means(sample mean)

group_mean=dt1.groupby([kmeans.labels_]).mean()
type(group_mean)
#Difference between sample mean and population mean

x_mean=group_mean.sub(colmeans,axis=1)
#Divide by std to get z score

x_mean.divide(std,axis=1)