# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis and wrangling
import pandas as pd
import numpy as np
import random

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
combine = [train_df, test_df]
train_df
train_df = train_df.drop(['PassengerId','Cabin','Ticket'], axis=1)
test_df = test_df.drop(['PassengerId','Cabin','Ticket'], axis=1)

#complete missing age with median
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)

#complete embarked with mode
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)

test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace = True)

train_df
train_df['Sex'] = train_df['Sex'].map({'male':0,'female':1})
test_df['Sex'] = test_df['Sex'].map({'male':0,'female':1})

train_df
for dataset in [train_df,test_df]:
    
    # Creating a categorical variable to tell if the passenger is alone
    dataset['IsAlone'] = ''
    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) > 0)] = 1
    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) == 0)] = 0
    
    
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # take only top 10 titles
    title_names = (dataset['Title'].value_counts() < 10) #this will create a true false series with title name as index

    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    print(dataset['Title'].value_counts())
    
    dataset.drop(['Name','SibSp','Parch'],axis=1,inplace=True)

train_df.head()
#define x and y variables for dummy features original
train_dummy = pd.get_dummies(train_df,drop_first=True)
test_dummy = pd.get_dummies(test_df,drop_first=True)

train_dummy
X_final = train_dummy.drop(['Survived'],axis=1).values # for original features
target = train_dummy['Survived'].values
X_final
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_final = sc.fit_transform(X_final)
print(X_final.shape,'\n',X_final)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class MYKMeans:
    def __init__(self,k=5,iters=1000,plot_steps=False):
        self.k=k
        self.iters=iters
        self.plot_steps = plot_steps

        # clusters are essentially a group of points.. for k clusters, we will have k group of points so let's make that
        self.n_clusters = [[] for _ in range(self.k)] # empty list of lists [[],[],[]]

        # We need to have a list of centroids for every cluster. For k clusters, we'll have k centroids in list
        self.centroids = [] 

    def predict(self,X):
        # the meat of the code!
        # So, how do you predict??
        self.X=X
        self.n_samples, self.n_features = X.shape # (1000,10) means 10000 samples with 10 feature each

        # Step 2: It is to chose random centroids!
        rand_idx = np.random.choice(self.n_samples, self.k, replace=False) # [34,54]
        self.centroids = [self.X[i] for i in rand_idx] # centroids = [X[34],X[54]], which is a list of size n_features, just a sample

        # Let's go through Optimizing clusters!
        for _ in range(self.iters):
            # first step to optimization is to create new clusters!
            # create clusters by assigning them to their nearest centroid
            self.clusters = self.create_clusters(self.centroids) # abstract and send centroids!
            if self.plot_steps:
                self.plot()

            # calculate new centroids from the clusters!
            # we do that by taking mean 
            old_centroids = self.centroids
            self.centroids = self.update_centroids(self.clusters) # again clusters is just a list of lists. 

            if self.plot_steps:
                self.plot()
                
            # check if clusters have changed, if yes, we are done else train again!
            if self.is_converged(old_centroids, self.centroids):
                break
            # Classify samples as the index of their clusters

        return self.get_cluster_labels(self.clusters)

    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]

        for idx, sample in enumerate(self.X): # for every sample
            # distance of the current sample to each centroid
            distances = [euclidean_distance(sample, centroid) for centroid in centroids] # remember centroid is also a sample, that we chose, we find dist between these 2 samples for every centroid
            closest_index = np.argmin(distances) # min dist of this sample with every centroid

            # once you have the cluster you want to assign to, put that sample idx in a cluster
            clusters[closest_index].append(idx)
        return clusters

    def update_centroids(self, clusters):
        # assign mean value of clusters to centroids
        # make empty list of zeroes
        centroids = np.zeros((self.k, self.n_features))

        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0) # since cluster has the idx of samples, we do X[cluster] to get actual sample
            centroids[cluster_idx] = cluster_mean

        return centroids

    def get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
          for sample_index in cluster:
            labels[sample_index] = cluster_idx
        return labels

    def is_converged(self,old , new):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(old[i], new[i]) for i in range(self.k)]
        return sum(distances) == 0
  
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
          point = self.X[index].T
          ax.scatter(*point,linewidth=3)

        for point in self.centroids:
          ax.scatter(*point, marker="x", color='black', linewidth=6)

        plt.show()

# X,y = make_blobs(centers = 3,n_samples = 1000, n_features = 3, shuffle = False)
kmeans_clf = MYKMeans(k=2)
preds = kmeans_clf.predict(X_final)
preds
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class myKMeans():
    def __init__(self,k,iters=1000):
        self.k=k
        self.iters=iters
        self.clusters=[[] for _ in range(self.k)]
        self.centroids=[]
    def predict(self,X):
        self.X=X
        self.samples,self.features=X.shape
        rand_idx = np.random.choice(self.samples,self.k,replace=False)
        self.centroids = [self.X[i] for i in rand_idx]
        
        for _ in range(self.iters):
            self.clusters = self.create_clusters(self.centroids)
            old_centroids = self.centroids
            self.centroids = self.update_centroids(self.clusters)
            if self.is_converged(old_centroids,self.centroids):
                break
        return self.get_cluster_label(self.clusters)
            
    def create_clusters(self,centroids):
        clusters = [[] for i in range(self.k)]
        for idx,sample in enumerate(self.X):
            distances = [euclidean_distance(sample,centroid) for centroid in centroids]
            closest_index = np.argmin(distances)
            clusters[closest_index].append(idx)
        return clusters
    
    def update_centroids(self,clusters):
        centroids = np.zeros((self.k, self.features))
        for cluster_idx,cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster],axis=0)
            centroids[cluster_idx]=cluster_mean
        return centroids
    
    def is_converged(self,old,new):
        dist = [euclidean_distance(old[i],new[i]) for i in range(self.k)]
        return sum(dist)==0
    
    def get_cluster_label(self, clusters):
        labels = np.empty(self.samples)

        for cluster_idx, cluster in enumerate(clusters):
          for sample_index in cluster:
            labels[sample_index] = cluster_idx
        return labels
    
kmeans_clf = myKMeans(k=2)
preds = kmeans_clf.predict(X_final)
# preds
accurate = 0
for i,val in enumerate(train_dummy['Survived']):
    if val==preds[i]:
        accurate+=1
print(accurate)
print("Accuracy = ",accurate/len(preds))
from sklearn.cluster import KMeans
kmeans_sk = KMeans(n_clusters=2, random_state=0).fit(X_final)
kmeans_sk.labels_
accurate = 0
for i,val in enumerate(train_dummy['Survived']):
    if val==kmeans_sk.labels_[i]:
        accurate+=1
print(accurate)
print("Accuracy = ",accurate/len(kmeans_sk.labels_))
