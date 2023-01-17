#DataFrames

import numpy as np

import pandas as pd



#Scikit learn

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.cluster import AgglomerativeClustering

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

from sklearn.pipeline import make_pipeline   

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder



#Visuals

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns

from pandas import plotting



#Others

import random

random.seed(42)

import warnings

import os

warnings.filterwarnings("ignore")

from IPython.display import Image

import math

# Eaton Center !

Image(url='https://i.gifer.com/QE7.gif')
df = pd.read_csv('../input/Mall_Customers.csv')

df.head(5)
# rows, columns

df.shape
df.info()
df.describe()
# Check for Nulls

df.isnull().sum().sort_values(ascending=False)
# Unique values count

print(df.nunique())
# drop Customer id 

df = df.drop('CustomerID', axis=1)

df.head(2)
# rename columns

new_cols = ['Gender', 'Age', 'AnnualIncome','SpendingScore']



df.columns = new_cols



df.head(3)
# Categorical Scatterplot on Gender Vs Annual Income

sns.catplot(x="Gender", y="AnnualIncome", kind="swarm",hue="Gender", data=df.sort_values("Gender"))
# Distributions of observations within categorical attribute - "Gender"

sns.catplot(x="Gender", y="AnnualIncome", kind="boxen",data=df.sort_values("Gender"))
sns.relplot(x="Age", y="AnnualIncome", hue="Gender", style="Gender",size="SpendingScore",sizes=(1, 100), data=df);
# Data distribution 

sns.pairplot(df);
# Visual linear relationship

plt.figure(1 , figsize = (15 , 7))

n=0

new_cols = ['Age', 'AnnualIncome','SpendingScore']



for x in new_cols:

    for y in new_cols:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = df)

            

plt.show()
# lets calculate the variance of each numerical attribute in the dataset.

df.var(ddof=0).plot(kind='bar')
# lets calculate the SD of each numerical attribute in the dataset.

df.std(ddof=0)
train_X, test_X = train_test_split(df, test_size=0.2, random_state=42)



print(len(train_X), "train +", len(test_X), "test")
# lets take copy of the data 

df2 = train_X.copy()
# Let fit and transform the Gender attribute into numeric

le = LabelEncoder()

le.fit(df2.Gender)
# 0 is Female, 1 is Male

le.classes_
#update df2 with transformed values of gender

df2.loc[:,'Gender'] = le.transform(df2.Gender)
df2.head(3)
# Create scaler: scaler

scaler = StandardScaler()

scaler.fit(df2)
# transform

data_scaled = scaler.transform(df2)

data_scaled[0:3]
pca = PCA()



# fit PCA

pca.fit(data_scaled)
# PCA features

features = range(pca.n_components_)

features
# PCA transformed data

data_pca = pca.transform(data_scaled)

data_pca.shape
# PCA components variance ratios.

pca.explained_variance_ratio_
plt.bar(features, pca.explained_variance_ratio_)

plt.xticks(features)

plt.ylabel('variance')

plt.xlabel('PCA feature')

plt.show()
# Principal component analysis (PCA) and singular value decomposition (SVD) 

# PCA and SVD are closely related approaches and can be both applied to decompose any rectangular matrices.

pca2 = PCA(n_components=2, svd_solver='full')



# fit PCA

pca2.fit(data_scaled)



# PCA transformed data

data_pca2 = pca2.transform(data_scaled)

data_pca2.shape
xs = data_pca2[:,0]

ys = data_pca2[:,1]

#zs = train_X.iloc[:,2]

plt.scatter(ys, xs)

#plt.scatter(ys, zs, c=labels)





plt.grid(False)

plt.title('Scatter Plot of Customers data')

plt.xlabel('PCA-01')

plt.ylabel('PCA-02')



plt.show()
# KMeans model



# lets assume 4 clusters to start with



k=4 

kmeans = KMeans(n_clusters=k, init = 'k-means++',random_state = 42) 
# Build pipeline

pipeline = make_pipeline(scaler, pca2, kmeans)

#pipeline = make_pipeline(kmeans)
# fit the model to the scaled dataset

model_fit = pipeline.fit(df2)

model_fit
# target/labels of train_X

labels = model_fit.predict(df2)

labels
# lets add the clusters to the dataset

train_X['Clusters'] = labels
# Number of data points for each feature in each cluster

train_X.groupby('Clusters').count()
# Scatter plot visuals with labels



xs = data_pca2[:,0]

ys = data_pca2[:,1]

#zs = train_X.iloc[:,2]

plt.scatter(ys, xs,c=labels)

#plt.scatter(ys, zs, c=labels)



plt.grid(False)

plt.title('Scatter Plot of Customers data')

plt.xlabel('PCA-01')

plt.ylabel('PCA-02')



plt.show()
# Centroids of each clusters.

centroids = model_fit[2].cluster_centers_

centroids
X = data_pca2

# Assign the columns of centroids: centroids_x, centroids_y

centroids_x = centroids[:,0]

centroids_y = centroids[:,1]
# Visualising the clusters & their Centriods

plt.figure(figsize=(15,7))

sns.scatterplot(X[labels == 0, 0], X[labels == 0, 1], color = 'grey', label = 'Cluster 1',s=50)

sns.scatterplot(X[labels == 1, 0], X[labels == 1, 1], color = 'blue', label = 'Cluster 2',s=50)

sns.scatterplot(X[labels == 2, 0], X[labels == 2, 1], color = 'yellow', label = 'Cluster 3',s=50)

sns.scatterplot(X[labels == 3, 0], X[labels == 3, 1], color = 'green', label = 'Cluster 4',s=50)



sns.scatterplot(centroids_x, centroids_y, color = 'red', 

                label = 'Centroids',s=300,marker='*')

plt.grid(False)

plt.title('Clusters of customers')

plt.xlabel('PCA-01')

plt.ylabel('PCA-02')

plt.legend()

plt.show()
# Distance from each sample to centroid of its cluster

model_fit[2].inertia_
# WCSS stands for Within Cluster Sum of Squares. It should be low.



ks = range(1, 10)

wcss = []

samples = data_pca2



for i in ks:

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(samples)

    # inertia method returns wcss for that model

    wcss.append(kmeans.inertia_)
# lets visualize 

plt.figure(figsize=(10,5))

sns.lineplot(ks, wcss,marker='o',color='skyblue')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
def getInertia2(X,kmeans):

    ''' This function is analogous to getInertia, but with respect to the 2nd closest center, rather than closest one'''

    inertia2 = 0

    for J in range(len(X)):

        L = min(1,len(kmeans.cluster_centers_)-1) # this is just for the case where there is only 1 cluster at all

        dist_to_center = sorted([np.linalg.norm(X[J] - z)**2 for z in kmeans.cluster_centers_])[L]

        inertia2 = inertia2 + dist_to_center

    return inertia2 
wcss = []

inertias_2 = []

silhouette_avgs = []



ks = range(1, 10)

samples = data_pca2



for i in ks:

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(samples)

    wcss.append(kmeans.inertia_)

    inertias_2.append(getInertia2(samples,kmeans))

    if i>1:

        silhouette_avgs.append(silhouette_score(samples, kmeans.labels_))
silhouette_avgs
plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

plt.title("wcss: sum square distances to closest cluster")

plt.plot(ks,wcss)

plt.xticks(ks)

plt.xlabel('number of clusters')

plt.grid()

    

plt.subplot(1,3,2)    

plt.title("Ratio: wcss VS. sum square distances to 2nd closest cluster")

plt.plot(ks,np.array(wcss)/np.array(inertias_2))

plt.xticks(ks)

plt.xlabel('number of clusters')

plt.grid()



plt.subplot(1,3,3)  

plt.title("Average Silhouette")

plt.plot(ks[1:], silhouette_avgs)

plt.xticks(ks)

plt.xlabel('number of clusters')

plt.grid()



plt.show()
# Copy the dataset

df_new = test_X.copy()
# predict the labels

le.fit(df_new.Gender)



#update df2 with transformed values of gender

df_new.loc[:,'Gender'] = le.transform(df_new.Gender)



labels_test = model_fit.predict(df_new)

labels_test
# lets add the clusters to the dataset

test_X['Clusters'] = labels_test

# Number of data points for each feature in each cluster

test_X.groupby('Clusters').count()
query = (test_X['Clusters']==1)

test_X[query]
from IPython.display import display, HTML



HTML('''<div style="display: flex; justify-content: row;">

    <img src="https://media.giphy.com/media/MEgGD8bV72hfq/giphy.gif">

    <img src="https://media.giphy.com/media/3k9gOXgimLWF2/giphy.gif">

    <img src="https://media.giphy.com/media/3o751RE4VSNLjpSLew/giphy.gif">

</div>''')
# Are these shoppers? No idea! what they are doing.

Image(url='https://media.giphy.com/media/fAhOtxIzrTxyE/giphy.gif')