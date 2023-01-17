# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# create dataframe from dataset 

df = pd.read_csv('../input/toy-dataset/toy_dataset.csv')
# first 5 rows of dataframe

df.head()
df.info()
# Convert the categorial features into numerical

df['Male'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

df['Illness'] = df['Illness'].apply(lambda x: 1 if x =='Yes' else 0)
# remove old categorical feature

df.drop('Gender',axis=1,inplace=True)
# Convert the categorial features into numerical

cities = pd.get_dummies(df['City'],drop_first=True)
# create new dataframe from old dataframe and new dummy features

df_new = pd.concat([df,cities],axis=1)
# remove old categorical feature

df_new.drop('City',axis=1,inplace=True)
# since theres more than 2 features, let's use principle components by importing it

from sklearn.decomposition import PCA
# create an instance of principle component analysis with 2 components

pca = PCA(n_components=2)
# convert the dataframe into 2 principle components

components = pca.fit_transform(df_new)
# create new dataframe for the principle components and column names

df2d = pd.DataFrame(data=components,columns=['PCA1','PCA2'])
# import K Means Clustering 

from sklearn.cluster import KMeans
# initialise the model and choose how many clusters we're looking for, 3 has been chosen arbitrarily

clusters = KMeans(n_clusters=3)
# fit the model to the principle components dataframe

clusters.fit(df2d)
# plot the points from the principle components dataframe on a scatter plot and colour the clusters according predicted labels

fig = plt.figure(figsize=(7,3))

plt.scatter(x=df2d['PCA1'],y=df2d['PCA2'],c=clusters.labels_)