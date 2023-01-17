# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mlp

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
##reading the data 

df = pd.read_csv('../input/BlackFriday.csv')
#missing value too many to delete. Decided to fill in with 0

df2=df.fillna(0)

df2.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df2["sub_enc_Gender"] = le.fit_transform(df2["Gender"])

df2["enc_Age"] = le.fit_transform(df2["Age"])

df2["enc_City"] = le.fit_transform(df2["City_Category"])

df2["enc_years"] = le.fit_transform(df2["Stay_In_Current_City_Years"])

df2["enc_ProductID"] = le.fit_transform(df2["Product_ID"])

df2 = df2.drop(['Product_ID','Gender','City_Category', 'Age','Stay_In_Current_City_Years'], axis=1)

df2.info()  #everything converted to numerical form rather than categories
# Let's investigate if we have non-numeric data left

df2.info()   #we have one more: Stay_In_Current_City_Years
#convert everything to int

df2.Product_Category_2 = df2.Product_Category_2.astype(int)

df2.Product_Category_3 = df2.Product_Category_3.astype(int)

df2.info()
#Normalize the data for KMeans

from sklearn import preprocessing



x = df2.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_scaled = pd.DataFrame(x_scaled,columns=df2.columns, index=df2.index)

df_scaled.head(20)
#Too many data : reduction

data=df_scaled.sample(frac=0.02, replace=True, random_state=1)

data.shape
# TODO: Apply PCA by fitting the good data with only two dimensions

# Instantiate

# Instantiate

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(data)



reduced_data = pca.transform(data)



# Create a DataFrame for the reduced data

reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
# Imports

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



# Create range of clusters 

range_n_clusters = list(range(2,11))

# Loop through clusters

for n_clusters in range_n_clusters:

    

    clusterer = KMeans(n_clusters).fit(reduced_data)



    # TODO: Predict the cluster for each data point

    preds = clusterer.predict(reduced_data)



    # TODO: Find the cluster centers

    centers = clusterer.cluster_centers_



    # TODO: Predict the cluster for each transformed sample data point

    sample_preds = clusterer.predict(reduced_data)



    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen

    score = silhouette_score(reduced_data, preds, metric='mahalanobis')

    print ("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))
#Best score when number of cluster is 4. 

# Let's try to visualize ncluster=2 first

datavalues=reduced_data.values  # convert to array

model = KMeans(n_clusters=2)

model.fit(datavalues)

labels = model.predict(datavalues)

print(labels)
print(datavalues)
# Assign the columns of new_points: xs and ys

xs = datavalues[:,0]

ys = datavalues[:,1]



# Make a scatter plot of xs and ys, using labels to define the colors

plt.scatter(xs,ys,c=labels,alpha=0.5)



# # Assign the cluster centers: centroids

# centroids = model.cluster_centers_



# # Assign the columns of centroids: centroids_x, centroids_y

# centroids_x = centroids[:,0]

# centroids_y = centroids[:,1]



# # Make a scatter plot of centroids_x and centroids_y

# plt.scatter(centroids_x,centroids_y,marker='D',s=50)

plt.show()
datavalues=reduced_data.values  # convert to array

model = KMeans(n_clusters=4)

model.fit(datavalues)

labels = model.predict(datavalues)

# Assign the columns of new_points: xs and ys

xs = datavalues[:,0]

ys = datavalues[:,1]



# Make a scatter plot of xs and ys, using labels to define the colors

plt.scatter(xs,ys,c=labels,alpha=0.5)

plt.title('KMeans, n_cluster=4')

plt.show()