# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#Import the dataset\

data=pd.read_csv('../input/Iris.csv')

print(data.head())

print(data.info()) #No missing values, data is clean

print(data.groupby('Species')['Species'].count()) #50 rows of each type of Species



#We are going to use KMeans to try classify these data within three clusters to see how

#well it can predict using the four available features

#StandardScalar can be done in cases where there is a significant difference in variance between features 

#(the four columns in this case), a Standard Scalar transforms all features to have a mean=0 and variance=1

#This makes the features standardized

#Plot a scatter graph to see what the distribution of the data looks like 

#(for any one of the two feature's (Sepal/Petal) length and width)

import seaborn as sns #Python's graphic library helps 

import matplotlib.pyplot as plt



# 'hue' argument used to 

sns.lmplot( x="SepalLengthCm", y="SepalWidthCm", data=data, fit_reg=False, hue='Species', legend=False)
#We can add a boxplot for each feature to check how the feature ranges varies for each type of Species

graph1 = sns.boxplot(x="Species", y="PetalLengthCm", data=data)

graph1 = sns.stripplot(x="Species", y="PetalLengthCm", data=data, jitter=True)



#This graph shows that the Iris-setosa has small Petal Length compared to the other two

#The dots are the actual PetalLength of each row 

#We can see that versivolor and virginica length's range overlap



data.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6)) #Done using pandas package for the others
#Create the data to build the model on

#Drop Id and Species column

data1=data.drop(['Id','Species'],axis=1)

print(data1.head())
#We will first try KMeans model without Standardizing the features

from sklearn.cluster import KMeans

model=KMeans(n_clusters=3) #Since we have three Species here

model.fit(data1)

labels=model.predict(data1) #Labels indicate the cluster assigned to each row, starting from 0-->2

print(labels)



#Get the centroids for each cluster

centroids=model.cluster_centers_

print(centroids) 

#There are four columns, but we use 0, 1 for x and y of centroids, not sure what the other two is for yet

cen_x=centroids[:,0]

cen_y=centroids[:,1]

#Create a new dataframe for this

cent=pd.DataFrame(data=centroids[:,:2],columns=["X","Y"]) 

print(cent)



#add labels to the data1 dataset

data1['Labels']=labels

print(data1.head())
#Predicted one:

xs=data1.iloc[:,0]

ys=data1.iloc[:,1]

zs=data1.iloc[:,4]

plt.scatter(xs,ys,c=zs)

plt.scatter(cen_x,cen_y,marker='D',s=50)



#Original Species model

sns.lmplot( x="SepalLengthCm", y="SepalWidthCm", data=data, fit_reg=False, hue='Species', legend=False)
#Evaluating the cluster

species=data['Species']

data2=pd.DataFrame({'Labels':labels,'Species':species})

print(data2.head())



print(pd.crosstab(data2['Labels'],data['Species']))

#We can see here that the mdoel predicted setosa perfectly,

#but there has been mix up in versiolor and virginica



#We can also find the inertia of the model (how spread out the model is)--> lower the better

print(model.inertia_) 

#with increasing cluters, we get a lower inertia, however here we are given three varieties
#We will now see if standardising the features make a difference in the quality of the results

print(data1.head())

data1.drop(['Labels'],axis=1)



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

scaler=StandardScaler()

kmeans=KMeans(n_clusters=3)



from sklearn.pipeline import make_pipeline

pipeline=make_pipeline(scaler,kmeans)

pipeline.fit(data1)

Labels2=pipeline.predict(data1)



data3=pd.DataFrame({'Labels':Labels2,'Species':species})

print(data3.head())

print("1.With standardscalar")

print(pd.crosstab(data3['Labels'],data['Species']))



#Added here for comparison

print("2.Without standardscalar")

print(pd.crosstab(data2['Labels'],data['Species']))



#For this particular example, standardscalar does not improve the model, infact makes it worst, because 

#there would not have been difference in variances between the four features.
#Trying t-SNE 2D map in this example

from sklearn.manifold import TSNE

model=TSNE(learning_rate=100) #can be 50-200

transformed=model.fit_transform(data1) #Fits data1 features into an embedded space and returns that transformed output.

xs=transformed[:,0]

ys=transformed[:,1]

plt.scatter(xs,ys,c=labels)

plt.show()