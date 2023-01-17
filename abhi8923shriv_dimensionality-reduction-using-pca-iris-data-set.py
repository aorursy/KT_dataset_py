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
#Dimensionality Reduction using PCA (iris data set)
# importing dataset package to import dataset



from sklearn import datasets 

from sklearn import model_selection



# Import PCA from sklearn.decomposition to project data from higher dimensions to lower dimensions



from sklearn.decomposition import PCA



# Import linear model to use the SGDClassifier further in the experiment



from sklearn import linear_model



# Use load_iris() to get the data 

# It returns an object of the dataset



data=datasets.load_iris()



# Get data from the dataset object 'data'



dataArr= data.data





#### Let us check the type of the variable 'dataArr'



type(dataArr)
#### Let us check the type of the dataset object 'data'

type(data)
#### Let us get features present in the dataset from dataset object 'data'



data.feature_names
data.keys
# Use train_test_split from model_selection to split the data into train and test data



traindata,testdata,labeltrain,labeltest=model_selection.train_test_split(dataArr,data.target)
# Look at the shape of the training and testing sets

# shape will return number of rows and columns in a dataset

traindata.shape

testdata.shape

labeltrain.shape

labeltest.shape
# Now, reduce the dimensions of data from 4D to 3D using PCA

# '''We use PCA technique from decomposition which takes as input

# number of components to keep in the lower dimension'''

### We create an object of PCA class



pca=PCA(n_components=3)



#### We are transforming and fitting the data to PCA by using principle components to project the data to lower dimensions 



datareduced=pca.fit_transform(dataArr)
type(datareduced)
# Plot the data after its dimensions are reduced

# plotting the data

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



### Create a scatter plot with the reduced data across the 3 principle components



### Create a scatter plot with the reduced data across the 3 principle components

fig= plt.figure(1,figsize=(8,6))

axes=Axes3D(fig, elev=-150, azim=110)

axes.scatter(datareduced[:, 0], datareduced[:, 1], datareduced[:, 2], c=data.target,

           cmap=plt.cm.Set1, edgecolor='k', s=40)

axes.set_title("First three principal components")

axes.set_xlabel('1st principal component')

axes.set_ylabel('2nd principal component')

axes.set_zlabel('3rd principal component')

axes.set_xticklabels([])

axes.set_yticklabels([])

axes.set_zticklabels([])



plt.show()



# Plot the data after its dimensions are reduced

# plotting the data

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



### Create a scatter plot with the reduced data across the 3 principle components



### Create a scatter plot with the reduced data across the 3 principle components

fig= plt.figure(1,figsize=(12,12))

axes=Axes3D(fig, elev=-250, azim=110)

axes.scatter(datareduced[:, 0], datareduced[:, 1], datareduced[:, 2], c=data.target,

           cmap=plt.cm.Set1, edgecolor='k', s=40)

axes.set_title("First three principal components")

axes.set_xlabel('1st principal component')

axes.set_ylabel('2nd principal component')

axes.set_zlabel('3rd principal component')

axes.set_xticklabels([])

axes.set_yticklabels([])

axes.set_zticklabels([])



plt.show()
# Plot the data after its dimensions are reduced

# plotting the data

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



### Create a scatter plot with the reduced data across the 3 principle components



### Create a scatter plot with the reduced data across the 3 principle components

fig= plt.figure(1,figsize=(12,12))

axes=Axes3D(fig, elev=-300, azim=40)

axes.scatter(datareduced[:, 0], datareduced[:, 1], datareduced[:, 2], c=data.target,

           cmap=plt.cm.Set1, edgecolor='k', s=40)

axes.set_title("First three principal components")

axes.set_xlabel('1st principal component')

axes.set_ylabel('2nd principal component')

axes.set_zlabel('3rd principal component')

axes.set_xticklabels([])

axes.set_yticklabels([])

axes.set_zticklabels([])



plt.show()
# Now let us try to train various models using the data after its dimensions are reduced

# #### Use train_test_split from model_selection to split the reduced data into train and test data



traindatared,testdatared=model_selection.train_test_split(datareduced)
# Apply a Linear classifier on the original data

# SGDClassifier calculates the gradient of the loss by iterating through each sample of the dataset and is updated with the learning rate .



# We create an object of Stochastic Gradient Descent Classifer class



# We create an object of Stochastic Gradient Descent Classifer class

clf = linear_model.SGDClassifier(max_iter=1080, tol=1e-3)

# We are fitting the data to SGDClassifier 

clf.fit(traindata,labeltrain)

# Returns the mean accuracy on the given test data and labels

clf.score(testdata,labeltest)
# Apply Linear classifier on the reduced data

# We create an object of Stochastic Gradient Descent Classifer class

clfPCA = linear_model.SGDClassifier(max_iter=99, tol=1e-3)

# We are fitting the reduced data to SGDClassifier 

clfPCA.fit(traindatared,labeltrain)

# Returns the mean accuracy on the given reduced test data and labels

clfPCA.score(testdatared,labeltest)
# Apply KNN on the original data



from sklearn.neighbors import KNeighborsClassifier

# We create an object of K Nearest Neighbors Classifer class

clf2=KNeighborsClassifier(n_neighbors=6)

# We are fitting the data to SGDClassifier

clf2.fit(traindata,labeltrain)

# Returns the mean accuracy on the given test data and labels

clf2.score(testdata,labeltest)

# Apply KNN on reduced data

# We create an object of K Nearest Neighbors Classifer class

clf2PCA = KNeighborsClassifier(n_neighbors=11)

# We are fitting the reduced data to SGDClassifier 

clf2PCA.fit(traindatared,labeltrain)

# Returns the mean accuracy on the given reduced test data and labels

clf2PCA.score(testdatared,labeltest)
# Apply KNN on reduced data

# We create an object of K Nearest Neighbors Classifer class

clf2PCA = KNeighborsClassifier(n_neighbors=7)

# We are fitting the reduced data to SGDClassifier 

clf2PCA.fit(traindatared,labeltrain)

# Returns the mean accuracy on the given reduced test data and labels

clf2PCA.score(testdatared,labeltest)
# Apply KNN on reduced data

# We create an object of K Nearest Neighbors Classifer class

clf2PCA = KNeighborsClassifier(n_neighbors=17)

# We are fitting the reduced data to SGDClassifier 

clf2PCA.fit(traindatared,labeltrain)

# Returns the mean accuracy on the given reduced test data and labels

clf2PCA.score(testdatared,labeltest)
# Apply KNN on reduced data

# We create an object of K Nearest Neighbors Classifer class

clf2PCA = KNeighborsClassifier(n_neighbors=23)

# We are fitting the reduced data to SGDClassifier 

clf2PCA.fit(traindatared,labeltrain)

# Returns the mean accuracy on the given reduced test data and labels

clf2PCA.score(testdatared,labeltest)
# Apply KNN on reduced data

# We create an object of K Nearest Neighbors Classifer class

clf2PCA = KNeighborsClassifier(n_neighbors=58)

# We are fitting the reduced data to SGDClassifier 

clf2PCA.fit(traindatared,labeltrain)

# Returns the mean accuracy on the given reduced test data and labels

clf2PCA.score(testdatared,labeltest)
# Apply KNN on reduced data

# We create an object of K Nearest Neighbors Classifer class

clf2PCA = KNeighborsClassifier(n_neighbors=72)

# We are fitting the reduced data to SGDClassifier 

clf2PCA.fit(traindatared,labeltrain)

# Returns the mean accuracy on the given reduced test data and labels

clf2PCA.score(testdatared,labeltest)