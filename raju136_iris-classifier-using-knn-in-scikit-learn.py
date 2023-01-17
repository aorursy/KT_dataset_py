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



# changing the working directory to the file located path

print(os.getcwd())

os.chdir('../input')

print(os.getcwd())

# Load the data

data=pd.read_csv('Iris.csv')

print(data.head()) # see the forst 5 columns



# setting the index as column Id

data=data.set_index(['Id'])



print(data.head()) # see the forst 5 columns
# Summarize the dataset



# shape of the data

print(data.shape)



# statistical summary of the data

print(data.describe())



# the count of each species type in the whole data (Class distribution)

print(data.groupby('Species').size())
# Visualizing the data



# import libraries

import matplotlib.pyplot as plt

# Univariate Plotting

# box and whisker plots [To understand the distribution of attributes]

data.plot(kind='box',subplots='True',layout=(2,2),sharex=False,sharey=False)

plt.show()



# histogram plotting

data.hist()

plt.show()
# multi variate plots

from pandas.plotting import scatter_matrix

scatter_matrix(data)

plt.show()



#import seaborn as sb

#sb.pairplot(data)
# Training the model



# splitting the data into features and targets

array=data.values

print(array[:3])

X=array[:,0:3]

Y=array[:,4]

validation_size=0.2

seed=7

from sklearn.model_selection import train_test_split

# splitting the data into training and test dataset

X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=validation_size,random_state=seed)



scoring='accuracy'

# Applying a simple KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,Y_train)

y_pred=knn.predict(X_validation)



print("The accuracy is {}".format(knn.score(X_validation,Y_validation)))



print(knn.score(X_validation,Y_validation))