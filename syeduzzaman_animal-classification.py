import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import datasets

from sklearn.cluster import KMeans

import sklearn.metrics as sm



from pandas import DataFrame



# Set some pandas options

pd.set_option('display.notebook_repr_html', False)

pd.set_option('display.max_rows', 60)

pd.set_option('display.max_columns', 60)

pd.set_option('display.width', 1000)

 

%matplotlib inline



# read csv file 

classData = pd.read_csv("../input/class.csv")

zooData = pd.read_csv("../input/zoo.csv")



# data cleaning : drop the NA rows 

df=df.dropna()

classData.set_index(['Class_Type'], inplace = True)

classData.head()
zooData.head()
print("Row: ",zooData.shape[0])

print("Column: ",zooData.shape[1])
zooData['domestic'].value_counts()
zooData['venomous'].value_counts().plot.bar()
mask1 = zooData['venomous'] == 1

temp1 = zooData[['venomous','class_type']]

venom = temp1[mask1]

venom
zooData['care'] = (np.sqrt(zooData.eggs*3+zooData.milk*3+zooData.domestic*1.5+zooData.breathes*2+zooData.aquatic))*1000
zooData['care'].head()
zooData['strong'] = (np.sqrt(zooData.predator*2.5+zooData.venomous*3+zooData.toothed+zooData.airborne+zooData.legs*2))*1000

zooData['strong'].head()
predictionData = zooData[['strong','care','rate']]

predictionDataFUN = zooData[['strong','care','rate']]



from sklearn.neighbors import KNeighborsClassifier

from matplotlib.colors import ListedColormap

knn = KNeighborsClassifier(n_neighbors = 5, 

                           p = 2)# p=2 for euclidean distance

knn.fit(predictionData[["strong", "care"]], 

        zooData.rate)