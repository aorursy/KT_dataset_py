## Importing the basic library

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import sklearn as sk

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

# Reading the dataset

iris = pd.read_csv('../input/Iris.csv')
# Doing basic checks on the dataset

iris.describe()

iris.info()

iris.columns

iris.dtypes
## Performin basic EDA
correlate = iris[list(iris.dtypes[iris.dtypes!='object'].index)].corr()
sns.pairplot(correlate)
sns.heatmap(correlate,annot = True)
sns.countplot(iris.Species)
plt.scatter(x ='PetalWidthCm', y = 'PetalLengthCm',data=iris)

#plt.ylim(5,6)

#plt.xlim(1.5,2.0)
iris_y = iris.pop('Species')

iris_x = iris
# Splitting the dataframe for training and testing the models

train_X,test_X,train_y,test_y = train_test_split(iris_x,iris_y,test_size = 0.5,random_state = 4)
## Running the model on top of taining data set

model = KNeighborsClassifier()

model.fit(train_X,train_y)
prediction = model.predict(test_X)
accuracy_score(prediction,test_y)
pd.DataFrame({'Pred':prediction,

              'Actual':test_y})