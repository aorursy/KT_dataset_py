#Import library

import numpy as np

import pandas as pd

import matplotlib



#Load csv

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



#Combine train and test dataset to better understand the data

titanic = pd.concat([train, test])

#Change index from default to the PassengerId in the dataset

titanic.index = titanic["PassengerId"]
titanic.head()
titanic["Age"].hist()
%matplotlib inline
titanic["Sex"].value_counts().plot.bar()