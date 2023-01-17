import numpy as np

import pandas as pd

import pylab as pl 



%pylab inline 
# Reading in data 

path = '../input/'

titanic = pd.read_csv(path + 'test.csv')

train = pd.read_csv(path + 'train.csv')

gendermodel = pd.read_csv(path + 'gendermodel.csv')
titanic.dtypes
train.dtypes
# Concatenating Titanic and Train data frames

frames = [titanic, train, gendermodel]

titanic = pd.concat(frames)

titanic.describe()
titanic.count(axis = 0)
titanic.head()
# Rename and reassign values to make data more readable

titanic.rename(columns = {'Pclass':'EconomicClass'}, inplace = True)

titanic['Gender'] = titanic['Sex'].map({'female':0,'male':1})

titanic['EnconomiClass'] = titanic['EconomicClass'].map({1: 'Upper', 2: 'Middle', 3: 'Lower'})

titanic = titanic [['PassengerId', 'Age', 'EconomicClass']]

titanic