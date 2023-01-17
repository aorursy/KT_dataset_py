# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold

from sklearn.model_selection import train_test_split



#get the data from a csv file. Basically a tabular excel file

df = pd.read_csv('../input/train.csv')

#get the testing data

testDf = pd.read_csv('../input/test.csv')

#peak at a few rows of the data 

print (df.head())

#get the value we wish to predict from the data set

target = df[['SalePrice']]

#see which values correlate to a change in the y value i.e. sale price

#print (df.corr())

#get the inputs to our linear regression model

features = df[['OverallQual']]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=0)

testFeatures = testDf[['OverallQual']]

#create a linear regression object

lr = LinearRegression()

lr.fit(features,target)

print ('Without k fold:',lr.score(features,target))

lr = LinearRegression()

#Our test data set does not have the sale price values so we need to use another method to check our model

kf = KFold(len(X_train), n_folds=10, random_state=0,shuffle=True)

for trainIndex, testIndex in kf:

    xTrain, xTest = X_train.iloc[trainIndex], X_train.iloc[testIndex]

    yTrain, yTest = y_train.iloc[trainIndex], y_train.iloc[testIndex]

    #fit our linear model on the training set

    lr.fit(xTrain,yTrain)

    print ('Score:',lr.score(xTest,yTest))

#print (df.dtypes)

print ('Score on Test Set:',lr.score(X_test,y_test))
import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')



plt.figure()

plt.title('Sale Price')

plt.ylabel('Price in $\'s')

plt.yscale('log')

df.SalePrice.plot.box()



plt.figure()

plt.title('Sale Price')

plt.ylabel('Price in $\'s')

df.SalePrice.plot.box()



plt.figure()

plt.title('Sale Price')

plt.xlabel('Price in $\'s')

df.SalePrice.plot.hist()



plt.figure()

plt.title('Sale Price by Overall Quality Ratings')

plt.ylabel('Price in $\'s')

df.groupby(['OverallQual'])['SalePrice'].mean().plot()



plt.figure()

plt.title('Sale Price by Overall Quality Ratings')

plt.ylabel('Price in $\'s')

df.groupby(['OverallQual'])['SalePrice'].mean().plot(kind='bar')



plt.figure()

plt.title('Sale Price by Overall Quality Ratings')

df.groupby(['OverallQual'])['SalePrice'].mean().plot(kind='pie')
features = df[['OverallQual','LotArea','YearBuilt','FullBath']]





lr = LinearRegression()

lr.fit(features,target)

print ('Without k fold:',lr.score(features,target))

#create a linear regression object

lr = LinearRegression()

#Our test data set does not have the sale price values so we need to use another method to check our model



for trainIndex, testIndex in kf:

    xTrain, xTest = features.iloc[trainIndex], features.iloc[testIndex]

    yTrain, yTest = target.iloc[trainIndex], target.iloc[testIndex]

    #fit our linear model on the training set

    lr.fit(xTrain,yTrain)

    print ('Score:',lr.score(xTest,yTest))

print ('Score on all Data:', lr.score(features,target))

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')



#plt.plot(df[['OverallQual']],df[['SalePrice']])

plt.figure()

plt.yscale('log')

df.SalePrice.plot.box()



plt.figure()

df.groupby(['OverallQual'])['SalePrice'].mean().plot()