# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import statsmodels.api as sm



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/prostate.data',sep='\t')
print(data.columns.values)
# preview the data

data.head()
data.shape # Display the number of rows and columns of variable data

print(data.columns.values) # display the columns

data=data.drop(columns=data.columns.values[0]) #remove first column

data.head()  # Show the first 5 rows of the dataset

train=pd.DataFrame(data[['train']]) # Save column train in a new variable called train and having type Series (the Pandas data structure used to represent DataFrame columns, then drop the column train from the data

# DataFrame)

data=data.drop(columns=['train']) # drop the column train from the data DataFrame

data.head()

lpsa=pd.DataFrame(data['lpsa']) # Save column lpsa in a new variable called lpsa and having type Series (the Pandas data

# structure used to represent DataFrame columns)

data=data.drop(columns=['lpsa']) # drop the column lpsa from the data DataFrame and save the result in a new DataFrame called predictors

print(lpsa)

data.head()
data.isna().sum() #searching missin value

rig=2

print((data.columns.values.size)/rig)

fig, axes = plt.subplots(nrows=2, ncols=(data.columns.values.size)//2)

col=0;

rig=0;

for i in data.columns.values:

    if(col>=data.columns.values.size//2):

        col=0;

        rig=rig+1;

    sns.countplot(x=i,data=data, ax=axes[rig,col]) #gives me a graph with the number of categorical variables

    col=col+1
#minimum

for i in data.columns.values:

    print(i+":")

    print(data[i].min())

#max

for i in data.columns.values:

    print(i+":")

    print(data[i].max())

 #mean

for i in data.columns.values:

    print(i+":")

    print(data[i].mean())
#Dividing traning from test sets

dataTrain=pd.DataFrame(data.loc[train['train']=='T'])

dataTest=pd.DataFrame(data.loc[train['train']=='F'])

print(data.shape)

print(dataTrain.shape)

print(dataTest.shape)
lpsaTrain=pd.DataFrame(lpsa.loc[train['train']=='T'])

lpsaTest=pd.DataFrame(lpsa.loc[train['train']=='F'])

print(lpsaTrain.shape)

print(lpsaTest.shape)

print(lpsa.shape)
dataTrain.corr()
data.insert(8,"lpsa",lpsa) #BBBOH
dataTrain.head()

#Dividing traning from test sets 

predictorsTrain=pd.DataFrame([data.loc[train['train']=='T']['lpsa']]).transpose()

predictorsTest=pd.DataFrame([data.loc[train['train']=='F']['lpsa']]).transpose()

dataTrain.head()

dataTrain.corr()
predictorsTrain=dataTrain

predictorsTest=dataTest

predictorsTrain_std=pd.DataFrame();

#let's standardize everthing

count=0;

for x in  predictorsTrain.columns.values:

    mean=predictorsTrain[x].mean();

    std=predictorsTrain[x].std();

    predictorsTrain_std.insert(count,x,(predictorsTrain[x]-mean).div(std))

    predictorsTrain_std.head()

    count=count+1;

predictorsTrain_std=pd.DataFrame(predictorsTrain_std)

predictorsTrain_std.head()
print(lpsaTrain.shape)

print(predictorsTrain_std.shape)

print(predictorsTest.shape)
reg=LinearRegression(fit_intercept=False).fit(predictorsTrain_std,lpsaTrain) #we set fit_intecept=false cause we already normilize our data

# NB --> predictorsTrain_std is the dependent variables and

# lpsaTrain is the independent variable

#i next time we can normalize by setting normalize=True

coef=reg.coef_ # Estimates coefficients for the linear regression problem

print(coef[0])

lpsaTrain_std=(lpsaTrain-lpsaTrain.mean()).div(lpsaTrain.std())
mod= sm.OLS(lpsaTrain,predictorsTrain_std)

print(mod)

ris=mod.fit()



ris.summary()





#ris.RegressionResults(lpsaTest)





#print(res.bse)



#np.divide(coef[0],std)