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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# 1. Reading csv files

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

# Data Cleaning

# we have "Unnamed" Column with all values  NaN, so we remove this column 

# df  = df.drop(columns =['Unnamed: 32'])

# another method

df  = df.drop('Unnamed: 32', axis= 1)

df  = df.drop('id', axis= 1)



#  Now, the data has categorical section, diagnosis 

#  Here, we give unordered nominal values to these categories M and B

nom = {'M': 1, 'B': 0}

df.diagnosis = [nom[i] for i in df.diagnosis.astype(str)]



# considering all columns as feature to find correlated datas

correlation_matrix = df.iloc[:,1:31]



# this plt is used to zoom the picture by 20*10 size

plt.figure(figsize=(31, 20))



# heatmap of correlated data

# x.corr() - gives correlated values

correlated_features = set()

correlation_matrix = correlation_matrix.corr(method='pearson', min_periods=1)



#  to see how many correlated features 

# sns.heatmap(correlation_matrix,annot =True)



#  filtering out only correlated feature

for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.85:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)

            

# dropping all correlated features            

df = df.drop(correlated_features, axis = 1)



 # removing zeros value rows to apply box-cox transformation

df = df[df != 0]

df = df.dropna()



# # for all datas except prediction one, normalize the data

# to reduce skewess and to normalize the data, we use boxcox transformation

from scipy import stats



# number of columns in df dataframe

num_of_cols = len(df.columns)

# number of rows in df dataframe

num_of_rows = len(df.index)



#I wanted to show 6 figures in each rows

ncol = 6

nrow = int(num_of_cols/ncol)

# Plots

# for multiple plots

fig, axs = plt.subplots(nrow,ncol)

c = 0

r = 0

for i in df.iloc[:,1:]:

#     df[i], fitness_values = stats.boxcox(df[i])

    sns.distplot(df[i], ax=axs[r,c])

    c += 1

    if c == ncol:

        c = 0

        r += 1





from sklearn.model_selection import train_test_split

import time



y = df['diagnosis'] # target output

f = df.iloc[:,1:18] # remaining features as input



# stratified splitting

Xtrain, Xtest, Ytrain, Ytest =  train_test_split(f, y, test_size=0.2)



print("Training DataSets Shape : ", Xtrain.shape, Xtest.shape)

print("Testing DataSets Shape  : ",Ytrain.shape, Ytest.shape)



from sklearn import *



# fit a model

lm = linear_model.LinearRegression()



# Basic linear regression technique

lmmodel = lm.fit(Xtrain, Ytrain)

print("LR Score:", lmmodel.score(Xtest, Ytest))



# Support Vector Regression Technique

# with tunning Parameters

svrScore,gamma1 = 0,0.001

for i in range(1,1000):

    gamma = i*0.001

    svr = svm.SVR(kernel='rbf', gamma= gamma, C=2)

    svrmodel = svr.fit(Xtrain, Ytrain)

    score = svrmodel.score(Xtest, Ytest)

    if svrScore < score:

        svrScore = score

        gamma1 = gamma



print("SVR Score: ",svrScore, " for gamma: ", gamma1)




