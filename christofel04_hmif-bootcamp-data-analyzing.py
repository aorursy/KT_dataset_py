# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import missingno as msno

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#classifiaction.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



#regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification



#Data Preprocess and Visualization

from statsmodels.stats.outliers_influence import variance_inflation_factor
data = pd.read_csv('../input/hmif-bootcamp/train-data.csv')

data.describe(include='all')
data.columns
data.isnull().sum()
#data =pd.get_dummies(data, drop_first = True)

#data.columns
#vif = pd.DataFrame()

#variables= data

#vif["VIF"]= [variance_inflation_factor(variables.values, i) for i in range (variables.shape[1])]

#vif["Features"]= variables.columns

#vif
sns.distplot(data['provinsi'])
sns.distplot(data['akreditasi'])
corr =data.corr()

fig, ax= plt.subplots()

im = ax.imshow(corr.values)



#set labels

ax.set_xticks(np.arange(len(corr.columns)))

ax.set_yticks(np.arange(len(corr.columns)))

ax.set_xticklabels(corr.columns)

ax.set_yticklabels(corr.columns)



# Rotate the thick label and set the alignment

plt.setp(ax.get_xticklabels(), rotation= 45, ha= "right", rotation_mode= "anchor")



# Create tex annotations

for i in range(len(corr.columns)):

    for j in range(len(corr.columns)):

        text= ax.text(j, i, np.around(corr.iloc[i,j], decimals= 2), ha="center", va="center", color="black")
plt.figure(figsize= (35,20))

dummy_data = pd.get_dummies(data, drop_first = True)

sns.heatmap(dummy_data.corr(), annot= True, linewidths= .5)
dummy_data.columns
plt.figure(figsize= (35,20))

dummy_data = pd.get_dummies(data, drop_first = True)

sns.heatmap(dummy_data.corr(), annot= True, linewidths= .5)
variables = data[['kurikulum' , 'penyelenggaraan' , 'akreditasi']]

dummy_variables = pd.get_dummies(variables, drop_first = True)

sns.heatmap(dummy_variables.corr(), annot= True, linewidths= .5)
model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']

model_err=[3.10522 , 3.25582, 3.17704, 3.15208, 3.20884]

d= {'Modelling Algorithm': model_names, 'Error' :model_err}

test_frame = pd.DataFrame(d)

test_frame