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
#SETUP

#Load the training dataset
train = pd.read_csv("../input/train.csv")

#Define target variable
SalePrice = train['SalePrice']
#Check for missing values
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
#DEAL WITH MISSING VALUES

#Delete columns where missing data is more than 1 (all except Electrical)
train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)

#Delete observation with missing Electrical data
train = train.drop(train.loc[train['Electrical'].isnull()].index)

#Need to redefine SalePrice to account for missing observation
SalePrice = train['SalePrice'];  

#Check all missing values are gone
print(train.isnull().sum().max())
#Descriptive statistics
print(train.head())
print(train.describe())
print(train.columns)

#Correlation Heatmap
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(train.corr());
corrmat=train.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#Define X
X = train[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']]
print(X.head())
print(X.shape)

#Define y
y = SalePrice
#Plot chosen features to look for outliers
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'GarageCars'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Identify IDs of outliers based on above charts - 2 in GrLivArea, 1 in TotalBsmtSF
print(train.sort_values(by = 'GrLivArea', ascending = False)[:2])
print(train.sort_values(by = 'TotalBsmtSF', ascending = False)[:1])
#Delete them
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)
#Split data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)
print(Xtrain.shape)
#DEFINE ERROR FUNCTION: RMSE
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
scorer = make_scorer(mean_squared_error, greater_is_better=False)
def rmse_cv_train(model):
    rmse = np.sqrt(-cross_val_score(model, Xtrain, ytrain, scoring=scorer, cv=10))
    return(rmse)
def rmse_cv_test(model):
    rmse = np.sqrt(-cross_val_score(model, Xtest, ytest, scoring=scorer, cv=10))
    return(rmse)
#FIT MODEL

#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtrain, ytrain)

#Check RMSE
print("Training RMSE: ", rmse_cv_train(lr).mean())
print("Test RMSE: ", rmse_cv_test(lr).mean())

#Plot Predictions
ytrainpred = lr.predict(Xtrain)
ytestpred = lr.predict(Xtest)
plt.clf()
plt.scatter(ytrainpred, ytrain, c = "blue", marker = "s", label = "Training data")
plt.scatter(ytestpred, ytest, c = "lightgreen", marker = "s", label = "Test data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
#Load Test File
TestData = pd.read_csv("../input/test.csv")

#Check out the Test File
print(TestData.head())
print(TestData.shape)
print(TestData.columns)

#Select features
TestX = TestData[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']]
print(TestX.head())
#Check for missing values
total = TestX.isnull().sum().sort_values(ascending=False)
percent = (TestX.isnull().sum()/TestX.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
# Mark zero values as missing or NaN
TestX.fillna(value=0, inplace=True)

#Check all missing values are gone
print(TestX.isnull().sum().max())
#Make predictions
TestY = lr.predict(TestX)
TestY = pd.Series(TestY)
print(TestY.shape)
#Prepare submission file
submission = pd.DataFrame({'Id':TestData['Id'], 'SalePrice':TestY})
print(submission.head())
print(submission.shape)
submission.to_csv('hpsubmission.csv', index=False)