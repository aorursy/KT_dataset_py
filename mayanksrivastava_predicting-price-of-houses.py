#Import the necessary Python Packages. 

import pandas as pd 

import numpy as np 

import seaborn as sns 

import matplotlib as plt 

import sklearn



# Set ipython's max row display

pd.set_option('display.max_row', 10000)



#Setting to print all the values in array

np.set_printoptions(threshold=np.nan)



# Set iPython's max column width to 50

pd.set_option('display.max_columns', 500)
#Import Dataset downloaded from the Kaggle https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data



traindata = pd.read_csv('../input/train.csv')

testdata = pd.read_csv('../input/test.csv')
#Let us try to understand more about the data.

traindata.info()
total = traindata.isnull().sum().sort_values(ascending=False)

percent = (traindata.isnull().sum()/traindata.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
#Get ridding of the columns with lot of missing data

traindata = traindata.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)

testdata = testdata.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
#First Deal with the numerical variables then move to the categorical string variables.

#Create Data set with numerical variables

num_trainData = traindata.select_dtypes(include = ['int64', 'float64'])

numcol = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']
#Find out correlation with numerical features

traindata_corr = num_trainData.corr()['SalePrice'][:-1]

golden_feature_list = traindata_corr[abs(traindata_corr) > 0].sort_values(ascending = False)

print("Below are {} correlated values with SalePrice:\n{}".format(len(golden_feature_list), golden_feature_list))
#Create heatmap for correlated numerical variables

%matplotlib inline

traindata_corrheatmap = num_trainData.corr()

cols = traindata_corrheatmap.nlargest(10, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(num_trainData[cols].values.T)

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#Understand the distribution of the Sale Price

traindata['SalePrice'].describe()
traindata['SalePrice'].skew()
traindata['SalePrice'].kurtosis()
sns.distplot(traindata['SalePrice'], color = 'b', bins = 100)
from scipy import stats

import matplotlib.pyplot as plt

res = stats.probplot(traindata['SalePrice'], plot=plt)
sns.distplot(np.log(traindata['SalePrice']), color = 'r', bins = 100)
res = stats.probplot(np.log(traindata['SalePrice']), plot=plt)
#Understand the behaviour of data in GrLivArea

from scipy import stats

import matplotlib.pyplot as plt

res = stats.probplot(traindata['GrLivArea'], plot=plt)
sns.distplot(traindata['GrLivArea'], color = 'b', bins = 100)
sns.distplot(np.log(traindata['GrLivArea']), color = 'b', bins = 100)
from scipy import stats

res = stats.probplot(np.log(traindata['GrLivArea']), plot=plt)
#Understand the skewness of TotalBsmtSF

sns.distplot(traindata['TotalBsmtSF'], color = 'b', bins = 100)
res = stats.probplot(traindata['TotalBsmtSF'], plot=plt)
traindata.plot.scatter(x = 'GrLivArea', y = 'SalePrice')



traindata.plot.scatter(x = 'GarageArea', y = 'SalePrice')



traindata.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')



traindata.plot.scatter(x = '1stFlrSF', y = 'SalePrice')



sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = traindata)



sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = traindata)



sns.boxplot(x = 'FullBath', y = 'SalePrice', data = traindata)



sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = traindata)



sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = traindata)



sns.boxplot(x = 'YearRemodAdd', y = 'SalePrice', data = traindata)
#Delete the outliers

traindata = traindata.drop(traindata[traindata['Id'] == 1299].index)

traindata = traindata.drop(traindata[traindata['Id'] == 524].index)
#On basis of EDA we did earlier, filter out the variable we want to use for predicting the sale price

finaldata = traindata.filter(['OverallQual','MSSubClass', 'KitchenAbvGr','OverallCond', 'GrLivArea', 'EnclosedPorch', 'GarageArea','TotalBsmtSF',  'YearBuilt', 'SalePrice'], axis = 1)

finaltest = testdata.filter(['OverallQual','MSSubClass', 'KitchenAbvGr', 'OverallCond','GrLivArea', 'EnclosedPorch', 'GarageArea','TotalBsmtSF',  'YearBuilt'], axis = 1)
#Handle mising values in test data 

finaltest.loc[finaltest.GarageArea.isnull(), 'GarageArea'] = 0

finaltest.loc[finaltest.TotalBsmtSF.isnull(), 'TotalBsmtSF'] = 0
#Transform Sale Price and GrLivArea to reduce standardize the data 

finaldata['SalePrice'] = np.log(finaldata['SalePrice'])

finaldata['GrLivArea'] = np.log(finaldata['GrLivArea'])

finaltest['GrLivArea'] = np.log(finaltest['GrLivArea'])
#Find out the columns which are missing in final data 

total = finaldata.isnull().sum().sort_values(ascending=False)

percent = (finaldata.isnull().sum()/finaldata.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#Splt into predictor and variable

xtrain = finaldata.iloc[:, :-1].values

ytrain = finaldata.iloc[:,9].values

xtest = finaltest.iloc[:, :9].values
#Prediction Model

import xgboost as xgb

regr = xgb.XGBRegressor()

regr.fit(xtrain, ytrain)



#Calculate the score for the XGBoost Model

regr.score(xtrain,ytrain)



# Run predictions using XGBoost

y_pred = regr.predict(xtrain)



#Predict the prices for Test Data Set

y_test = regr.predict(xtest)
##Fit Linear Regression Model 

from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()

regressor.fit(xtrain, ytrain)



#Calculate score for the Linear Regression model

regressor.score(xtrain,ytrain)



#Predict Value of the house using Linear Regression

ytrainpred = regressor.predict(xtrain)



#Predict Value of the house on test data set 

ytestpred = regressor.predict(xtest)
#Average out the predicted value from XGBoost and Linear Regression

finalpred = (y_test+ytestpred)/2

finalpred = np.exp(finalpred)
#Output to csv



my_submission = pd.DataFrame(finalpred, index=testdata["Id"], columns=["SalePrice"])

my_submission.to_csv('submission.csv', header=True, index_label='Id')