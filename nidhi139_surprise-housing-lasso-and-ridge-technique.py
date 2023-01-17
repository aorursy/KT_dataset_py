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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#read the dataset1
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.shape
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
#Check the missing values in dataset
missing_null_values1=round(100*(train.isnull().sum()/len(train.index)),2).sort_values(ascending=False)
missing_null_values1
#impute for MiscFeature has most NA values
train['MiscFeature'].fillna('None', inplace=True)
# Check for MiscFeature
fig = plt.figure(figsize =(10,8))
sns.countplot(train['MiscFeature'])
plt.xticks(rotation=60)
train = train.drop('MiscFeature',axis = 1)
#impute
train['Alley'].fillna('No alley access',inplace = True)
train['BsmtQual'].fillna('No Basement for qual',inplace = True)
train['BsmtCond'].fillna('No Basement for cond', inplace=True)
train['BsmtExposure'].fillna('No Basement for expo', inplace=True)
train['BsmtFinType1'].fillna('No Basement for fintype1', inplace=True)
train['BsmtFinType2'].fillna('No Basement for fintype2', inplace=True)
train['FireplaceQu'].fillna('No Fireplace', inplace=True)
train['GarageType'].fillna('No Garage', inplace=True)
train['GarageFinish'].fillna('No Garage finish', inplace=True)
train['GarageQual'].fillna('No Garage finish', inplace=True)
train['GarageCond'].fillna('No Garage cond', inplace=True)
train['PoolQC'].fillna('No Pool', inplace=True)
train['Fence'].fillna('No Fence', inplace=True)
missing_null_values = round(100*(train.isnull().sum()/len(train.index)),2).sort_values(ascending=False)
missing_null_values
#conversion fo years to number of years
train['YearBuilt'] = 2019 - train['YearBuilt']
train['YearRemodAdd'] = 2019 - train['YearRemodAdd']
train['GarageYrBlt'] = 2019 - train['GarageYrBlt']
train['YrSold'] = 2019 - train['YrSold']
train.head()
train.shape
train['LotFrontage'].describe()
# impute values mean or median for the columns with less null values
train["LotFrontage"]=train["LotFrontage"].fillna(train["LotFrontage"].median())
train['GarageYrBlt'].describe()
# impute values mean or median for the columns with less null values
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].median())
#checking the missing values again
missing_null_values = round(100*(train.isnull().sum()/len(train.index)),2).sort_values(ascending=False)
missing_null_values
train.dropna(inplace = True)
train.isnull().sum()
# Understanding the Data Dictionary

#The data dictionary contains the meaning of various attributes; some non-obvious ones are:
train.info()
#converting some columns from int type to object, we'd rather treat it as categorical since it has only some discrete values.
train['MSSubClass'] = train['MSSubClass'].astype('object')
train['OverallQual'] = train['OverallQual'].astype('object')
train['OverallCond'] = train['OverallCond'].astype('object')
train['FullBath'] = train['FullBath'].astype('object')
train['HalfBath'] = train['HalfBath'].astype('object')
train['Fireplaces'] = train['Fireplaces'].astype('object')
train['GarageCars'] = train['GarageCars'].astype('object')
train['BedroomAbvGr'] = train['BedroomAbvGr'].astype('object')
train['KitchenAbvGr'] = train['KitchenAbvGr'].astype('object')
train['TotRmsAbvGrd'] = train['TotRmsAbvGrd'].astype('object')
train['BsmtFullBath'] = train['BsmtFullBath'].astype('object')
train['BsmtHalfBath'] = train['BsmtHalfBath'].astype('object')
# all numeric (float and int) variables in the dataset
train_numeric = train.select_dtypes(include=['float64', 'int64'])
train_numeric.head()
# correlation matrix
cor = train_numeric.corr()
cor
# plotting correlations on a heatmap

# figure size
plt.figure(figsize=(16,8))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()
#for Ridge and Lasso
train_R_L = train
# all categorical variables in the dataset
train_categorical = train_R_L.select_dtypes(include=['object'])
train_categorical.head()
# convert into dummies
train_dummies = pd.get_dummies(train_categorical, drop_first=True)
train_dummies.head()
# drop categorical variables 
train_R_L = train_R_L.drop(list(train_categorical.columns), axis=1)
# concat dummy variables with X
train_R_L = pd.concat([train_R_L, train_dummies], axis=1)
# split into X and y
X = train_R_L.drop([ 'SalePrice'], axis=1)

y = train_R_L['SalePrice']
# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 
#checking the value of optimum number of parameters
print(model_cv.best_params_)
print(model_cv.best_score_)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=1000]
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
alpha = 500
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_
#R square value of test and train data
from sklearn import metrics
y_train_pred = ridge.predict(X_train)
print(metrics.r2_score(y_true=y_train,y_pred=y_train_pred))
# lasso regression
lm = Lasso(alpha=0.001)
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# lasso model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))
# cross validation
folds = 5

# specify range of hyperparameters
params = {'alpha': [0.001, 0.01, 1.0, 5.0, 10.0]}

# grid search
# lasso model
lasso = Lasso()
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'r2', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
# plot
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('r2 score')
plt.xscale('log')
plt.show()
#checking the value of optimum number of parameters
print(model_cv.best_params_)
print(model_cv.best_score_)
# model with optimal alpha
# lasso regression
lm = Lasso(alpha=0.001)
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# lasso model parameters
model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))