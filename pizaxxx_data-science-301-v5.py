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

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



#Loading Packages
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



#Manually set panda max outputs for print, .info, .describe, etc...

#Doesn't work... no big deal :/
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



#Reading in train and test files
train.head()
test.head()
train.info()
test.info()
print("train_shape=",train.shape)

print("test_shape",test.shape)
#

#

#

#

#Let's examine the categorical variables

#

#

#

#
train['MSZoning'].value_counts() #Let's look at categorical variable values...

                                 #Best way to think of these is leves of a 'factor' variable in r 
train["LotShape"].value_counts()
train["LotConfig"].value_counts()
train["Neighborhood"].value_counts()
train["HouseStyle"].value_counts()
train["ExterQual"].value_counts()
train["ExterCond"].value_counts()
train["BsmtQual"].value_counts()
train["BsmtCond"].value_counts()
train["HeatingQC"].value_counts()
train["KitchenQual"].value_counts()
train["SaleCondition"].value_counts()
train.hist(bins = 50, figsize = (20, 15)) #plot histogram relationship 

plt.show()
# Plot Correlation Matrix of train

corrmat = train.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat, vmax = 1, square = True);
print("train_shape=",train.shape)

print("test_shape",test.shape) 
train.info()
test.info()
#

#

#

#

# Let's plot the relationship between quantitative variables and sale.price

# Sale price is an important variable sense we are going to use it as our target variable

#

#

#

#
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['LotArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('LotArea', fontsize=13)

plt.show()



fig, ax = plt.subplots()

ax.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['OverallCond'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallCond', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['BsmtFinSF1'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('BsmtFinSF1', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['BsmtFullBath'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('BsmtFullBath', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['BsmtHalfBath'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('BsmtHalfBath', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['FullBath'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('FullBath', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['HalfBath'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('HalfBath', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GarageCars'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageCars', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GarageArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageArea', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['YearRemodAdd'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('YearRemodAdd', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['YearBuilt'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('YearBuilt', fontsize=13)

plt.show()
train.info()
test.info()
# Examine Missing Values of train_set

sns.heatmap(train.isnull(), cbar=False)

#Examine Missing values of test_set

sns.heatmap(test.isnull(), cbar=False)

#Examine correlation between missing values

msno.heatmap(train)
msno.heatmap(test)

msno.dendrogram(train)

msno.dendrogram(test, orientation = "top")
#

#

#

#

# Looking at the Dendrogram plots for train and test dataframes we can see many interesting relationships between missing variables

# We can use a dendrogram to represent the relationships between any kinds of entities 

# As long as we can measure their similarity to each other.

# Good use if we merge feature columns together

#

#

#

#
#Looking at the data description for the competition we can see that a missing value is always a lack of "User Input"

#Instead of having an # Na value which results in less data for our model to learn from, change NaN to "nonve"



categorical_V = []

for i in train.columns:

    if train[i].dtype == object:

        categorical_V.append(i)

train.update(train[categorical_V].fillna('None'))
categorical_V2 = []

for i in test.columns:

    if test[i].dtype == object:

        categorical_V2.append(i)

test.update(test[categorical_V2].fillna('None'))
# Same philosphy for quantitative variables



Quantitative = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

Quantitative_V = []

for i in train.columns:

    if train[i].dtype in Quantitative:

        Quantitative_V.append(i)

train.update(train[Quantitative_V].fillna(0))



Quantitative_V2 = []

for i in test.columns:

    if test[i].dtype in Quantitative:

        Quantitative_V2.append(i)

test.update(test[Quantitative_V2].fillna(0))
train.info()
test.info()
print("train_shape=",train.shape)

print("test_shape",test.shape)
# Perform One-Hot Encoding on concatenated Dataframe



temp = pd.get_dummies(pd.concat([train,test],keys=[0,1]))



# Split concatenated dataframe back into train and test dataframes



train,test = temp.xs(0),temp.xs(1)



test.drop(["SalePrice"], axis = 1, inplace = True) # Earlier concatenation requires removal of SalePrice from test dataframe

                                                   # Check to make sure all SalePrice values are Null for ID 1461-End

    

train.drop(["Id"], axis = 1, inplace = True)

#test.drop(["Id"], axis = 1, inplace = True)
print("train_shape=",train.shape)

print("test_shape",test.shape)
# Double-check that missing values have been handled accordingly 



print(train.isnull().values.any())

print(test.isnull().values.any())
# Let's create our predictors and target variable

# As mentioned above, SalePrice is our target variable(what we are trying to predict)

# All other variables are predictor variables(Used to predict the target variable)

X, y = train.loc[:, train.columns != 'SalePrice'], train[["SalePrice"]]
# Cross Validation for RMSE

from sklearn.model_selection import KFold, cross_val_score # Libraries needed





kfolds = KFold(n_splits= 10, shuffle=True, random_state=42) # Generic Number of Folds

                                                            # n_splits = 10 is optimal from previous version results

def rmse_cv(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)   # Following rsme_cv obtained from: 

                        #https://www.programcreek.com/python/example/91148/sklearn.model_selection.cross_val_score

# Library for PipeLines



from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline
# Scaler Libraries

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler
# Library for Models

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import LassoCV

from sklearn.ensemble import GradientBoostingRegressor





# Possible Models to try but have not attempted 

from xgboost import XGBClassifier 

from sklearn.svm import LinearSVC

import sklearn.linear_model as linear_model
# PipeLine for Linear Regression

linear = make_pipeline(LinearRegression())
#PipeLine for Lasso Regression



lasso = make_pipeline(RobustScaler(), #Scale features using statistics that are robust to outliers.

                      LassoCV(max_iter=1e7, 

                              random_state=42, cv=kfolds))
# Pipeline for Random Forest Model

rnd_reg = make_pipeline(RandomForestRegressor(n_estimators = 500, 

                                              max_leaf_nodes = 16,

                                             n_jobs = -1))
# Pipeline for AdaBoost Model

# Both imports are required cause AdaBoost uses Decision Tree

ada_reg = make_pipeline(AdaBoostRegressor(DecisionTreeRegressor(max_depth = 5),

                            n_estimators = 200,

                           learning_rate = 0.5)) #Need to explore learning rate more
xgb_reg = make_pipeline(XGBRegressor())
score_1 = rmse_cv(linear)

print("Linear Regression Score: ", score_1.mean())



score_2 = rmse_cv(lasso)

print("Lasso Regression Score:  ", score_2.mean())



score_3 = rmse_cv(rnd_reg)

print("Random Forest Score: ", score_3.mean())



score_4 = rmse_cv(ada_reg)

print("AdaBoost Score: ", score_4.mean())



score_5 = rmse_cv(xgb_reg)

print("XGBoost Score: ", score_5.mean())
# Fit Models with X and y trains



Linear_model = linear.fit(X, y)



lasso_model = lasso.fit(X, y)



random_F_model = rnd_reg.fit(X, y)



ada_model = ada_reg.fit(X, y)



XGB_model = xgb_reg.fit(X, y)
# Test our model predictions



test_X = test.loc[:, test.columns != 'Id']



predicted_prices_1 = linear.predict(test_X)

predicted_prices_1 = predicted_prices_1.ravel()

print(predicted_prices_1)





predicted_prices_2 = lasso.predict(test_X)

print(predicted_prices_2)



predicted_prices_3 = rnd_reg.predict(test_X)

print(predicted_prices_3)



predicted_prices_4 = ada_reg.predict(test_X)

print(predicted_prices_4)



predicted_prices_5 = xgb_reg.predict(test_X)

print(predicted_prices_5)



#from sklearn.metrics import mean_squared_error

#from math import sqrt



#print("Linear Regression RMSE:", sqrt(mean_squared_error(test_X, predicted_prices_1)))

#print("Lasso Regression RMSE:", sqrt(mean_squared_error(test_X, predicted_prices_2)))

#print("RandomForestRegressor RMSE:", sqrt(mean_squared_error(test_X, predicted_prices_3)))

#print("AdaBoost RMSE:", sqrt(mean_squared_error(test_X, predicted_prices_4)))

#print("XGBoost RMSE:", sqrt(mean_squared_error(test_X, predicted_prices_5)))
# Ensemble Model 

# Model actually performs worse than previous models... 



from sklearn.ensemble import VotingRegressor

voting_reg = VotingRegressor(estimators = [('rf', rnd_reg),

                                          ('ada', ada_reg),

                                          ('lr', lasso),

                                          ('xg', xgb_reg)])



voting_model = voting_reg.fit(X, y)



predicted_prices_6 = voting_reg.predict(test_X)

print(predicted_prices_6)



print(predicted_prices_1.shape)

print(predicted_prices_2.shape)

print(predicted_prices_3.shape)

print(predicted_prices_4.shape)

print(predicted_prices_5.shape)

print(predicted_prices_6.shape)
# Current thoughts: 

# I think tuning XGBoost is best bet for right now

# Voting Regressor performing worse than XGBoost alone was a little shocking

# TUNE TUNE TUNE to come !
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_5})# Choose which predicted_price for submissio

                                                                              # 1 for linear, 2 for lasso... etc and will refine later 

                                                                              # Ensemble model to come

my_submission.to_csv('Ames_House_Lasso.csv', index=False)