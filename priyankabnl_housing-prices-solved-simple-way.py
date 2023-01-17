# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
training = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

testing = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
training.head()
training.describe()
correlations = training.corr()

correlations = correlations["SalePrice"].sort_values(ascending=False)

correlations
training_null = pd.isnull(training).sum()

testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null,testing_null],axis =1 , keys = ['Training','Testing'])
print(null)
null_many = null[null.sum(axis=1)>200]

null_few = null[(null.sum(axis=1)>0) & (null.sum(axis=1)<200)]

print(null_many)
null_objects = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "MasVnrType","BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
for i in null_objects:

    training[i].fillna('None', inplace = True)

    testing[i].fillna('None', inplace = True)
training_null = pd.isnull(training).sum()

testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null,testing_null],axis =1 , keys = ['Training','Testing'])
print(null)
null_many = null[null.sum(axis=1)>200]

null_few = null[(null.sum(axis=1)>0) & (null.sum(axis=1)<200)]

print(null_many)
training.drop('LotFrontage', axis=1, inplace = True)

testing.drop('LotFrontage', axis=1, inplace = True)
null_few
from sklearn.preprocessing import Imputer

Imputer = Imputer(strategy = 'median')
training["GarageYrBlt"].fillna(training["GarageYrBlt"].median(), inplace = True)

testing["GarageYrBlt"].fillna(testing["GarageYrBlt"].median(), inplace = True)

training["MasVnrArea"].fillna(training["MasVnrArea"].median(), inplace = True)

testing["MasVnrArea"].fillna(testing["MasVnrArea"].median(), inplace = True)
train_types = training.dtypes

test_types = testing.dtypes
num_train = train_types[(train_types==int) | (train_types==float)]

cat_train = train_types[train_types==object]

num_test = test_types[(test_types==int) | (test_types==float)]

cat_test = test_types[test_types==object]

num_trainval = list(num_train.index)

num_testval = list(num_test.index)

cat_trainval = list(cat_train.index)

cat_testval = list(cat_test.index)
fill_num = []



for i in num_trainval:

    if i in list(null_few.index):

        fill_num.append(i)
print(fill_num)
for i in fill_num:

    training[i].fillna(training[i].median(), inplace=True)

    testing[i].fillna(testing[i].median(), inplace=True)
fill_cat = []



for i in cat_trainval:

    if i in list(null_few.index):

        fill_cat.append(i)
print(fill_cat)
def most_common_term(a):

    a = list(a)

    return max(set(a), key = a.count)

most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities"]



counter = 0

for i in fill_cat:

    most_common[counter] = most_common_term(training[i])

    counter += 1
most_common_dictionary = {fill_cat[0]: [most_common[0]], fill_cat[1]: [most_common[1]], fill_cat[2]: [most_common[2]], fill_cat[3]: [most_common[3]],

                          fill_cat[4]: [most_common[4]], fill_cat[5]: [most_common[5]], fill_cat[6]: [most_common[6]], fill_cat[7]: [most_common[7]]}

most_common_dictionary
counter = 0

for i in fill_cat:  

    training[i].fillna(most_common[counter], inplace=True)

    testing[i].fillna(most_common[counter], inplace=True)

    counter += 1
training_null = pd.isnull(training).sum()

testing_null = pd.isnull(testing).sum()



null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])

null[null.sum(axis=1) > 0]
sns.distplot(training['SalePrice'])
sns.distplot(np.log(training['SalePrice']))
training["newprice"] = np.log(training["SalePrice"])
print(cat_trainval)
for i in cat_trainval:

    feature_set = set(training[i])

    for j in feature_set:

        feature_list= list(feature_set)

        training.loc[training[i]==j,i] = feature_list.index(j)

        

for i in cat_testval:

    feature_set = set(testing[i])

    for j in feature_set:

        feature_list= list(feature_set)

        testing.loc[testing[i]==j,i] = feature_list.index(j)           
training.head()
testing.info()
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import train_test_split
X_train = training.drop(['Id','SalePrice','newprice'],axis=1)

Y_train = training['newprice'].values

X_test = testing.drop(['Id'],axis=1)
train_X, val_X, train_Y, val_Y = train_test_split(X_train,Y_train,random_state=0)
model = DecisionTreeRegressor(random_state=1)

model.fit(train_X, train_Y)

val_predict = model.predict(val_X)

r2_model = r2_score(val_Y, val_predict) 

mse_model = mean_squared_error(val_Y, val_predict)

print(r2_model)

print(mse_model)

scores_model = cross_val_score(model, train_X, train_Y, scoring="r2")

print("Cross Validation Score: " ,np.mean(scores_model))
linreg = LinearRegression()

linreg.fit(train_X, train_Y)

val_predict = model.predict(val_X)

r2_linreg = r2_score(val_Y, val_predict) 

mse_linreg = mean_squared_error(val_Y, val_predict)

print(r2_linreg)

print(mse_linreg)

scores_linreg = cross_val_score(linreg, train_X, train_Y, scoring="r2")

print("Cross Validation Score: " ,np.mean(scores_linreg))
lasso = Lasso()

lasso.fit(train_X, train_Y)

val_predict = model.predict(val_X)

r2_lasso = r2_score(val_Y, val_predict) 

mse_lasso = mean_squared_error(val_Y, val_predict)

print(r2_lasso)

print(mse_lasso)

scores_lasso = cross_val_score(lasso, train_X, train_Y, scoring="r2")

print("Cross Validation Score: " ,np.mean(scores_lasso))
ridge = Ridge()

ridge.fit(train_X, train_Y)

val_predict = model.predict(val_X)

r2_ridge = r2_score(val_Y, val_predict) 

mse_ridge = mean_squared_error(val_Y, val_predict)

print(r2_ridge)

print(mse_ridge)

scores_ridge = cross_val_score(ridge, train_X, train_Y, scoring="r2")

print("Cross Validation Score: " ,np.mean(scores_ridge))
random = RandomForestRegressor()

random.fit(train_X, train_Y)

val_predict = model.predict(val_X)

r2_random = r2_score(val_Y, val_predict) 

mse_random = mean_squared_error(val_Y, val_predict)

print(r2_random)

print(mse_random)

scores_random = cross_val_score(random, train_X, train_Y, scoring="r2")

print("Cross Validation Score: " ,np.mean(scores_random))
model_performances = pd.DataFrame({'Model': ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],

                                  "R Squared": [r2_linreg, r2_ridge, r2_lasso, r2_model, r2_random],

                                  "MSE": [mse_linreg,mse_ridge, mse_lasso, mse_model, mse_random],

                                  "Cross_val": [np.mean(scores_linreg), np.mean(scores_ridge),np.mean(scores_lasso),np.mean(scores_model),np.mean(scores_random)]})

model_performances.sort_values(by='Cross_val', ascending = False)

ridge.fit(X_train, Y_train)
submission_predictions = np.exp(ridge.predict(X_test))
submission = pd.DataFrame({'Id': testing['Id'],

                          'SalePrice': submission_predictions})

submission.to_csv("Housing_prices.csv", index=False)

print(submission.shape)