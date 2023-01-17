import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
train.shape
#feature description > https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

train.columns
train['SalePrice'].describe() # Notice the minimam price is larger than zero?
import matplotlib.pyplot as plt

import seaborn as sns



#histogram

sns.distplot(train['SalePrice']);
sns.distplot(np.log(train["SalePrice"]))
train["TransformedPrice"] = np.log(train["SalePrice"])
correlations = train.corr()

correlations = correlations["SalePrice"].sort_values(ascending=False)

correlations #feature description > https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
#scatter plot OverallQual/saleprice

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# Violinplot would give you more info for discrete features

plt.figure(figsize=(12, 6))

g = sns.violinplot(x=var, y='SalePrice', data=train)

g.set(ylim=(0, 800000))

g.set_xticklabels(g.get_xticklabels(), rotation=90);
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'GarageCars' #Other highly correlated features > GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt



#for continuous values

#data = pd.concat([train['SalePrice'], train[var]], axis=1)

#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



#for discrete values

plt.figure(figsize=(12, 6))

g = sns.violinplot(x=var, y='SalePrice', data=train)

g.set(ylim=(0, 800000))

g.set_xticklabels(g.get_xticklabels(), rotation=90);
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)
# You may think Total Basement Square Feet == 0 can be additional feature

sns.distplot(train['TotalBsmtSF'])
# ..but it's already included as nan in another feature :BsmtQual 

train['BsmtQual'].unique()
houses=pd.concat([train,test], sort=False)
#missing data

total = houses.select_dtypes(include='object').isnull().sum().sort_values(ascending=False)

percent = (houses.select_dtypes(include='object').isnull().sum()/houses.select_dtypes(include='object').isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    train[col]=train[col].fillna('None')

    test[col]=test[col].fillna('None')
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    train[col]=train[col].fillna(train[col].mode()[0])

    test[col]=test[col].fillna(train[col].mode()[0])
#missing data

total = houses.select_dtypes(include=['int','float']).isnull().sum().sort_values(ascending=False)

percent = (houses.select_dtypes(include=['int','float']).isnull().sum()/houses.select_dtypes(include=['int','float']).isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)
for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):

    train[col]=train[col].fillna(0)

    test[col]=test[col].fillna(0)
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())

test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())
# just to check we now don't have any nan value

print(train.isnull().sum().sum())

print(test.isnull().sum().sum())
#finding outliers

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# removing outliers

# train.sort_values(by = 'GrLivArea', ascending = False)[:2]

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
len_train=train.shape[0]

print(train.shape)
houses=pd.concat([train,test], sort=False)



# Transform numerical to categorical (*see data description)

houses['MSSubClass']=houses['MSSubClass'].astype(str)
# Categorical to one hot encoding

houses=pd.get_dummies(houses)

train=houses[:len_train]

test=houses[len_train:]
# Define Training/Test sets

X_train = train.drop(["Id", "SalePrice", "TransformedPrice"], axis=1)

y_train = train["TransformedPrice"]

X_test = test.drop(["Id", "SalePrice", "TransformedPrice"], axis=1)
# Split into Validation

from sklearn.model_selection import train_test_split #to create validation data set



X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
from sklearn import ensemble



my_model = ensemble.GradientBoostingRegressor()

my_model.fit(X_training, y_training)
# make predictions

predictions = my_model.predict(X_valid)



from sklearn.metrics import mean_squared_error

print("Mean Absolute Error : " + str(np.sqrt(mean_squared_error(predictions, y_valid))))
submission_predictions = np.exp(my_model.predict(X_test))
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": submission_predictions

    })



submission.to_csv("prices.csv", index=False)

print(submission.shape)