import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head(5)
train.shape
train.select_dtypes(exclude=['object']).columns
len(train.select_dtypes(exclude=['object']).columns)
train.describe(include='all')
train.select_dtypes(include=['object']).columns
len(train.select_dtypes(include=['object']).columns)
#It appears to be good practice to minimise the skew of the dataset.

#The reason often given is that skewed data adversely affects the prediction accuracy of regression models.

#Note: While important for linear regression, correcting skew is not necessary for Decisions Trees and Random Forests.



sns.distplot(train['SalePrice'], color='r')
sns.distplot(np.log(train.SalePrice), color='g')
train.SalePrice.skew()
np.log(train.SalePrice).skew()
numerical_data = train.select_dtypes(exclude=['object']).drop('SalePrice', axis=1).copy()
numerical_data
#scatter plots for target versus numerical attributes

target = train.SalePrice



f = plt.figure(figsize=(12,20))



for i in range(len(numerical_data.columns)):

         f.add_subplot(13, 3, i+1)

         sns.scatterplot(numerical_data.iloc[:,i], target)

plt.tight_layout()

plt.show()



#Based on a first viewing of the scatter plots against SalePrice, there appears to be:



#A few outliers on the LotFrontage (say, >200) and LotArea (>100000) data.

#BsmtFinSF1 (>4000) and TotalBsmtSF (>6000)

#1stFlrSF (>4000)

#GrLivArea (>4000 AND SalePrice <300000)

#LowQualFinSF (>550)
#Below is a heatmap of the correlation of the numerical columns:

correlation_matrix = train.corr()

fig = plt.figure(figsize=(20,8))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
correlation_matrix['SalePrice'].sort_values(ascending=False).head(15)
#checking for missi

numerical_data.isnull().sum().sort_values(ascending=False)
train.select_dtypes(include=['object']).columns
#count plot for kitchenql

f, ax = plt.subplots(figsize=(10,6))

sns.barplot(x=train.KitchenQual, y=train.SalePrice)
#count plot for neigbourhood

f, ax = plt.subplots(figsize=(35,20))

sns.barplot(x=train.Neighborhood, y=train.SalePrice)
categorical_val = train.select_dtypes(include=['object']).columns
train[categorical_val].isnull().sum().sort_values(ascending=False)

#looking for missing values in categorical varaible
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
train_data = train.copy()
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)
cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',

                     'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',

                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',

                     'MasVnrType']
for items in cols:

    train_data[items] = train_data[items].fillna('None')
train_data.isnull().sum().sort_values(ascending=False).head(6)
#taking care of outliers. outliers are not good for regression so we have to remove them

train_data = train_data.drop(train_data['LotFrontage']

                            [train_data['LotFrontage'] > 200].index)



train_data = train_data.drop(train_data['LotArea']

                            [train_data['LotArea'] > 1000000].index)



train_data = train_data.drop(train_data['BsmtFinSF1']

                            [train_data['BsmtFinSF1'] > 4000].index)



train_data = train_data.drop(train_data['TotalBsmtSF']

                            [train_data['TotalBsmtSF'] > 6000].index)



train_data = train_data.drop(train_data['1stFlrSF']

                            [train_data['1stFlrSF'] > 4000].index)



train_data = train_data.drop(train_data['GrLivArea']

                                     [(train_data['GrLivArea']>4000) & 

                                      (target<300000)].index)



train_data = train_data.drop(train_data['LowQualFinSF']

                            [train_data['LowQualFinSF'] > 550].index)
train_data['SalePrice'] = np.log(train_data['SalePrice'])

train_data = train_data.rename(columns={'SalePrice' : 'SalePrice_log'})
train_data = train_data.drop(['Id'], axis=1)
correlation_matrix = train_data.corr()

fig = plt.figure(figsize=(20,8))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
correlation_matrix['SalePrice_log'].sort_values(ascending=False).head(15)
# Remove attributes that were identified for excluding when viewing scatter plots & corr values

must_drop = ['SalePrice_log', 'MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 

                   'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd'] # high corr with other attributes
X = train_data.drop(must_drop, axis=1)

y = train_data['SalePrice_log']
X = pd.get_dummies(X)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

val_X = my_imputer.transform(val_X)
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor


def inv_y(transformed_y):

    return np.exp(transformed_y)

# Series to collate mean absolute errors for each algorithm

mae_compare = pd.Series()

mae_compare.index.name = 'Algorithm'
reg_forest = RandomForestRegressor(n_estimators = 100,

                                  criterion = 'mse',

                                  random_state=0)

reg_forest.fit(train_X, train_y)

forest_pred = reg_forest.predict(val_X)

forest_val_mae = mean_absolute_error(inv_y(forest_pred), inv_y(val_y))



mae_compare['RandomForest'] = forest_val_mae
linear_reg = LinearRegression()

linear_reg.fit(train_X, train_y)

linear_pred = linear_reg.predict(val_X)

linear_val_mae = mean_absolute_error(inv_y(linear_pred), inv_y(val_y))



mae_compare['LinearRegression'] = linear_val_mae



lasso_model = Lasso(alpha=0.0005, random_state=5)

lasso_model.fit(train_X, train_y)

lasso_val_predictions = lasso_model.predict(val_X)

lasso_val_mae = mean_absolute_error(inv_y(lasso_val_predictions), inv_y(val_y))

mae_compare['Lasso'] = lasso_val_mae

print('MAE values for different algorithms:')

mae_compare.sort_values(ascending=True).round()
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_data = test.copy()
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)
for items in cols:

    test_data[items] = test_data[items].fillna('None')
if 'SalePrice_log' in must_drop:

    must_drop.remove('SalePrice_log')
test_data = test.drop(must_drop, axis=1)
test_data = pd.get_dummies(test_data)
# Ensure test data is encoded in the same manner as training data with align command

final_train, final_test = X.align(test_data, join='left', axis=1)
final_test_imputed = my_imputer.transform(final_test)
final_result = Lasso(alpha=0.0005, random_state=0)
imputed_train = my_imputer.fit_transform(final_train)
final_result.fit(imputed_train, y)
test_preds = final_result.predict(final_test_imputed)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': inv_y(test_preds)})



output.to_csv('submission.csv', index=False)
#please upvote if you like 