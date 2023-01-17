import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd  # data frame operations  

import numpy as np  # arrays and math functions

import matplotlib.pyplot as plt  # static plotting

import seaborn as sns  # pretty plotting, including heat map

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
housing_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

housing_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

housing_combined = pd.concat((housing_train, housing_test), sort=False).reset_index(drop = True) # Imputations will be done on combined dataset

housing_sale_price_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
housing_train.head()
housing_train.shape
housing_test.shape
housing_combined.shape
housing_sale_price_test.shape
housing_train.describe()
# correlation heat map setup for seaborn

def corr_chart(df_corr):

    corr=df_corr.corr()

    #screen top half to get a triangle

    top = np.zeros_like(corr, dtype=np.bool)

    top[np.triu_indices_from(top)] = True

    fig=plt.figure()

    fig, ax = plt.subplots(figsize=(40,40))

    sns.heatmap(corr, mask=top, cmap='coolwarm', 

        center = 0, square=True, 

        linewidths=.5, cbar_kws={'shrink':.5}, 

        annot = True, annot_kws={'size': 9}, fmt = '.5f')           

    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)

    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)

    plt.title('Correlation Heat Map')

  

np.set_printoptions(precision=3)
corr_chart(df_corr = housing_train)
refined_housing_train = housing_train.filter(['OverallQual', 'TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars'], axis=1)

refined_housing_test = housing_test.filter(['OverallQual', 'TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars'], axis=1)

refined_housing_train.head()
# Identifying missing values

# Reference function from this submitted kernel - https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing#Feature-engineering



def missing_percentage(df):

    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""

    ## the two following line may seem complicated but its actually very simple. 

    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])



#missing_percentage(housing_combined)
#missing_percentage(refined_housing_train)
#Handling Missing values - Pass 1 - Replacing empty values with 'None' where appropriate

missing_value_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageFinish','GarageQual', 'GarageYrBlt', 'GarageCond', 'GarageType', 

                         'BsmtCond', 'BsmtExposure','BsmtQual', 'BsmtFinType2', 'BsmtFinType1','MasVnrType']

for i in missing_value_columns:

    housing_combined[i] = housing_combined[i].fillna('None')
missing_percentage(housing_combined)
#Handling Missing values - Pass 2 - Replace categorical data with dummy variables

#Dealing with categorical data

housing_combined = pd.get_dummies(housing_combined)

housing_combined = housing_combined.fillna(housing_combined.mean())
missing_percentage(housing_combined)
#set up the model

X_train = housing_combined[:housing_train.shape[0]]

X_test = housing_combined[housing_train.shape[0]:]

y_train = housing_train.SalePrice

y_test = housing_sale_price_test.SalePrice



#refined_housing = housing_combined.filter(['OverallQual', 'TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars'], axis=1)



#X_train = refined_housing[:housing_train.shape[0]]

#X_test = refined_housing[housing_train.shape[0]:]

#y_train = housing_train.SalePrice

# = housing_sale_price_test.SalePrice
# Linear Regression Model Creation

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)



#predict on test data

y_test_predictions_lin_reg = lin_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, y_test_predictions_lin_reg)

lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1,solver="cholesky")

ridge_reg.fit(X_train, y_train)



#predict on test data

y_test_predictions_ridge_reg = ridge_reg.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_test_predictions_ridge_reg)

ridge_rmse = np.sqrt(ridge_mse)

print(ridge_rmse)
from sklearn.linear_model import Lasso

lasso_reg = Ridge(alpha=0.1)

lasso_reg.fit(X_train, y_train)



#predict on test data

y_test_predictions_lasso_reg = lasso_reg.predict(X_test)

lasso_mse = mean_squared_error(y_test, y_test_predictions_lasso_reg)

lasso_rmse = np.sqrt(lasso_mse)

print(lasso_rmse)
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

regr.fit(X_train, y_train)



#predict on test data

y_test_predictions_randomforest_reg = regr.predict(X_test)

randomforest_mse = mean_squared_error(y_test, y_test_predictions_randomforest_reg)

randomforest_rmse = np.sqrt(randomforest_mse)

print(randomforest_rmse)

submission = pd.DataFrame({'Id': housing_test.Id, 'SalePrice': y_test_predictions_randomforest_reg})

submission.to_csv('submission.csv', index=False)