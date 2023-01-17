import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_columns', 999)#Showing All the columns
train_org = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train_org.head()
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()
train=train_org.drop(labels=['SalePrice'],axis=1)
df = train.append(test,ignore_index=True)
df.head()
df.shape
train = df[:1460]

test = df[1460:]
len(df)
df.isnull().sum().sort_values(ascending=False).head(30)
new_df = df.fillna({

    'PoolQC': 'No PoolQc',

    'MiscFeature': 'No MiscFeature',

    'Alley':'No Alley',

    'Fence':'No Fence',

    'FireplaceQu':'No FireplaceQu',

    'LotFrontage':60.0,

    'GarageCond':'No GarageCond',

    'GarageQual':'No GarageQual',

    'GarageYrBlt':2005.0,

    'GarageFinish':'No GarageFinish',

    'GarageType':'No GarageType',

    'BsmtCond':'No BsmtCond',

    'BsmtExposure':'No BsmtExposure',

    'BsmtQual':'No BsmtQual',

    'BsmtExposure':'No BsmtExposure',

    'BsmtQual':'No BsmtQual',

    'BsmtFinType1':'No BsmtFinType1',

    'BsmtFinType2':'No BsmtFinType2',

    'MasVnrType':'No MasVnrType',

    'MasVnrArea':0.0,

    'MSZoning':'No MSZoning',

    'BsmtHalfBath':0.0,

    'Utilities':'No Utilities',

    'Functional':'No Functional',

    'BsmtFullBath':0.0,

    'BsmtFinSF1':0.0,

    'Exterior1st':'No Exterior1st',

    'Exterior2nd':'No Exterior2nd',

    'BsmtFinSF2':0.0,

    'BsmtUnfSF':0.0,

    'TotalBsmtSF':0.0,

    'SaleType':'No SaleType',

    'GarageCars':2.0,

    'Electrical':'No Electrical',

    'GarageArea':0.0,

    'KitchenQual':'No KitchenQual'

})
df['KitchenQual'].mode()
new_df.isnull().sum().sort_values(ascending=False).head(30)
new_df['TotalSf'] = new_df['TotalBsmtSF'] + new_df['1stFlrSF'] + new_df['2ndFlrSF']

new_df = new_df.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF','Id'],axis=1)
from sklearn.preprocessing import LabelEncoder
df_new = new_df.select_dtypes(exclude=['int64', 'float64'])

df_new1 = new_df.select_dtypes(include=['int64', 'float64'])

df_encoded = df_new.apply(LabelEncoder().fit_transform)

new_data = df_new1.join(df_encoded )

new_data.dtypes
import xgboost as xgb

from sklearn.metrics import mean_squared_error
train=new_data[:1460]

test=new_data[1460:]
test.head()
from sklearn.metrics import mean_squared_error

X, y = train,train_org['SalePrice']
X.describe()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn import metrics
#Creating training Model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#calculating r squared

SS_Residual = sum((y_test-predictions)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

print('R Squared:', r_squared)
#regression plot of the real test values versus the predicted values



plt.figure(figsize=(16,8))

sns.regplot(y_test,predictions)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Linear Model Predictions")

plt.grid(False)

plt.show()
#Ridge Regression

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



ridge=Ridge()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X,y)
prediction_ridge=ridge_regressor.predict(X_test)
print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)
import seaborn as sns



sns.distplot(y_test-prediction_ridge)
plt.figure(figsize=(16,8))

sns.regplot(y_test,prediction_ridge)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Ridge Model Predictions")

plt.grid(False)

plt.show()
y_test[:20]
prediction_ridge[:20]
SS_Residual = sum((y_test-prediction_ridge)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

print('R Squared:', r_squared)
#Lasso Regression

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso=Lasso()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)



lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)
prediction_lasso=lasso_regressor.predict(X_test)
import seaborn as sns



sns.distplot(y_test-prediction_lasso)
plt.figure(figsize=(16,8))

sns.regplot(y_test,prediction_lasso)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Lasso Model Predictions")

plt.grid(False)

plt.show()
SS_Residual = sum((y_test-prediction_lasso)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

print('R Squared:', r_squared)