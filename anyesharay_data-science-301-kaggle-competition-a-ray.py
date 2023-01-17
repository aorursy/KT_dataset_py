# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()
train.columns
test.columns
#correlation matrix



corr_matrix=train.corr()

corr_matrix
import seaborn as sn

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(rc={'figure.figsize':(30,30)})



corr_plot=sn.heatmap(corr_matrix, annot=True)

plt.show(corr_plot)
# Pick the features with correlation > 0.55

# Essentially, we want to pick features with high correlation.

data_encoded=pd.get_dummies(train)

abs_corr=abs(data_encoded.corr()['SalePrice'])

abs_corr.sort_values(ascending=False)

attribs_encoded = data_encoded.columns[abs_corr > 0.55]

attribs_encoded = attribs_encoded[(attribs_encoded != "SalePrice")]

attribs_encoded
#cutting down train set to just what I need



new=['SalePrice','OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath',

       'GarageCars', 'GarageArea']

new_train=train[new]

new_train.head()

#This gives us relationship between all of the quantatative variables we want and saleprice. 

import seaborn as sns



sns.pairplot(new_train)
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([new_train['SalePrice'], new_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
print(new_train.isnull().values.any())
import pandas as pd

new_train=new_train.dropna()

new_train=pd.DataFrame(new_train)



new_train.head(40)

new_train.shape
features = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea']

label = 'SalePrice'



X_train = new_train[features]

y_train = new_train[label]



newtestvars=['Id','OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea']



new_test=test[newtestvars]

new_test.head()

new_test.isna().sum()

new_test=new_test.dropna()

X_test = new_test[features]

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

import xgboost as xgb

from lightgbm import LGBMRegressor

from xgboost.sklearn import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
#Polynomial Feature

poly = Pipeline([

                    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),

                    ("std_scaler", StandardScaler()),

                    ("regul_reg", Ridge(alpha=0.05, solver="cholesky")),

                ])

poly.fit(X_train, y_train)





y_pred_poly = poly.predict(X_test)

polynomialfeature= pd.DataFrame({'Id': new_test.Id, 'SalePrice': y_pred_poly})

polynomialfeature.head()
#Linear Regression



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred_lin_reg = lin_reg.predict(X_test)

linear_regression= pd.DataFrame({'Id': new_test.Id, 'SalePrice': y_pred_lin_reg})
#Random Forest

from sklearn.ensemble import RandomForestRegressor



rnd_reg = RandomForestRegressor(n_estimators=100, criterion= 'mse',

                               n_jobs = -1)

rnd_reg.fit(X_train, y_train)



y_pred_rnd_reg = rnd_reg.predict(X_test)

randomforrest_reg= pd.DataFrame({'Id': new_test.Id, 'SalePrice': y_pred_rnd_reg})

randomforrest_reg.head()
new_row = {'Id': 2121, 'SalePrice':109321}

new_row2 = {'Id': 2577, 'SalePrice':109321}

df_marks = randomforrest_reg.append(new_row, ignore_index=True)

df_marks = df_marks.append(new_row2, ignore_index=True)
rndforreg=df_marks.to_csv('anyesha_ray_randomforest_submission.csv', index=False)
#AdaBoostRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor





adb_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),n_estimators=100, loss='square', random_state=4)

adb_reg.fit(X_train, y_train)

y_pred_adb_reg = adb_reg.predict(X_test)

AdaBoost_reg= pd.DataFrame({'Id': new_test.Id, 'SalePrice': y_pred_adb_reg})

AdaBoost_reg.head()
#XGB Regression

from xgboost import XGBRegressor





xgb_reg = XGBRegressor(n_estimators=100, n_jobs=-1)

xgb_reg.fit(X_train, y_train)

y_pred_xgb_reg = xgb_reg.predict(X_test)

XGB_reg= pd.DataFrame({'Id': new_test.Id, 'SalePrice': y_pred_xgb_reg})

XGB_reg.head()
#poly=polynomialfeature.to_csv('anyesha_ray_polynomialfeature_submission.csv', index=False)

#lin=linear_regression.to_csv('anyesha_ray_linreg_submission.csv', index=False)

#rnd=randomforrest_reg.to_csv('anyesha_ray_rnd_submission.csv', index=False)

#adb=AdaBoost_reg.to_csv('anyesha_ray_adb_submission.csv', index=False)

#xgb=XGB_reg.to_csv('anyesha_ray_xgb_submission.csv', index=False)