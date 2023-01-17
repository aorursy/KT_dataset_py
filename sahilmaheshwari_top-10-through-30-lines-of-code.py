# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

import lightgbm as lgb

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PolynomialFeatures 

from scipy.stats import skew

from scipy.stats.stats import pearsonr



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

#test.describe()
#train.info()
#test.info()
corr_matrix=train.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)
train['AllflrSF']=train['TotalBsmtSF']+train['1stFlrSF']+train['2ndFlrSF']

corr_matrix=train.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes=["SalePrice","OverallQual","GrLivArea","GarageCars","YearBuilt","AllflrSF"]

scatter_matrix(train[attributes],figsize=(20,14))

plt.show()
#preprocessing  data

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))

all_data['AllflrSF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']

#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
import xgboost as xgb



dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model1=RandomForestRegressor()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)

model1.fit(X_train,y)

rf_preds=np.expm1(model1.predict(X_test))

xgb_preds = np.expm1(model_xgb.predict(X_test))

final_preds=rf_preds+xgb_preds/2

test_id=test['Id']

output = pd.DataFrame( {"Id" :test_id ,'SalePrice': final_preds})

output.to_csv('my_submission.csv', index=False)

print(output)

print("Your submission was successfully saved!")