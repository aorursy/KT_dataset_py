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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn import preprocessing



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor



import lightgbm as lgb

import warnings


house_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

house_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
house_train.head()
house_test.head()
house_train.shape
house_test.shape

Test_ID = house_test.Id
house_train.columns

house_train.info()



sns.set_style("whitegrid")

nullsum = house_train.isnull().sum()

nullsum = nullsum[nullsum > 0]

nullsum.sort_values(inplace=True)

nullsum.plot.bar(color='#a98d19')


(nullsum/len(house_train)*100).sort_values(ascending = False).round(2).head(10)

sns.set_style("whitegrid")

test_null = house_test.isnull().sum()

test_null = test_null[test_null > 0]

test_null.sort_values(inplace=True)

test_null.plot.bar(color='#a98d19')

(test_null/len(house_test)*100).sort_values(ascending = False).round(2).head(10)
house_train = house_train.drop(columns={"PoolQC","MiscFeature","Alley","Fence","FireplaceQu","Id"})

house_train.set_index("SalePrice")

house_test = house_test.drop(columns={"PoolQC","MiscFeature","Alley","Fence","FireplaceQu","Id"})



house_train.LandContour.value_counts()
house_train.Utilities.value_counts()
house_train['MSSubClass'].value_counts()
house_train = house_train.drop(columns="Utilities")

house_test = house_test.drop(columns="Utilities")

house_train['LotFrontage'].fillna(house_train['LotFrontage'].mean(),inplace=True)

house_train.fillna(method="bfill", inplace=True)

house_train.isnull().sum().sort_values(ascending = False)
house_test['LotFrontage'].fillna(house_test['LotFrontage'].mean(),inplace=True)

house_test.fillna(method="bfill", inplace=True)

house_test.isnull().sum().sort_values(ascending = False)
house_train.shape
#correlation matrix

corr = house_train.corr()

plt.subplots(figsize=(15,10))

#saleprice correlation matrix

k=15

cols= corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm= np.corrcoef(house_train[cols].values.T)

sns.set(font_scale=1.25)

sns.heatmap(cm, annot=True,

                fmt='.2f', annot_kws={'size':10},cmap='BrBG',

                yticklabels=cols.values,

                xticklabels=cols.values)

numeric = [f for f in house_train.columns if house_train.dtypes[f] != 'object']

numeric.remove('SalePrice')

categorical = [f for f in house_train.columns if house_train.dtypes[f] == 'object']

numeric_df = pd.DataFrame(house_train[numeric])



numeric_df= pd.DataFrame(preprocessing.normalize(numeric_df),columns=numeric)

numeric_df[0:5]
numeric_t = [f for f in house_test.columns if house_test.dtypes[f] != 'object']



categorical_test = [f for f in house_test.columns if house_test.dtypes[f] == 'object']

numeric_test = pd.DataFrame(house_test[numeric_t])



numeric_test= pd.DataFrame(preprocessing.normalize(numeric_test),columns=numeric)

numeric_test[0:5]
from sklearn.preprocessing import LabelEncoder

house_train[categorical]=house_train[categorical].apply(LabelEncoder().fit_transform)

house_train[categorical]
from sklearn.preprocessing import LabelEncoder

house_test[categorical_test]=house_test[categorical_test].apply(LabelEncoder().fit_transform)

house_test[categorical_test]
X_train = pd.concat([numeric_df,house_train[categorical]], axis = 1, sort=False)

X_train.head()
Y_train = house_train['SalePrice']

Y_train.head()
X_test = pd.concat([numeric_test,house_test[categorical_test]], axis = 1, sort=False)

X_test.head()

X_test.shape
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, Y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
from sklearn.model_selection import GridSearchCV

xgb = XGBRegressor()

parameters = {'objective':['reg:squarederror'],

            'learning_rate': [.1],

              'max_depth': [3],

              'min_child_weight': [2],

              'subsample': [0.7],

              'colsample_bytree': [.7],

              'colsample_bylevel':[.7],

              'alpha' : [.05],

              'lambda' : [.3],

              'n_estimators': [2500]}

XGB = GridSearchCV(xgb,

                        parameters,

                        cv = 2)

XGB.fit(X_train,

         Y_train)
print(XGB.best_score_)

print(XGB.best_params_)
ypredXG = XGB.predict(X_test)
XGB_Accuracy = "{:.2f} %".format(XGB.score(X_train,Y_train)*100)

print("Accuracy: ",XGB_Accuracy)
from sklearn.ensemble import RandomForestRegressor

RFReg_Price = RandomForestRegressor(n_estimators= 1000, random_state=100)

RFReg_Price.fit(X_train, Y_train)

Y_test = RFReg_Price.predict(X_test)

RFReg_Accuracy = "{:.2f} %".format(RFReg_Price.score(X_train,Y_train)*100)

print("Accuracy: ",RFReg_Accuracy)
from sklearn.linear_model import Ridge

from sklearn.linear_model import RidgeCV

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

ridge_alphas = [13.5, 14, 14.5, 15, 15.5]

RR_Price = make_pipeline(RobustScaler(),

                      RidgeCV(alphas=ridge_alphas, cv=kfolds))

RR_Price.fit(X_train, Y_train)

Y_test_RR = RR_Price.predict(X_test)

RR_Accuracy = "{:.2f} %".format(RR_Price.score(X_train,Y_train)*100)

print("Accuracy: ",RR_Accuracy)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)

lasso.fit(X_train,Y_train)

Y_lasso = lasso.predict(X_test)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
submission = pd.DataFrame({

        "Id": Test_ID,

        "SalePrice": ypredXG

    })

submission.to_csv('submission.csv', index=False)
submission1 = pd.DataFrame({

        "Id": Test_ID,

        "SalePrice": Y_test

    })

submission1.to_csv('submission1.csv', index=False)
submission2 = pd.DataFrame({

        "Id": Test_ID,

        "SalePrice": Y_lasso

    })

submission2.to_csv('submission2.csv', index=False)