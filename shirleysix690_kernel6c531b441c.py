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
#import tensorflow as tf 

#import tensorflow.keras as tfk

#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#tf.config.experimental_connect_to_cluster(tpu)

#tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy

#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from scipy import stats

from sklearn.linear_model import Lasso

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler

import sys

from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



print('ready')

df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_train.head()
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_test.head()
y=df_train['SalePrice']

sns.distplot(y)
#y_test=df_test['SalePrice']
y=np.log1p(y)

sns.distplot(y)
#y_test=np.log1p(y_test)

#sns.distplot(y_test)
X=df_train.drop(['Id','SalePrice'],axis=1)

X.head()
X_test=df_test.drop(['Id'],axis=1)

X_test.head()
for column in X.columns:

    print(column, X[column].dtype)
#missing data

total = X.isnull().sum().sort_values(ascending=False)

percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
X.PoolQC.unique()
X.PoolArea.describe()
X.Alley.unique()
#missing data



percent_nogarage =1- X[X['GarageArea']>0]['GarageArea'].count()/X['GarageArea'].count()

percent_nogarage
var='TotalBsmtSF'

percent_no =1- X[X[var]>0][var].count()/X[var].count()

percent_no
#drop variables with more than 15% missing values

X = X.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)

X.head()
X_test=X_test.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)

X_test.head()
X.filter(like='Yr').columns
X['GarageYrBlt']=X['GarageYrBlt'].astype('str')

X['YrSold']=X['YrSold'].astype('str')

X['MSSubClass']=X['MSSubClass'].astype('str')

X['YearBuilt']=X['YearBuilt'].astype('str')

X['YearRemodAdd']=X['YearRemodAdd'].astype('str')

X['GarageYrBlt']=X['GarageYrBlt'].astype('str')
X_test['GarageYrBlt']=X_test['GarageYrBlt'].astype('str')

X_test['YrSold']=X_test['YrSold'].astype('str')

X_test['MSSubClass']=X_test['MSSubClass'].astype('str')

X_test['YearBuilt']=X_test['YearBuilt'].astype('str')

X_test['YearRemodAdd']=X_test['YearRemodAdd'].astype('str')

X_test['GarageYrBlt']=X_test['GarageYrBlt'].astype('str')
X[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']]=X[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']].fillna("No Garage") 

X['BsmtExposure']=X['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0])

X['BsmtFinType2']=X['BsmtFinType2'].fillna(X['BsmtFinType2'].mode()[0])

X['BsmtFinType1']=X['BsmtFinType1'].fillna(X['BsmtFinType1'].mode()[0])

X['BsmtCond']=X['BsmtCond'].fillna(X['BsmtCond'].mode()[0])

X['BsmtQual']=X['BsmtQual'].fillna(X['BsmtQual'].mode()[0])

X['MasVnrType']=X['MasVnrType'].fillna(X['MasVnrType'].mode()[0])

X['MasVnrArea']=X['MasVnrArea'].fillna(X['MasVnrArea'].mean())

X['Electrical']=X['Electrical'].fillna(X['Electrical'].mode()[0])

X_test[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']]=X_test[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']].fillna("No Garage") 

X_test['BsmtExposure']=X_test['BsmtExposure'].fillna(X_test['BsmtExposure'].mode()[0])

X_test['BsmtFinType2']=X_test['BsmtFinType2'].fillna(X_test['BsmtFinType2'].mode()[0])

X_test['BsmtFinType1']=X_test['BsmtFinType1'].fillna(X_test['BsmtFinType1'].mode()[0])

X_test['BsmtCond']=X_test['BsmtCond'].fillna(X_test['BsmtCond'].mode()[0])

X_test['BsmtQual']=X_test['BsmtQual'].fillna(X_test['BsmtQual'].mode()[0])

X_test['MasVnrType']=X_test['MasVnrType'].fillna(X_test['MasVnrType'].mode()[0])

X_test['MasVnrArea']=X_test['MasVnrArea'].fillna(X_test['MasVnrArea'].mean())

X_test['Electrical']=X_test['Electrical'].fillna(X_test['Electrical'].mode()[0])
sum(X.isnull().sum())
X.head()
sum(X_test.isnull().sum())
X_test.head()
#missing data

total_test = X_test.isnull().sum().sort_values(ascending=False)

percent_test = (X_test.isnull().sum()/X_test.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])

missing_data_test.head(20)
for col in X_test.columns:

    X_test[col]=X_test[col].fillna(X_test[col].mode()[0])
X_test.head()
sum(X_test.isnull().sum())
#Log transform all float variables

for col in X.columns:

    X_test[col]=X_test[col].astype(X[col].dtypes)

    if X[col].dtypes != 'object' and X[col].max()>100 :

        X[col]=np.log1p(X[col])

        X_test[col]=np.log1p(X_test[col])

    else:

        X[col]=X[col]

        X_test[col]=X_test[col]

    

X['SaleCondition'].unique()
X_test['SaleCondition'].unique()
columns0=X.columns
columns0
X=pd.get_dummies(X)

X_test=pd.get_dummies(X_test)

X, X_test = X.align(X_test, join='outer', axis=1,fill_value=0)
dummycol= [col for col in X.columns if col not in columns0]

dummycol
sum(X.isnull().sum())
X.filter(like='MSZoning')
X_test.filter(like='MSZoning')
X.head()
X_test.head()
from sklearn import metrics

from sklearn import linear_model,decomposition

from sklearn.preprocessing import StandardScaler,FunctionTransformer

from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import GridSearchCV

import sklearn.model_selection as ms

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

from sklearn.feature_selection import SelectFromModel

pca = decomposition.PCA()

lasso1=Lasso()

    
X_train, X_test0, y_train, y_test0 = train_test_split(X, y, test_size=0.2, random_state=42)
alpha_range=np.linspace(0.00001,0.05,250)

#n_components = [20, 40, 64]

pipe = Pipeline(steps=[('scale',StandardScaler()),('mylasso',lasso1)])
#('poly', PolynomialFeatures()),('pca',pca)
#with tpu_strategy.scope():

CVlasso=GridSearchCV(pipe,dict(mylasso__alpha=alpha_range),cv=5)#pca__n_components=n_components

CVlasso.fit(X_train,y_train)
bestalpha=CVlasso.best_params_['mylasso__alpha']

print(bestalpha)
#pipe.set_params(mylasso__alpha=bestalpha)

#Lassoreg=pipe.fit(X_train,y_train)

#Lassoreg.score(X_test0,y_test0)
#y_predict0=Lassoreg.predict(X_test0)

#mse(y_test0,y_predict0)
y_predict0=CVlasso.best_estimator_.predict(X_test0)

mse(y_test0,y_predict0)
poly = PolynomialFeatures(2).fit(X_train)


X_train_poly= pd.DataFrame(poly.transform(X_train), columns = poly.get_feature_names(X_train.columns))
X_train_poly.head()
X_test0_poly= pd.DataFrame(poly.transform(X_test0), columns = poly.get_feature_names(X_train.columns))
pipe_poly = Pipeline(steps=[('scale',StandardScaler()),('mylasso',lasso1)])
scaler=StandardScaler()
sel_ = SelectFromModel(Lasso(alpha=0.00001))

sel_.fit(scaler.fit_transform(X_train_poly), y_train)
selected_feat = X_train_poly.columns[(sel_.get_support())]

print('total features: {}'.format((X_train_poly.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(

      np.sum(sel_.estimator_.coef_ == 0)))
X_train_poly_select= X_train_poly[selected_feat] 
alpha_range_poly=np.linspace(0.00001,0.05,250)


CVlasso_poly=GridSearchCV(pipe_poly,dict(mylasso__alpha=alpha_range_poly),cv=5)#pca__n_components=n_components

CVlasso_poly.fit(X_train_poly_select,y_train)
bestalpha_poly=CVlasso_poly.best_params_['mylasso__alpha']

print(bestalpha_poly)
pipe_poly.set_params(mylasso__alpha=bestalpha_poly)

LassoModel_poly=pipe_poly.fit(X_train_poly_select,y_train)
LassoModel_poly.score(X_test0_poly[selected_feat],y_test0)
y_predict0_poly=LassoModel_poly.predict(X_test0_poly[selected_feat])

mse(y_test0,y_predict0_poly)
sel_ = SelectFromModel(Lasso(alpha=bestalpha))

sel_.fit(scaler.fit_transform(X_train), y_train)
selected_feat = X_train.columns[(sel_.get_support())]

print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(

      np.sum(sel_.estimator_.coef_ == 0)))
selected_feat
from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

xgb = XGBRegressor()

xgb_p = { 'ntread':[12],

          'objective':['reg:linear'],

            'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.1,0.2,0.25,0.3], 

                      'max_depth': [3,4,5,6,7,8,9,10],

                      'min_child_weight': [4],

                      'silent': [1],

                      'subsample': [0.8],

                      'colsample_bytree': [0.7],

                      'n_estimators': [500]}#'nthread':[4]
xgb.fit(X_train,y_train)
y_xgb=xgb.predict(X_test0)

mse(y_test0,y_xgb)
CV_xgb=GridSearchCV(xgb,xgb_p,cv=5)

CV_xgb.fit(X_train[selected_feat],y_train)
CV_xgb.best_params_
y_xgb=CV_xgb.best_estimator_.predict(X_test0[selected_feat])

mse(y_test0,y_xgb)
rf=RandomForestRegressor(random_state=42)
rf_p={'n_estimators':[500],

      'max_features':['auto','sqrt'],

      'min_samples_split':[2,4,8],

      'min_samples_leaf':[3,4,5],

      'bootstrap':[True]

     }
CVrf=GridSearchCV(rf,param_grid=rf_p,cv=5)

CVrf.fit(X_train[selected_feat],y_train)
CVrf.get_params().keys()
y_rf=CVrf.best_estimator_.predict(X_test0[selected_feat])

mse(y_test0,y_rf)
from sklearn.linear_model import LinearRegression 
pipe_linear=Pipeline(steps=[('scale',StandardScaler()),('linear',LinearRegression())])

pipe_linear.fit(X_train,y_train)

y_linear=pipe_linear.predict(X_test0)

mse(y_test0,y_linear)
y_predict_lasso=CVlasso.best_estimator_.predict(X_test)


y_predict_l=np.exp(y_predict_lasso) - 1
y_predict_xgb=CV_xgb.best_estimator_.predict(X_test[selected_feat])
y_predict_x=np.exp(y_predict_xgb) - 1
y_predict=(2*y_predict_l+y_predict_x)/3
sub = pd.DataFrame()

sub['Id'] = df_test['Id']

sub['SalePrice'] = y_predict

sub.to_csv('submission.csv',index=False)