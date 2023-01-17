import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 



from sklearn.metrics import mean_squared_error,mean_squared_log_error

from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV,cross_val_score,RepeatedKFold

from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,RobustScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import PolynomialFeatures



import xgboost as xgb

import lightgbm as lgb

import sklearn.ensemble as ensemble

import sklearn.metrics as metrics

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestRegressor,BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor

from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso, Ridge,LogisticRegressionCV,RidgeCV,LassoCV,ElasticNetCV,OrthogonalMatchingPursuit,ElasticNet,LassoLarsCV,BayesianRidge

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC,SVR

from scipy import stats

from scipy.stats import norm, skew

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import make_pipeline

from sklearn.kernel_ridge import KernelRidge





from category_encoders.ordinal import OrdinalEncoder

from category_encoders.woe import WOEEncoder

from category_encoders.target_encoder import TargetEncoder

from category_encoders.sum_coding import SumEncoder

from category_encoders.m_estimate import MEstimateEncoder

from category_encoders.leave_one_out import LeaveOneOutEncoder

from category_encoders.helmert import HelmertEncoder

from category_encoders.cat_boost import CatBoostEncoder

from category_encoders.james_stein import JamesSteinEncoder

from category_encoders.one_hot import OneHotEncoder

from scipy.special import boxcox1p

from bayes_opt import BayesianOptimization
warnings.filterwarnings('ignore')
import os

print(os.listdir("../input"))
train=pd.read_csv('../input/train_1.csv')

test=pd.read_csv('../input/test_1.csv')

submission=pd.read_csv('../input/sample_submission_1.csv')
train.head()
test.head()
submission.head()
train.describe()
train.dtypes
train.isna().sum()
#Imputing missing value with the relevant total price

train.total_price=train.total_price.fillna(469.5375)
print(train.isna().sum().sum())

print(test.isna().sum().sum())
train.columns
#New Feature Creation functions



def gen_count_id(train,test,col,name):

    temp=train.groupby(col)['record_ID'].count().reset_index().rename(columns={'record_ID':name})

    train=pd.merge(train,temp,how='left',on=col)

    test=pd.merge(test,temp,how='left',on=col)

    train[name]=train[name].astype(float)

    test[name]=test[name].astype(float)

    train[name].fillna(np.median(temp[name]),inplace=True)

    test[name].fillna(np.median(temp[name]),inplace=True)

    return train,test



def gen_average_units(train,test,col,name):

    temp=train.groupby(col)['units_sold'].mean().reset_index().rename(columns={'units_sold':name})

    train=pd.merge(train,temp,how='left',on=col)

    test=pd.merge(test,temp,how='left',on=col)

    train[name].fillna(np.median(temp[name]),inplace=True)

    test[name].fillna(np.median(temp[name]),inplace=True)

    return train,test



def gen_average_price(train,test,col,price='base_price',name='name'):

    temp=train.groupby(col)[price].mean().reset_index().rename(columns={price:name})

    train=pd.merge(train,temp,how='left',on=col)

    test=pd.merge(test,temp,how='left',on=col)

    train[name].fillna(np.median(temp[name]),inplace=True)

    test[name].fillna(np.median(temp[name]),inplace=True)

    return train,test
train,test = gen_count_id(train,test,col=['sku_id','store_id'],name='count_id_sku_store') #Genearting count of records per 'sku-id & store-id' 

train,test = gen_count_id(train,test,col=['sku_id'],name='count_id_sku') #Genearting count of records per 'sku-id'

train,test = gen_count_id(train,test,col=['store_id'],name='count_id_store') #Genearting count of records per 'store-id'



train,test = gen_average_units(train,test,col=['sku_id','store_id'],name='count_sku_store_id') #Genearting average units sold per 'sku-id & store-id'

train,test = gen_average_units(train,test,col=['store_id'],name='count_store_id') #Genearting average units sold per 'store-id'

train,test = gen_average_units(train,test,col=['sku_id'],name='count_sku_id') #Genearting average units sold per 'sku-id'



train,test = gen_average_price(train,test,col=['sku_id','store_id'],price='base_price',name='price_sku_store') #Genearting average base price per 'sku-id & store-id'

train,test = gen_average_price(train,test,col=['sku_id','store_id'],price='total_price',name='price_to_sku_store') #Genearting average total price per 'sku-id & store-id'

train,test = gen_average_price(train,test,col=['store_id'],price='base_price',name='price_store_id') #Genearting average base price per 'store-id'

train,test = gen_average_price(train,test,col=['sku_id'],price='base_price',name='price_sku_id') #Genearting average base price per 'sku-id'

train,test = gen_average_price(train,test,col=['store_id'],price='total_price',name='price_to_store_id') #Genearting average total price per 'store-id'

train,test = gen_average_price(train,test,col=['sku_id'],price='total_price',name='price_to_sku_id') #Genearting average total price per 'sku-id'
#Converting week feature

le = OrdinalEncoder()

train['week_1']=le.fit_transform(train['week'])

le = OrdinalEncoder()

test['week_1']=le.fit_transform(test['week'])+130



#Creating week number feature

train['week_num']=train.week_1%52

test['week_num']=test.week_1%52



train['week_num1']=train.week_1%4

test['week_num1']=test.week_1%4



# Encoding 'week' it using sine and cosine transform; considering it as a cyclic feature 

train['week_sin'] = np.sin(2 * np.pi * train['week_1'] / 52.143)

train['week_cos'] = np.cos(2 * np.pi * train['week_1'] / 52.143)

test['week_sin'] = np.sin(2 * np.pi * test['week_1'] / 52.143)

test['week_cos'] = np.cos(2 * np.pi * test['week_1'] / 52.143)



#Creating feature: percent difference between base price and checkout price.

train['price_diff_percent'] = (train['base_price'] - train['total_price']) / train['base_price']

test['price_diff_percent'] = (test['base_price'] - test['total_price']) / test['base_price']
train.tail()
test.head()
X=train[list(set(train.columns)-set(['record_ID','units_sold','week']))]

Y= np.log1p(train['units_sold'])

X_test=test[list(set(test.columns)-set(['record_ID','week']))]
X.dtypes
X['sku_id'] = X['sku_id'].astype('category')

X['store_id'] = X['store_id'].astype('category')
X.info()
print(len(X_test.columns))

print(len(X.columns))
print(X_test.isna().sum().sum())

print(X.isna().sum().sum())
category_list=['store_id','sku_id']
encoder_final=MEstimateEncoder()

encoder_final.fit(X[category_list], Y)



cat_enc = encoder_final.transform(X[category_list], Y)

continuous_train = X.drop(columns= category_list)

X = pd.concat([cat_enc,continuous_train],axis=1)



test_enc=encoder_final.transform(X_test[category_list])

continuous_test=X_test.drop(columns= category_list)

X_test=pd.concat([test_enc,continuous_test],axis=1)
X.head()
X.info()
X_test.head()
X.columns
del X['week_num1']
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2,random_state=23)
len(x_train.columns)
rf_base = RandomForestRegressor()

rf_base.fit(x_train,y_train)





rf_tuned = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,

                      max_features='sqrt', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=10,

                      min_weight_fraction_leaf=0.0, n_estimators=600,

                      n_jobs=None, oob_score=True, random_state=None,

                      verbose=0, warm_start=False)

rf_tuned.fit(x_train,y_train)
model_lgb_base=lgb.LGBMRegressor(objective='regression')

model_lgb_base.fit(x_train,y_train)



model_lgb_tuned=lgb.LGBMRegressor(bagging_fraction=0.8, bagging_frequency=4, boosting_type='gbdt',

              class_weight=None, colsample_bytree=1.0, feature_fraction=0.5,

              importance_type='split', learning_rate=0.1, max_depth=30,

              min_child_samples=20, min_child_weight=30, min_data_in_leaf=70,

              min_split_gain=0.0001, n_estimators=200, n_jobs=-1,

              num_leaves=1200, objective=None, random_state=None, reg_alpha=0.0,

              reg_lambda=0.0, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)



model_lgb_tuned.fit(x_train,y_train)
prediction_rfb_valid=rf_base.predict(x_valid)

prediction_rft_valid=rf_tuned.predict(x_valid)

prediction_lgbmb_valid=model_lgb_base.predict(x_valid)

prediction_lgbmt_valid=model_lgb_tuned.predict(x_valid)



rf_base_msle=100*mean_squared_log_error(y_valid,prediction_rfb_valid)

rf_tuned_msle=100*mean_squared_log_error(y_valid,prediction_rft_valid)

lgbm_base_msle=100*mean_squared_log_error(y_valid,prediction_lgbmb_valid)

lgbm_tuned_msle=100*mean_squared_log_error(y_valid,prediction_lgbmt_valid)



prediction_ensemble_base=(((1-rf_base_msle)*prediction_rfb_valid)+((1-lgbm_base_msle)*prediction_lgbmb_valid))/(2-rf_base_msle-lgbm_base_msle)

prediction_ensemble_tuned=(((1-rf_tuned_msle)*prediction_rft_valid)+((1-lgbm_tuned_msle)*prediction_lgbmt_valid))/(2-rf_tuned_msle-lgbm_tuned_msle)



ensemble_base_msle=100*mean_squared_log_error(y_valid,prediction_ensemble_base)

ensemble_tuned_msle=100*mean_squared_log_error(y_valid,prediction_ensemble_tuned)





print("RF Base: {}; RF Tuned: {}".format(rf_base_msle,rf_tuned_msle))

print("LGBM Base: {}; LGBM Tuned: {}".format(lgbm_base_msle,lgbm_tuned_msle))

print("Ensemble Base: {}; Ensemble Tuned: {}".format(ensemble_base_msle,ensemble_tuned_msle))
model = lgb.LGBMRegressor(bagging_fraction=0.8, bagging_frequency=4, boosting_type='gbdt',

              class_weight=None, colsample_bytree=1.0, feature_fraction=0.5,

              importance_type='split', learning_rate=0.1, max_depth=30,

              min_child_samples=20, min_child_weight=30, min_data_in_leaf=70,

              min_split_gain=0.0001, n_estimators=100, n_jobs=-1,

              num_leaves=1400, objective=None, random_state=None, reg_alpha=0.0,

              reg_lambda=0.0, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)



model.fit(X,Y)
X_test.head()
del X_test['week_num1']
prediction=model.predict(X_test)
final_prediction=np.round(np.expm1(prediction))

submission['units_sold']=final_prediction
submission.head()
#submission.to_csv('AV_DemandForecast_05.csv',index=False)