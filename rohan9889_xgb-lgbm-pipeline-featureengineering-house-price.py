import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
plt.figure(figsize=(16, 10))

sns.heatmap(train_data.corr(), cmap='RdBu')
plt.figure(figsize=(16, 10))

train_data.corr()['SalePrice'].sort_values().plot(kind='barh')
obj_list = []

for i in train_data.columns :

    if train_data[i].dtypes == 'object':

        obj_list.append(i)
train_data[obj_list].info()
train_data.drop(['PoolQC', 'Utilities', 'Street', 'MiscFeature'], axis=1, inplace=True)

test_data.drop(['PoolQC', 'Utilities', 'Street', 'MiscFeature'], axis=1, inplace=True)
# Adding status features based on other features

# eg. if PoolArea is present then PoolStatus is one



train_data['PoolStatus'] = train_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train_data['SeFlrStatus'] = train_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

train_data['GarageStatus'] = train_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

train_data['BsmtStatus'] = train_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

train_data['FirePlaceStatus'] = train_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



test_data['PoolStatus'] = test_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

test_data['SeFlrStatus'] = test_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

test_data['GarageStatus'] = test_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

test_data['BsmtStatus'] = test_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test_data['FirePlaceStatus'] = test_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



# Info about a particular feature is spread across multiple features

# Combining those to build new features

train_data['RemodNBuild']=train_data['YearBuilt']+train_data['YearRemodAdd']

train_data['SF']=train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']

train_data['AreaInFt'] = (train_data['BsmtFinSF1'] + train_data['BsmtFinSF2'] +

                                 train_data['1stFlrSF'] + train_data['2ndFlrSF'])

train_data['NumBathroom'] = (train_data['FullBath'] + (0.5 * train_data['HalfBath']) +

                               train_data['BsmtFullBath'] + (0.5 * train_data['BsmtHalfBath']))

train_data['Total_porch_sf'] = (train_data['OpenPorchSF'] + train_data['3SsnPorch'] +

                              train_data['EnclosedPorch'] + train_data['ScreenPorch'] +

                              train_data['WoodDeckSF'])



test_data['RemodNBuild']=test_data['YearBuilt']+test_data['YearRemodAdd']

test_data['SF']=test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']

test_data['AreaInFt'] = (test_data['BsmtFinSF1'] + test_data['BsmtFinSF2'] +

                                 test_data['1stFlrSF'] + test_data['2ndFlrSF'])

test_data['NumBathroom'] = (test_data['FullBath'] + (0.5 * test_data['HalfBath']) +

                               test_data['BsmtFullBath'] + (0.5 * test_data['BsmtHalfBath']))

test_data['Total_porch_sf'] = (test_data['OpenPorchSF'] + test_data['3SsnPorch'] +

                              test_data['EnclosedPorch'] + test_data['ScreenPorch'] +

                              test_data['WoodDeckSF'])
ID = test_data['Id']

del test_data['Id']
del train_data['Id']
from sklearn_pandas import DataFrameMapper, CategoricalImputer

from sklearn.pipeline import Pipeline, FeatureUnion

import xgboost as xgb

from sklearn.preprocessing import Imputer, FunctionTransformer, MinMaxScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction import DictVectorizer
quasi_constant = []

for i in train_data.columns:

    counts = train_data[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(train_data) * 100 > 99.94:

        quasi_constant.append(i)

train_data = train_data.drop(quasi_constant, axis=1)

test_data = test_data.drop(quasi_constant, axis=1)
list_of_numerical = []

list_of_categorical = []

for i in train_data.columns:

    if train_data[i].dtypes == 'object':

        list_of_categorical.append(i)

    else:

        if i == 'SalePrice':

            pass

        else:

            list_of_numerical.append(i)
transformers = []



transformers.extend([([num_feat],SimpleImputer(strategy='constant',fill_value=-1)) for num_feat in list_of_numerical])

transformers.extend([(cat_feat,CategoricalImputer()) for cat_feat in list_of_categorical])

combined_pipeline = DataFrameMapper(transformers,

                                   input_df=True,

                                   df_out=True)
def return_dict(blob):

    return blob.to_dict("records")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    train_data.drop(['SalePrice'],axis=1), train_data['SalePrice'], test_size=0.15, random_state=42)
pipeline1 = Pipeline([

    ('featureunion',combined_pipeline),

    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),

    ('vectorizer',DictVectorizer(sort=False,sparse=False)),

    ('reg',xgb.XGBRegressor(objective="reg:squarederror"))

])
mse1 = cross_val_score(pipeline1, X_train, y_train, scoring="neg_mean_squared_log_error", cv=5)
print(np.sqrt(-mse1))
from sklearn.model_selection import RandomizedSearchCV
params = {

    'reg__max_depth' : [5,10,15,20],

    'reg__gamma' : [.1,.2,.3,.4,.5,.6],

    'reg__colsample_bytree' : [.1,.2,.3,.4,.5,.6,.7],

    'reg__n_estimators' : [50,100,200]

}
grid_pipe_xgboost = RandomizedSearchCV(estimator=pipeline1,param_distributions=params,cv=4,

                               scoring="neg_mean_squared_log_error",n_iter=15)
%%time

grid_pipe_xgboost.fit(X_train,y_train)
grid_pipe_xgboost.best_params_
grid_pipe_xgboost.best_score_
pipe_predict_xgboost = grid_pipe_xgboost.predict(X_test)
from sklearn.metrics import mean_squared_log_error
print(np.sqrt(mean_squared_log_error(y_test,pipe_predict_xgboost)))
import lightgbm as lgb
pipeline_lgm = Pipeline([

    ('featureunion',combined_pipeline),

    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),

    ('vectorizer',DictVectorizer(sort=False,sparse=False)),

    ('reg_lgb',lgb.LGBMRegressor())

])
%%time

pipeline_lgm.fit(X_train,y_train)
pipeline_lgm_predict = pipeline_lgm.predict(X_test)

print(np.sqrt(mean_squared_log_error(y_test,pipeline_lgm_predict)))
params_lgb = {

    'reg_lgb__learning_rate' : [.1,.2,.3,.4,.5,.6],

    'reg_lgb__n_estimators' : [50,100,200],

    'reg_lgb__colsample_bytree' : [.1,.2,.3,.4,.5,.6,.7],

    'reg_lgb__reg_lambda' : np.arange(0,0.6,0.1)

}
grid_pipe_lgm = RandomizedSearchCV(estimator=pipeline_lgm,param_distributions=params_lgb,cv=4,

                               scoring="neg_mean_squared_log_error",n_iter=15)
%%time

grid_pipe_lgm.fit(X_train,y_train)
grid_pipe_lgm.best_params_
grid_pipe_lgm.best_score_
lgm_pipe_predict = grid_pipe_lgm.predict(X_test)

print(np.sqrt(mean_squared_log_error(y_test,lgm_pipe_predict)))
from sklearn.linear_model import ElasticNet

pipeline_elasticnet = Pipeline([

    ('featureunion',combined_pipeline),

    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),

    ('vectorizer',DictVectorizer(sort=False,sparse=False)),

    ('reg_en',ElasticNet())

])
pipeline_elasticnet.fit(X_train,y_train)
predicted_base_elastic_net = pipeline_elasticnet.predict(X_test)

print(np.sqrt(mean_squared_log_error(y_test,predicted_base_elastic_net)))
params_elasticnet = {

    'reg_en__l1_ratio' : [0.2,0.4,0.6,0.8],

    'reg_en__alpha' : [0.5,1,1.5,2],

}
grid_pipe_elastic = RandomizedSearchCV(estimator=pipeline_elasticnet,param_distributions=params_elasticnet,cv=4,

                               scoring="neg_mean_squared_log_error")
%%time

grid_pipe_elastic.fit(X_train,y_train)
grid_pipe_elastic.best_params_
grid_pipe_elastic.best_score_
from sklearn.linear_model import Lasso
lasso = Lasso()
pipeline_lasso = Pipeline([

    ('featureunion',combined_pipeline),

    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),

    ('vectorizer',DictVectorizer(sort=False,sparse=False)),

    ('reg_lasso',Lasso())

])
params_lasso = {

    'reg_lasso__alpha' :  np.logspace(-4, -3, 5),

}
grid_pipe_lasso = RandomizedSearchCV(estimator=pipeline_lasso,param_distributions=params_lasso,cv=4,

                               scoring="neg_mean_squared_error")
%%time

grid_pipe_lasso.fit(X_train,y_train)
grid_pipe_lasso.best_params_
grid_pipe_lasso.best_score_
print(np.sqrt(mean_squared_log_error(y_test,grid_pipe_lasso.predict(X_test))))
from sklearn.ensemble import GradientBoostingRegressor
gbdt_model = GradientBoostingRegressor(learning_rate=0.05, min_samples_leaf=5,

                                       min_samples_split=10, max_depth=4, n_estimators=3000)
pipeline_gbt = Pipeline([

    ('featureunion',combined_pipeline),

    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),

    ('vectorizer',DictVectorizer(sort=False,sparse=False)),

    ('reg_lasso',gbdt_model)

])
%%time

pipeline_gbt.fit(X_train,y_train)
print(np.sqrt(mean_squared_log_error(y_test,pipeline_gbt.predict(X_test))))
from sklearn.ensemble import VotingRegressor
xgb_final = xgb.XGBRegressor(objective="reg:squarederror",

                             n_estimators=200,

                             max_depth=5,

                             gamma=0.5,

                             colsample_bytree=0.3)
lgm_final = lgb.LGBMRegressor(reg_lambda=0.1,

                              n_estimators=100,

                              learning_rate=0.1,

                              colsample_bytree=0.2)
en_final = ElasticNet(l1_ratio=0.8,alpha=0.5)
lasso_final = Lasso(alpha=0.001)
gb_final = GradientBoostingRegressor(learning_rate=0.05, min_samples_leaf=5,

                                       min_samples_split=10, max_depth=4, n_estimators=3000)
final_regressor = VotingRegressor(estimators=[

    ('xgb',xgb_final),

    ('lgm',lgm_final),

    ('elastic',en_final),

    ('lasso',lasso_final),

    ('gbreg',gb_final)

])
pipeline_final = Pipeline([

    ('featureunion',combined_pipeline),

    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),

    ('vectorizer',DictVectorizer(sort=False)),

    ('regressor',final_regressor)

])
X = train_data.drop(['SalePrice'],axis=1)

y = train_data['SalePrice']
pipeline_final.fit(X,y)
final_preds = pipeline_final.predict(test_data)
submission = pd.DataFrame({ 'Id': ID,

                            'SalePrice': final_preds })

submission.to_csv(path_or_buf ="Advanced_Housing_Regression.csv", index=False)
