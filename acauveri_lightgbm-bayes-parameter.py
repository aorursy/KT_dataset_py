%%time
import pandas as pd

import  lightgbm as lgb
%%time

import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
print(df_train.head(3))
print(df_test.head(3))
# Define column date as datatype date and define new date features
for dataset in combine:
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['year'] = dataset.date.dt.year
    dataset['month'] = dataset.date.dt.month
    dataset['day'] = dataset.date.dt.day
    dataset['dayofyear'] = dataset.date.dt.dayofyear
    dataset['dayofweek'] = dataset.date.dt.dayofweek
    dataset['weekofyear'] = dataset.date.dt.weekofyear
    dataset['is_month_start'] = (dataset.date.dt.is_month_start).astype(int)
    dataset['is_month_end'] = (dataset.date.dt.is_month_end).astype(int)
    dataset['quarter'] =dataset.date.dt.quarter
dataset.drop('date', axis=1, inplace=True)
df_train.head()
df_train['daily_avg']=df_train.groupby(['item','store','dayofweek'])['sales'].transform('mean')
df_train['monthly_avg']=df_train.groupby(['item','store','month'])['sales'].transform('mean')
df_train['quarter_avg']=df_train.groupby(['item','store','quarter'])['sales'].transform('mean')
daily_avg=df_train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg=df_train.groupby(['item','store','month'])['sales'].mean().reset_index()
quarter_avg=df_train.groupby(['item','store','quarter'])['sales'].mean().reset_index()
monthly_avg
def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    
    x=x.rename(columns={'sales':col_name})
    return x

df_test=merge(df_test, daily_avg,['item','store','dayofweek'],'daily_avg')
df_test=merge(df_test, monthly_avg,['item','store','month'],'monthly_avg')
df_test=merge(df_test, quarter_avg,['item','store','quarter'],'quater_avg')
print(df_test.columns)
print(df_train.columns)

df_test=df_test.drop(['id'],axis=1)
df_train=df_train.drop(['date'],axis=1)
df_test.columns
df_train.shape
df_test.shape
df_train.head(2)
df_test.head(2)
df_train.isnull().sum()
df_test.isnull().sum()
df_train.dtypes
df_test.dtypes






y=pd.DataFrame()
y=df_train['sales']

df_train=df_train.drop(['sales'],axis=1)
x=df_train

from bayes_opt import BayesianOptimization
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.02, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'regression_l1','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']

opt_params = bayes_parameter_opt_lgb(x, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.02)
opt_params
param={'num_leaves': 33,
 'feature_fraction': 0.3327159326237221,
 'bagging_fraction': 0.8116430828587762,
 'max_depth': 6,
 'lambda_l1': 4.904560754684299,
 'lambda_l2': 2.4603987133536127,
 'min_split_gain': 0.03378397300297007,
 'min_child_weight': 5.241922465773013}
%%time
train_data = lgb.Dataset(x,y)
model =lgb.train(param,train_data,)
%%time
output=model.predict(df_test)
result=pd.DataFrame(output)
result

%%time
test=pd.read_csv('../input/test.csv',usecols=['id'])
fin=pd.DataFrame(test)
fin['sales']=result
fin.to_csv('Sales_bayesianoptimization.csv',index=False)
 