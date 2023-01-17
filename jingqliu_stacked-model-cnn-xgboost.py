import pandas as pd
import numpy as np
import re
import gc
from keras.preprocessing import text, sequence
import math
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
train = pd.read_csv('../input/readyforuse/trainforuse_mix.csv')
test = pd.read_csv('../input/readyforuse/testforuse_mix.csv')
submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')

train = train.drop(columns = ['item_id'])
train['date'] = pd.to_datetime(train['activation_date']).dt.weekday.astype('int')
train = train.drop(columns = ['activation_date','image'])

test = test.drop(columns = ['item_id'])
test['date'] = pd.to_datetime(test['activation_date']).dt.weekday.astype('int')
test = test.drop(columns = ['activation_date','image'])
encoding_predictors = ['user_id', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'item_seq_number', 'user_type', 'date']
for i in encoding_predictors:
    print(str(i) +': ' + str(len(set(train[i]))))
    print('Na contains? ' + str(train[i].isnull().values.any()))
for i in encoding_predictors:
    print(str(i) +': ' + str(len(set(test[i]))))
    print('Na contains? ' + str(test[i].isnull().values.any()))
l_region = dict(zip(list(set(train['region'])),range(1,29)))
l_parent_category_name = dict(zip(list(set(train['parent_category_name'])),range(1,10)))
l_category_name = dict(zip(list(set(train['category_name'])),range(1,48)))
l_user_type = dict(zip(list(set(train['user_type'])),range(1,4)))
l_date = dict(zip(list(set(train['date'])),range(1,8)))
l_param1 = dict(zip(list(set(train['param_1'])),range(1,373)))
l_param2 = dict(zip(list(set(train['param_2'])),range(1,273)))
l_city = dict(zip(list(set(train['city'])),range(1,1734)))
l_param3 = dict(zip(list(set(train['param_3'])),range(1,1221)))
train['region'] = train['region'].replace(l_region)
train['parent_category_name'] = train['parent_category_name'].replace(l_parent_category_name)
train['category_name'] = train['category_name'].replace(l_category_name)
train['user_type'] = train['user_type'].replace(l_user_type)
train['date'] = train['date'].replace(l_date)
train['param_1'] = train['param_1'].replace(l_param1)
train['param_2'] = train['param_2'].replace(l_param2)
train['city'] = train['city'].replace(l_city)
train['param_3'] = train['param_3'].replace(l_param3)
test['region'] = test['region'].map(l_region)
test['parent_category_name'] = test['parent_category_name'].map(l_parent_category_name)
test['category_name'] = test['category_name'].map(l_category_name)
test['user_type'] = test['user_type'].map(l_user_type)
test['date'] = test['date'].map(l_date)
test['param_1'] = test['param_1'].map(l_param1)
test['param_2'] = test['param_2'].map(l_param2)
test['city'] = test['city'].map(l_city)
test['param_3'] = test['param_3'].map(l_param3)
def modelfit(alg,dtrain,predictors,useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain['deal_probability'].values, feature_names = predictors)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold = cv_folds, metrics = 'rmse', early_stopping_rounds = early_stopping_rounds)
        alg.set_params(n_estimators = cvresult.shape[0])
        print('Best n_estimator = ' + str(cvresult.shape[0]))
    alg.fit(dtrain[predictors], dtrain['deal_probability'], eval_metric = 'rmse')
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    print('\nModel Report:')
    print('RMSE: %f' % math.sqrt(metrics.mean_squared_error(dtrain['deal_probability'].values, dtrain_predictions)))
train.loc[:,'image_item_n'] = train.groupby(['image_top_1','item_seq_number']).user_id.transform('nunique')
train.loc[:,'image_city_n'] = train.groupby(['city','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_region_n'] = train.groupby(['region','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_categoryname_n'] = train.groupby(['category_name','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_param2_n'] = train.groupby(['image_top_1','param_2']).user_id.transform('nunique')
train.loc[:,'image_parentcategoryname_n'] = train.groupby(['parent_category_name','image_top_1']).user_id.transform('nunique')
train.loc[:,'image_date_n'] = train.groupby(['image_top_1','date']).user_id.transform('nunique')
train.loc[:,'image_usertype_n'] = train.groupby(['image_top_1','user_type']).user_id.transform('nunique')
train.loc[:,'image_param1_n'] = train.groupby(['image_top_1','param_1']).user_id.transform('nunique')
train.loc[:,'image_param3_n'] = train.groupby(['image_top_1','param_3']).user_id.transform('nunique')
test.loc[:,'image_item_n'] = test.groupby(['image_top_1','item_seq_number']).user_id.transform('nunique')
test.loc[:,'image_city_n'] = test.groupby(['city','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_region_n'] = test.groupby(['region','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_categoryname_n'] = test.groupby(['category_name','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_param2_n'] = test.groupby(['image_top_1','param_2']).user_id.transform('nunique')
test.loc[:,'image_parentcategoryname_n'] = test.groupby(['parent_category_name','image_top_1']).user_id.transform('nunique')
test.loc[:,'image_date_n'] = test.groupby(['image_top_1','date']).user_id.transform('nunique')
test.loc[:,'image_usertype_n'] = test.groupby(['image_top_1','user_type']).user_id.transform('nunique')
test.loc[:,'image_param1_n'] = test.groupby(['image_top_1','param_1']).user_id.transform('nunique')
test.loc[:,'image_param3_n'] = test.groupby(['image_top_1','param_3']).user_id.transform('nunique')
def clean(string):
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub('[\W]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string.lower())
    return string

def find_punc(string):
    string = re.sub(r'\s','',string)
    string = re.findall('[\W]',string)
    l = len(string)
    return l
train_t = train['title'].apply(clean)
test_t = test['title'].apply(clean)
train_d = train['description'].astype(str).apply(clean)
test_d = test['description'].astype(str).apply(clean)
train_t_len = []
for line in train_t:
    train_t_len.append(len(line.split()))
    
train_d_len = []
for line in train_d:
    train_d_len.append(len(line.split()))
    
test_t_len = []
for line in test_t:
    test_t_len.append(len(line.split()))
    
test_d_len = []
for line in test_d:
    test_d_len.append(len(line.split()))
#train['t_n'] = train_t_len
train['t_per'] = np.array(train['title'].apply(find_punc))/np.array(train_t_len)
#test['t_n'] = test_t_len
test['t_per'] = np.array(test['title'].apply(find_punc))/np.array(test_t_len)
train['d_n'] = train_d_len
#train['d_per'] = np.array(train['description'].astype(str).apply(find_punc))/np.array(train_d_len)
test['d_n'] = test_d_len
#test['d_per'] = np.array(test['description'].astype(str).apply(find_punc))/np.array(test_d_len)
train = train.drop(columns = ['title','description'])
test = test.drop(columns = ['title','description'])
ready_train = train[['price','image_top_1','param_1','item_seq_number','city','region','parent_category_name','category_name','user_type','date','param_2','deal_probability','param_3','mix'] + list(train.columns[15:])]
ready_test = test[['price','image_top_1','param_1','item_seq_number','city','region','parent_category_name','category_name','user_type','date','param_2','param_3','mix'] + list(test.columns[14:])]
predictors = ready_train.columns[ready_train.columns != 'deal_probability']
len(predictors)
xgb1 = XGBRegressor(objective = 'reg:logistic', learning_rate = 0.1, n_estimators = 1000, max_depth = 5, min_child_weight = 6, gamma = 0, subsample = 0.9, colsample_bytree = 0.7, reg_alpha = 1.4, seed = 2018)
modelfit(xgb1, ready_train, predictors, useTrainCV = False)
xgb.plot_importance(xgb1)
plt.show()
pred = xgb1.predict(ready_test[predictors])
submission['deal_probability'] = pred
submission.to_csv('submission.csv',index=False)