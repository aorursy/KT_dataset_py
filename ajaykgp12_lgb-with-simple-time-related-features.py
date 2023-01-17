import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
random_state = 42
import os
print(os.listdir("../input"))


train = pd.read_csv('../input/ga-encoded/train_new.csv', dtype={'fullVisitorId': 'str'},)
test = pd.read_csv('../input/ga-encoded/test_new.csv', dtype={'fullVisitorId': 'str'},)
print('Training data shape {},  Test Data Shape {}'.format(train.shape, test.shape))


all_columns = train.columns.tolist()

#Remove Columns which have only one unique values
cols_to_delete = []
for col in all_columns:
    dist_vals = train[col].value_counts().shape[0]
    if dist_vals == 1:
       cols_to_delete.append(col)
   

# Remove columns with more than 90% Missing Values
cols_to_delete = cols_to_delete + ['trafficSource.adContent', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.adNetworkType', 
                                   'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.adwordsClickInfo.gclId']


#Remove columns which are Id's ot traget values
cols_to_delete =  cols_to_delete + ['fullVisitorId', 'sessionId', 'visitId', 'totals.transactionRevenue', 'trafficSource.campaignCode']

num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',
             'totals.newVisits', 'totals.visits']  

cat_cols = [x for x in all_columns if x not in num_cols and x not in cols_to_delete]
features = cat_cols + num_cols

print('Number of features:', len(features))
print('Number of cat cols:', len(cat_cols))
print('Number of num cols:', len(num_cols))
for col in num_cols:
    train[col] = pd.to_numeric(train[col])    
    test[col] = pd.to_numeric(test[col])
    
train['totals.transactionRevenue'].fillna(0, inplace = True)
train['totals.transactionRevenue'] = pd.to_numeric(train['totals.transactionRevenue'])
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'])    

train['visitStartTime'] = pd.to_datetime(train['visitStartTime'], unit ='s')
test['visitStartTime']  = pd.to_datetime(test['visitStartTime'], unit ='s')

train['date'] = pd.to_datetime(train['date'], format = '%Y%m%d')
test['date'] = pd.to_datetime(test['date'], format = '%Y%m%d')

train['year'] = train['date'].dt.year
test['year'] =  test['date'].dt.year

train['month'] = train['date'].dt.month
test['month'] =   test['date'].dt.month

train['day'] =  train['date'].dt.day
test['day'] =   test['date'].dt.day  

train['hour'] =  train['visitStartTime'].dt.hour
test['hour'] =   test['visitStartTime'].dt.hour

cat_cols = cat_cols + ['year', 'month', 'day', 'hour']
# num_cols = num_cols + ['year', 'month', 'day', 'hour']
cat_cols.remove('date')
num_cols.remove('visitStartTime')
features = cat_cols + num_cols

print('Number of features:', len(features))
y_train =    train['totals.transactionRevenue']
X_train  =   train[features]
X_test =   test[features]
print('Training data shape {} Test Data Shape {}'.format(X_train.shape, X_test.shape))

params = {}
params['learning_rate'] = 0.08
params['boosting_type'] = 'gbdt'
params['objective'] =  'regression'  
params['metric'] = 'rmse'
params['seed'] = random_state
params['num_threads'] = 4
params['lambda_l1'] = 0.1
params['min_gain_to_split'] = 1

folds = KFold(n_splits = 5, shuffle = True, random_state = random_state)

oof_pred = np.zeros(shape=(X_train.shape[0])) 
test_pred = np.zeros(shape=(X_test.shape[0]))  
cv_score = np.zeros(shape =  folds.n_splits)
feature_imp = pd.DataFrame()


for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
    train_x, train_y = X_train.iloc[train_idx],  y_train.iloc[train_idx]
    valid_x, valid_y = X_train.iloc[valid_idx],  y_train.iloc[valid_idx]
    
    dtrain = lgb.Dataset(train_x, label= train_y)
    dvalid = lgb.Dataset(valid_x, label= valid_y)    

    model = lgb.train( params,
                 dtrain,
                 num_boost_round= 10000,
                 valid_sets= [ dtrain, dvalid],
                 early_stopping_rounds=200,        
                                 
                 verbose_eval = 50
                 )    
    
    oof_pred[valid_idx] = model.predict(valid_x)    
    test_pred += model.predict(X_test) / folds.n_splits    
    cv_score[n_fold] = round(np.sqrt(mean_squared_error(valid_y,  oof_pred[valid_idx])), 5)   
    print('\nFold %2d RMSE: %.6f' %(n_fold + 1, cv_score[n_fold] ))
    
    
    fold_importance = pd.DataFrame()
    fold_importance["feature"] =  model.feature_name()
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = n_fold + 1    
    feature_imp = pd.concat([feature_imp, fold_importance], axis=0)

print('CV OOF RMSE:{:.5f}, Mean CV RMSE: {:.5f}, Fold CV RMSE:{}'.format(np.sqrt(mean_squared_error(y_train, oof_pred)), 
                                                                      np.mean(cv_score),
                                                                      cv_score                                                                     
                                                                     ))


test_pred[test_pred<0] = 0
sub = pd.DataFrame()
sub['fullVisitorId'] = test['fullVisitorId'].copy()
# sub['PredictedLogRevenue'] = test_pred
sub['PredictedLogRevenue'] = np.expm1(test_pred)
sub_grp = sub[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
sub_grp['PredictedLogRevenue'] = np.log1p(sub_grp['PredictedLogRevenue']) 
sub_grp.to_csv('lgb_sub.csv',index=False)

plt.figure(figsize=(8, 10))
feature_imp_avg = feature_imp[['feature', 'importance']].groupby(['feature']).mean().reset_index()
feature_imp_avg.sort_values(['importance'], ascending= False, inplace = True)
sns.barplot(x= 'importance', y = 'feature', data =feature_imp_avg)
plt.show()

