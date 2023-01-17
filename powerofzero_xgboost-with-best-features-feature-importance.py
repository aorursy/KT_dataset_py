#Importing libraries
import pandas as pd#data manipulation
pd.set_option('display.max_columns', None)
import numpy as np # mathematical operations
import scipy as sci # math ops
import seaborn as sns # visualizations
import matplotlib.pyplot as plt # for plottings
%matplotlib inline 
from sklearn.utils import shuffle
%%time
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
train = shuffle(train)
train.shape
test.shape
sample_submission.shape
train.head(3)
#Types of columns
df_type = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(columns={0:'type_of_column'})
df_type.head()
df_type['numerical'] = df_type['type_of_column'].apply(lambda x: 1 if x in ['int8','int16','int32','int64','float16','float32','float64'] else 0)
numeric_columns = list(df_type[(df_type['numerical'] ==1) & (df_type['index'] !='target')]['index'])
categorical_columns = list(df_type[(df_type['type_of_column'] =='object') & (df_type['index'] !='target')]['index'])
categorical_columns
print("Number of Numeric Columns = ", len(numeric_columns))
print("Number of Categorical Columns = ", len(categorical_columns))
import missingno as msno
%%time
number_of_nans_in_train = []
for i in list(train.columns.values):
    number_of_nans_in_train.append(train[i].isnull().sum())
sorted(number_of_nans_in_train,reverse=True)[:5]
%%time
number_of_nans_in_test = []
for i in list(test.columns.values):
    number_of_nans_in_test.append(test[i].isnull().sum())
sorted(number_of_nans_in_test,reverse=True)[:5]
train['target'].describe()
#Log transformation
train['target2'] = np.log(train['target'])
train['target'].plot(kind='hist',bins=50)
train['target2'].plot(kind='hist',bins=50)
train_id = train['ID']
del train['ID']
del train['target']
import xgboost as xgb
from xgboost import XGBRegressor
model_xgb = XGBRegressor()
train.shape
important_features = []
%%time
for i in range(0,5000,500):
    print(i)
    if i != 4500:
        X = train.iloc[:,i:i+500]
        y = train['target2']
        X_train = X[:int(X.shape[0]*0.8)]
        y_train = y[:int(y.shape[0]*0.8)]
        X_cv = X[int(X.shape[0]*0.8):]
        y_cv = y[int(y.shape[0]*0.8):]
        model_xgb.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_cv,y_cv)],eval_metric='rmse',verbose=False)
        [important_features.append(x) for x in list(pd.concat([pd.Series(model_xgb.feature_importances_),pd.Series(list(X_train.columns.values))],axis=1).rename(columns={0:'importance',1:'column'}).sort_values(by='importance',ascending=False)[:25]['column'])]
    else:
        y = train['target2']
        X = train[list(train.columns.values)[i:train.shape[1]-1]]
        y = train['target2']
        X_train = X[:int(X.shape[0]*0.8)]
        y_train = y[:int(y.shape[0]*0.8)]
        X_cv = X[int(X.shape[0]*0.8):]
        y_cv = y[int(y.shape[0]*0.8):]
        model_xgb.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_cv,y_cv)],eval_metric='rmse',verbose=False)
        [important_features.append(x) for x in list(pd.concat([pd.Series(model_xgb.feature_importances_),pd.Series(list(X_train.columns.values))],axis=1).rename(columns={0:'importance',1:'column'}).sort_values(by='importance',ascending=False)[:25]['column'])]
train = train[important_features + ['target2']]
test = test[important_features]
train.shape
test.shape
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state= 33)
md,lr,ne = [3,6,9,12],[0.01,0.05,0.10,0.15,0.2],[100,150,200,250,300]
params = [[x,y,z] for x in md for y in lr for z in ne]
print(len(params))
def rmsle(a,b):
    return np.sqrt(np.mean(np.square( np.log( (np.exp(a)) + 1 ) - np.log((np.exp(b))+1) )))
params_dict = {}
X = train[[a for a in list(train.columns.values) if a != 'target2']]
y = train['target2']
X = X.reset_index(drop=True)
X = X.values
y = y.reset_index(drop=True)
y = y.values
"""%%time
for i in range(len(params)):
    error_rate = []
    for train_index, test_index in kf.split(X):
        X_train, X_cv= X[train_index], X[test_index]
        y_train, y_cv= y[train_index], y[test_index]
        dtrain=xgb.DMatrix(X_train,label=y_train)
        dcv=xgb.DMatrix(X_cv,label=y_cv)
        dtest =xgb.DMatrix(X_cv)
        watchlist = [(dtrain,'train-rmse'), (dcv, 'eval-rmse')]
        parameters={'max_depth':params[i][0], 'silent':1,'objective':'reg:linear','eval_metric':'rmse','learning_rate':params[i][1]}
        num_round=params[i][2]
        xg=xgb.train(parameters,dtrain,num_boost_round = num_round,evals = watchlist,early_stopping_rounds = 7,verbose_eval =False) 
        y_pred=xg.predict(dtest) 
        rmsle_calculated = rmsle(y_pred,y_cv)
        error_rate.append(rmsle_calculated) 
    params_dict[str(params[i])] = round(np.mean(error_rate),5)
    if i % 5 ==0:
        print(i)"""
"""params_df = pd.Series(params_dict)
print(len(params_dict))
params_df = params_df.sort_values(ascending=True)
params_df[:20]"""
#MAX_DEPTH= 9 , LEARNING_RATE = 0.05, NUMBER_OF_ROUND = 250 OR 300
%%time
error_rate = []
for train_index, test_index in kf.split(X):
    X_train, X_cv= X[train_index], X[test_index]
    y_train, y_cv= y[train_index], y[test_index]
    dtrain=xgb.DMatrix(X_train,label=y_train)
    dcv=xgb.DMatrix(X_cv,label=y_cv)
    dtest =xgb.DMatrix(X_cv)
    watchlist = [(dtrain,'train-rmse'), (dcv, 'eval-rmse')]
    parameters={'max_depth':9, 'silent':1,'objective':'reg:linear','eval_metric':'rmse','learning_rate':0.04}
    num_round=250
    xg=xgb.train(parameters,dtrain,num_boost_round = num_round,evals = watchlist,early_stopping_rounds = 5,verbose_eval =False) 
    y_pred=xg.predict(dtest) 
    rmsle_calculated = rmsle(y_pred,y_cv)
    error_rate.append(rmsle_calculated) 
xgb.plot_importance(xg,max_num_features=10)
for i in [0,1,3,2,125,4,200,25,27,175]:
    print(list(train.columns.values)[i])
test_matrix = test.values
dtest =xgb.DMatrix(test_matrix)
y_pred=xg.predict(dtest)
y_pred[:5]
y_pred = np.exp(y_pred)
y_pred = pd.Series(y_pred)
sample_submission.head()
del sample_submission['target']
sample_submission['target'] = y_pred
sample_submission.head()
sample_submission.shape
sample_submission.to_csv('my_second.csv',index=False)