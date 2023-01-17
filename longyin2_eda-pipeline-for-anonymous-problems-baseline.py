import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# import catboost as cbt
import lightgbm as lgb
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test =  pd.read_csv('../input/test.csv')
train.head()
train.info()
null_num =  train.isnull().sum().sum()
print('There are {} null elements in train data'.format(null_num))
cols_with_onlyone_val = train.columns[train.nunique() == 1]
cols_with_onlyone_val
len(cols_with_onlyone_val)
cols = [x for x in train.columns if x not in cols_with_onlyone_val]
train_clean = train[cols].copy()
train_clean_info = train_clean.describe()
train_clean_info
columns = train_clean_info.columns
cols_del = []  # del those duplicated columns 
dup_dict = {} 

for i in range(len(columns)-1):
    if columns[i] in cols_del:
        continue 
    if i % 10 ==0:
        print(i / len(columns))
    first = train_clean_info[columns[i]].values  
    res = train_clean_info.iloc[:,i+1:] - np.tile([first],[train_clean_info.shape[1]-i -1,1]).T  
    cols_del.extend(res.columns[np.sum(res) == 0]) 
    if np.sum(np.sum(res) == 0) > 0: 
        dup_dict[columns[i]] = res.columns[np.sum(res) == 0] 
cols_del
dup_dict
train_clean[['168b3e5bc','f8d75792f','34ceb0081','d60ddde1b','70f3a87ec','66f57f2e5']]
cols = [x for x in train_clean.columns if x not in cols_del]
train_clean = train_clean[cols].copy()
train.head()
test.info()
cols = [x for x in train_clean.columns if x!='target']
test_clean = test[cols].copy()
cols_with_onlyone_val = test_clean.columns[test_clean.nunique() == 1]
cols_with_onlyone_val
test_clean_info = test_clean.describe()
test_clean_info
columns = test_clean_info.columns
cols_del = []  # del those duplicated columns 
dup_dict = {} 

for i in range(len(columns)-1):
    if columns[i] in cols_del:
        continue 
    if i % 10 ==0:
        print(i / len(columns))
    first = test_clean_info[columns[i]].values  
    res = test_clean_info.iloc[:,i+1:] - np.tile([first],[test_clean_info.shape[1]-i -1,1]).T  
    cols_del.extend(res.columns[np.sum(res) == 0]) 
    if np.sum(np.sum(res) == 0) > 0: 
        dup_dict[columns[i]] = res.columns[np.sum(res) == 0] 
cols_del
train['target']
plt.scatter(x = range(train.shape[0]), y = train['target'].values)
train['target'].plot()
plt.figure(figsize=(12,8))
sns.distplot(train_clean["target"].values, bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Target Histogram", fontsize=14) 
train['target'].describe()
(train['target'].describe().loc['max'] - train['target'].describe().loc['mean']) / train['target'].describe().loc['std']
train_clean["target"] = train_clean["target"].apply(np.log1p)
from sklearn import model_selection
from sklearn.model_selection import train_test_split
def run_lgb_val(train_X, train_y, val_X, val_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 64,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.85,
        "feature_fraction" : 0.85,
        "bagging_frequency" : 5,
        "bagging_seed" : 100,
        "verbosity" : -1,
        "seed": 921212
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)  
    
feature_columns = [x for x in train_clean.columns if x not in ['ID','target']]
for rnd in range(3):
    print('*' * 50)
    print(rnd)
    print('*' * 50)
    train_X, val_X, train_y, val_y = train_test_split(train_clean[feature_columns], train_clean['target'], test_size = 0.3, random_state = rnd)
    run_lgb_val(train_X, train_y, val_X, val_y) 
def run_lgb_test(train_X, train_y, val_X, val_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 64,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.85,
        "feature_fraction" : 0.85,
        "bagging_frequency" : 5,
        "bagging_seed" : 100,
        "verbosity" : -1,
        "seed": 921212
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)  
    return model
feature_columns = [x for x in train_clean.columns if x not in ['ID','target']]
res = []
for rnd in range(3):
    print('*' * 50)
    print(rnd)
    print('*' * 50)
    train_X, val_X, train_y, val_y = train_test_split(train_clean[feature_columns], train_clean['target'], test_size = 0.3, random_state = rnd)
    model = run_lgb_test(train_X, train_y, val_X, val_y) 
    pred = model.predict(test_clean[feature_columns])
    res.append(pred) 
test['target'] = np.expm1(np.mean(res,axis=0))
test[['ID','target']].head(100)
test[['ID','target']].to_csv('Baseline.csv',index = False)
