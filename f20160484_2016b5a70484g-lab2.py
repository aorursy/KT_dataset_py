import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb

# from sklearn import cross_validation, metrics

from sklearn.model_selection import GridSearchCV 

%matplotlib inline
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df.head()
df1 = df

df1 = df1.set_index('id')
df1 = df1.drop( df1.index[ (df1.chem_3 < 11) & (df1['class'] == 7) ] )

df1 = df1.drop( df1.index[ (df1.chem_3 > 14) & (df1['class'] == 6) ] )

df1 = df1.drop( df1.index[ (df1.chem_5 > 3) & (df1['class'] == 3) ] )

df1 = df1.drop( df1.index[ (df1.chem_5 > 2) & (df1['class'] == 5) ] )

# df1 = df1.drop( df1.index[ (df1.chem_1 > 1) & (df1['class'] == 7) ] )

df1 = df1.drop( df1.index[ (df1.chem_6 > 20) ] )

df1 = df1.drop(df1.index[df1.attribute > 3])

# df1 = df1.drop(df1.index[df1.chem_2 > 50])
def test_ETC(df, test):

#     df = df.drop('id', axis = 1)

#     test = test.drop('id', axis = 1)

    

    X = df.drop('class', axis = 1)

    y = df['class']

    

    scaler = RobustScaler()

    scaler.fit(X)

    X = scaler.transform(X)

    test = scaler.transform(test)

    

    clas = ExtraTreesClassifier(n_estimators = 5000, random_state = 4)

    cl = clas.fit(X, y)

    y_pred = cl.predict(test)

    

    return y_pred
def test_RFC(df, test):

#     df = df.drop('id', axis = 1)

#     test = test.drop('id', axis = 1)

    

    X = df.drop('class', axis = 1)

    y = df['class']

    

    scaler = RobustScaler()

    scaler.fit(X)

    X = scaler.transform(X)

    test = scaler.transform(test)

    

    clas = RandomForestClassifier(n_estimators = 4000, random_state = 4)

    cl = clas.fit(X, y)

    y_pred = cl.predict(test)

    

    return y_pred
def test_xgboost(df, test):

    

    X = df.drop('class', axis = 1)

    y = df['class']

    

    scaler = RobustScaler()

    scaler.fit(X)

    X = scaler.transform(X)

    test = scaler.transform(test)

    

    param = {}

    param['booster'] = 'gbtree'

    param['objective'] = 'multi:softmax'

    param["eval_metric"] = "merror"

    param['eta'] = 0.3

    param['gamma'] = 0

#     param['max_depth'] = 6

#     param['min_child_weight']=1

#     param['max_delta_step'] = 0

#     param['subsample']= 1

#     param['colsample_bytree']=1

#     param['silent'] = 1

#     param['seed'] = 0

#     param['base_score'] = 0.5



#     clf = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,

#        gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,

#        min_child_weight=1, missing=None, n_estimators=100, nthread=-1,

#        objective='multi:softmax', reg_alpha=0, reg_lambda=1,

#        scale_pos_weight=1, seed=0, silent=True, subsample=1)

    clf = XGBClassifier(booster = 'gbtree', objective = 'multi:softmax', eval_metric = 'merror',

                       eta = 0.3, max_depth = 3, num_round = 500)

    xgb_clf = clf.fit(X,y)

    y_pred_xgb = xgb_clf.predict(test)

#     xgb_acc = accuracy_score(y_val,y_pred_xgb)

    

    return y_pred_xgb

    
def test_gbm(df, test):

    

    X = df.drop('class', axis = 1)

    y = df['class']

    

    scaler = RobustScaler()

    scaler.fit(X)

    X = scaler.transform(X)

    test = scaler.transform(test)

    

    train_data=lgb.Dataset(X, label = y)

    param = {'num_class': 6 , 'objective':'multiclass', 'max_depth':4, 'learning_rate':.05}

    param['metric'] = ['multi_error']

    

    num_round = 100

    lgbm = lgb.train(param, train_data, num_round)

    

    y_pred = lgbm.predict(test)

    
test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

Xtest = test

Xtest = test.set_index('id')

Xtest.head()

test.describe()
#dropping columns if needed on df1 and Xtest

df1 = df1.drop(['chem_2', 'chem_3', 'chem_5', 'chem_7'], axis = 1)

Xtest = Xtest.drop(['chem_2', 'chem_3', 'chem_5', 'chem_7'], axis = 1)
test_ans = pd.DataFrame(test_ETC(df1, Xtest))
test_ans = pd.DataFrame(test_RFC(df1, Xtest))
test_ans = pd.DataFrame(test_xgboost(df1, Xtest))
test_ans.head()
test_ans = pd.concat([test['id'], test_ans], axis = 1)
test_ans.describe()
test_ans.columns = ['id', 'class']

test_ans.head()
test_ans.to_csv('submission10.csv', index = False)
len(test_ans)
Xtest