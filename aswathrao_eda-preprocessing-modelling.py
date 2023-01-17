# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/janatahack/train_8wry4cB.csv')
test = pd.read_csv('/kaggle/input/janatahack/test_Yix80N0.csv')
sample = pd.read_csv('/kaggle/input/janatahack/sample_submission_opxHi4g.csv')
print(train.shape)
print(train.info)
train.apply(lambda x:len(x.unique()))
train.isna().sum()
sns.countplot(train['gender'])
train['startTime'] = pd.to_datetime(train['startTime'])
train['endTime'] = pd.to_datetime(train['endTime'])
train
df = train.append(test)
df = df[['session_id','startTime','endTime','ProductList','gender']]
df['ProductCount'] = df.ProductList.str.count(';')+1
df['ProductList']
df['startTime'] = pd.to_datetime(df['startTime'])
df['endTime'] = pd.to_datetime(df['endTime'])
import numpy as np
from itertools import chain

# return list from series of comma-separated strings
def chainer(s):
    return list(chain.from_iterable(s.str.split(';')))

# calculate lengths of splits
lens = df['ProductList'].str.split(';').map(len)

# create new dataframe, repeating or chaining as appropriate
df1 = pd.DataFrame({'session_id': np.repeat(df['session_id'], lens),
                    'startTime': np.repeat(df['startTime'], lens),
                    'endTime':np.repeat(df['endTime'],lens),
                    'ProductCount': np.repeat(df['ProductCount'], lens),
                    'ProductList': chainer(df['ProductList']),
                    'gender':np.repeat(df['gender'],lens)})

print(df1)
df1.head()
df1['TimeTaken'] = abs(df1['endTime'] - df1['startTime']).astype('timedelta64[m]')
df1[['Date','Time']] = df1['startTime'].astype(str).str.split(" ",expand=True) 
df1['Date'] = pd.to_datetime(df1['Date'])
df1['Day'] = df1['Date'].apply(lambda x: x.weekday())
df1.head()
df1['TimeTaken'].max()
df1[['Category','SubCategory','SubSubCategory','SubSubSubCategory','Extra']] = df1['ProductList'].str.split("/",expand=True) 
del df1['Extra']
del df1['ProductList']

del df1['Time']
del df1['Date']
len(df1['session_id'].unique())
df1.columns
df1 = df1[['session_id','TimeTaken','Day','ProductCount','Category','SubCategory','SubSubCategory','SubSubSubCategory','gender']]
df1['TimeTaken'] = df1.TimeTaken.apply(lambda x:int(x))
df1.head()
from sklearn import preprocessing 
columns = ['Category','SubCategory','SubSubCategory','SubSubSubCategory']
label_encoder = preprocessing.LabelEncoder() 
  
for i in columns:
    df1[i]= label_encoder.fit_transform(df1[i]) 
  
df1['session_id'].describe()
test1 = df1[df1['gender'].isnull() == True]
train1 = df1[df1['gender'].isnull() == False]

train1.head()
test1.head()
from sklearn import preprocessing 
columns = ['gender']
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for i in columns:
    train1[i]= label_encoder.fit_transform(train1[i]) 
  
train1.corr()
sns.countplot(train1['Category'])
sns.countplot(train1['Day'])
train1.groupby('Day')['gender'].size()
sns.countplot(train1['ProductCount'])
pd.crosstab(train1['Day'],train1['gender'])
del test1['gender']
def extra_tree(Xtrain,Ytrain,Xtest):
    extra = ExtraTreesClassifier()
    extra.fit(Xtrain, Ytrain) 
    extra_prediction = extra.predict(Xtest)
    return extra_prediction
def Xg_boost(Xtrain,Ytrain,Xtest):
    xg = XGBClassifier(loss='exponential', learning_rate=0.05, n_estimators=1000, subsample=1.0, criterion='friedman_mse', 
                                  min_samples_split=2, 
                                  min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_depth=10, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, 
                                  init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None, warm_start=False, 
                                  presort='deprecated', 
                                  validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    xg.fit(Xtrain, Ytrain) 
    xg_prediction = xg.predict(Xtest)
    return xg_prediction
def LGBM(Xtrain,Ytrain,Xtest):
    lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=40,
                            max_depth=5, learning_rate=0.05, n_estimators=1000, subsample_for_bin=200, objective='binary', 
                            min_split_gain=0.0, min_child_weight=0.001, min_child_samples=10,
                            subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0,
                            reg_lambda=0.0, random_state=None, n_jobs=1, silent=True, importance_type='split')
    #lgbm = LGBMClassifier(n_estimators= 500)
    lgbm.fit(X_train, Y_train)
    lgbm_preds = lgbm.predict(X_test)
    return lgbm_preds
print(train1.columns)
print(test1.columns)
X_train = train1[['TimeTaken','Day','ProductCount','Category', 'SubCategory', 'SubSubCategory','SubSubSubCategory']]
Y_train = train1['gender']
X_test = test1[['TimeTaken','Day','ProductCount', 'Category', 'SubCategory', 'SubSubCategory','SubSubSubCategory']]
from autoviml.Auto_ViML import Auto_ViML
target = 'gender'
scoring_parameter = 'balanced-accuracy'

m, feats, trainm, testm = Auto_ViML(train1, target, test1,
                                    scoring_parameter=scoring_parameter,
                                    hyper_param='GS',feature_reduction=True,
                                     Boosting_Flag='Boosting_Flag',Binning_Flag=False)
sam = pd.read_csv('/kaggle/input/sample/gender_Binary_Classification_submission.csv')

test1['gender'] = sam['gender_predictions']
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('Auto.csv',index = False)

X_train.head()
cate_features_index = np.where(X_train.dtypes != float)[0]
xtrain,xtest,ytrain,ytest = train_test_split(X_train,Y_train,train_size=0.99,random_state=1236)
from catboost import Pool, CatBoostClassifier, cv, CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
model = CatBoostClassifier(iterations=7000, learning_rate=0.001, l2_leaf_reg=3.5, depth=5, 
                           rsm=0.99, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=50)
model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
predss = model.predict(X_test)

test1['gender'] = predss
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('Catboost+PC.csv',index = False)
#pred_xg = Xg_boost(X_train,Y_train,X_test)
#pred_et = extra_tree(X_train,Y_train,X_test)
pred_l = LGBM(X_train,Y_train,X_test)

# 0 - female, 1 male
test1['gender'] = pred_xg
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DXG.csv',index = False)
test1['gender'] = pred_et
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DETC.csv',index = False)

test1['gender'] = pred_l
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DPCLGBM.csv',index = False)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
ans = clf.predict(X_test)

print(len(pred_l))
test1['gender'] = ans
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DLR.csv',index = False)
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100).fit(X_train, Y_train)
prediction_of_ada = ada.predict(X_test)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', 
                                  min_samples_split=2, 
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=10, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, 
                                  init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, 
                                  presort='deprecated', 
                                  validation_fraction=0.1, n_iter_no_change=None, tol=0.0001).fit(X_train, Y_train)
prediction_of_gbc = gbc.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10).fit(X_train, Y_train)
prediction_of_rf = rf.predict(X_test)

test1['gender'] = prediction_of_ada
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DADA.csv',index = False)

test1['gender'] = prediction_of_gbc
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('Dgbc.csv',index = False)

test1['gender'] = prediction_of_rf
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DRF.csv',index = False)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train)

# Predicted class
nri = neigh.predict(X_test)


test1['gender'] = nri
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('Dknn.csv',index = False)
from sklearn.calibration import CalibratedClassifierCV

model = XGBClassifier()
metLearn=CalibratedClassifierCV(model, method='isotonic', cv=2)
metLearn.fit(X_train, Y_train)
testPredictions = metLearn.predict(X_test)
def submissions(predictions_by_model,string):
    test1['gender'] = predictions_by_model
    testn = test1[['session_id','gender']]
    print(testn.isna().sum())
    test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
    dic = {1:'male',0:'female'}
    test_final['gender'] = test_final['gender'].map(dic)
    test_final.to_csv(string.csv,index = False)

test1['gender'] = testPredictions
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DCCV.csv',index = False)
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt

# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import (roc_curve, auc, accuracy_score)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'metric_freq': 1,
    'is_training_metric': True,
    'max_bin': 255,
    'learning_rate': 0.1,
    'num_leaves': 63,
    'tree_learner': 'serial',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 5,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_save_binary_file': False,
    'output_model': 'LightGBM_model.txt',
    'num_machines': 1,
    'local_listen_port': 12400,
    'machine_list_file': 'mlist.txt',
    'verbose': 0,
    'subsample_for_bin': 200000,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0
}


lgb_train = lgb.Dataset(X_train, Y_train)

 
lgb_train
lgb_eval = lgb.Dataset(X_test)
# train
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval)



gridParams = {
    'learning_rate': [ 0.1],
    'num_leaves': [63],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary']
}

mdl = lgb.LGBMClassifier(
    task = params['task'],
    metric = params['metric'],
    metric_freq = params['metric_freq'],
    is_training_metric = params['is_training_metric'],
    max_bin = params['max_bin'],
    tree_learner = params['tree_learner'],
    feature_fraction = params['feature_fraction'],
    bagging_fraction = params['bagging_fraction'],
    bagging_freq = params['bagging_freq'],
    min_data_in_leaf = params['min_data_in_leaf'],
    min_sum_hessian_in_leaf = params['min_sum_hessian_in_leaf'],
    is_enable_sparse = params['is_enable_sparse'],
    use_two_round_loading = params['use_two_round_loading'],
    is_save_binary_file = params['is_save_binary_file'],
    n_jobs = -1
)

scoring = {'AUC': 'roc_auc'}

# Create the grid
#grid = GridSearchCV(mdl, gridParams, verbose=2, cv=5, scoring=scoring, n_jobs=-1, refit='AUC')
# Run the grid


#print('Best parameters found by grid search are:', grid.best_params_)
#print('Best score found by grid search is:', grid.best_score_)

yes = gbm.predict(X_test)
yess =[]
for i in yes:
    if i>=0.5:
        yess.append(1)
    else:
        yess.append(0)
test1['gender'] = yess
testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DDCCV.csv',index = False)
import h2o
h2o.init()
train2 = h2o.H2OFrame(train1)
test2 = h2o.H2OFrame(X_test)
train1.columns
y = 'gender'
x = train2.col_names
x.remove(y)
train2['gender'] = train2['gender'].asfactor()
train2['gender'].levels()
from h2o.automl import H2OAutoML
aml1 = H2OAutoML(max_models = 30, max_runtime_secs=200, seed = 1)
aml1.train(x = x, y = y, training_frame = train2)
preds = aml1.predict(test2)
print(sample.columns)
test1['gender'] = preds
#ans=h2o.as_list(preds) 
#sample['gender'] = ans['predict']
#sample.to_csv('Solution_H2O(Divided).csv',index=False)
#lb = aml.leaderboard
#lb.head()
#lb.head(rows=lb.nrows)
test1['gender'] = (h2o.as_list(preds['predict']))

testn = test1[['session_id','gender']]
print(testn.isna().sum())
test_final = testn.drop_duplicates(subset='session_id', keep='first', inplace=False)
dic = {1:'male',0:'female'}
test_final['gender'] = test_final['gender'].map(dic)
test_final.to_csv('DH2o.csv',index = False)