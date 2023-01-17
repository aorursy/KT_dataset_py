# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath="/kaggle/input/santander-value-prediction-challenge/"
%matplotlib inline

import gc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')
traindf = pd.read_csv(filepath+"train.csv")
testdf = pd.read_csv(filepath+"test.csv")
print(traindf.head())
print("-------------------------")
print(testdf.head())
traindf.info()
testdf.info()
traindf.columns[traindf.isnull().sum() != 0].size
testdf.columns[testdf.isnull().sum() != 0].size
colstoremove=[]
for col in traindf.columns:
    if col != 'ID' and col != 'target' :
        if traindf[col].std() == 0:
            colstoremove.append(col)

traindf.drop(colstoremove, axis=1, inplace=True)
testdf.drop(colstoremove, axis=1, inplace=True)

print("Total constant columns removed : ", len(colstoremove))
%%time

def duplicate_columns(df):
    groups = df.columns.to_series().groupby(df.dtypes).groups
    dups=[]
    
    i=1
    for t,v in groups.items():
        print("i=",i, "----->")
        cs = df[v].columns
        vs = df[v]
        lcs = len(cs)
        #print(vs)
        i += 1
        print("lcs=",lcs)    
        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups

dupcols = duplicate_columns(traindf)
print(dupcols)
traindf.drop(dupcols, axis=1, inplace=True)
testdf.drop(dupcols, axis=1, inplace=True)
print("Removed duplicated columns: ",dupcols)
def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID', 'target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test
%%time
traindf , testdf = drop_sparse(traindf, testdf)
gc.collect()

print(traindf.shape)
print(testdf.shape)
xtrain = traindf.drop(['ID', 'target'] ,  axis=1)
ytrain = np.log1p(traindf['target'].values)

xtest = testdf.drop(['ID'], axis=1)

print(xtrain.shape, ytrain.shape)
print(xtest.shape)
##  split train data into train and validation

xtrain, xval, ytrain, yval = model_selection.train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
print(xtrain.shape, ytrain.shape)
print(xval.shape, yval.shape)
def run_lgb(xtrain, ytrain, xval, yval, xtest):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.004,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed" : 42
    }
    
    lgtrain = lgb.Dataset(xtrain, label=ytrain)
    lgval = lgb.Dataset(xval, label=yval)
    evals_result={}
    model = lgb.train(params, lgtrain, 5000, valid_sets = [lgtrain, lgval], early_stopping_rounds=100, 
                     verbose_eval=150, evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(xtest, num_iteration=model.best_iteration))
    
    return pred_test_y, model, evals_result
pred_test_y , model, evals_result = run_lgb(xtrain, ytrain, xval, yval, xtest)

print("LightGBM model training completed..")
## feature importance

print("Feature Importance : ")
gain = model.feature_importance('gain')
print("gain : ", gain)
featureimp = pd.DataFrame({'feature': model.feature_name(), 'split':model.feature_importance('split'), 
                          'gain': 100*gain/gain.sum()}).sort_values(by='gain', ascending=False)
print(featureimp[:50])
def run_xgb(xtrain, ytrain, xval, yval, xtest):
    params={
        "objective" : "reg:linear",
        "eval_metric" : "rmse" ,
        "eta" : 0.001,
        "max_depth" : 10,
        "subsample" : 0.6,
        "colsample_bytree" : 0.6,
        "alpha" : 0.001,
        "random_state" : 42,
        "silent" : True
    }
    
    trdata = xgb.DMatrix(xtrain, ytrain)
    valdata = xgb.DMatrix(xval, yval)
    
    watchlist = [(trdata, 'train'), (valdata, 'valid')]
    
    model_xgb = xgb.train(params, trdata, 2000, watchlist, maximize=False, early_stopping_rounds=100, 
                         verbose_eval=100)
    
    dtest = xgb.DMatrix(xtest)
    
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb
xgb_pred_y, model_xgb = run_xgb(xtrain, ytrain, xval, yval, xtest)
cb_model = CatBoostRegressor(iterations = 500,
                            learning_rate=0.05,
                            depth=10,
                            eval_metric='RMSE',
                            random_seed=42,
                            bagging_temperature=0.2,
                            od_type='Iter',
                            metric_period=50,
                            od_wait=20
                            )
cb_model.fit(xtrain, ytrain, eval_set=(xval, yval), use_best_model=True, verbose=50)
pred_test_cat = np.expm1(cb_model.predict(xtest))
sub_lgb=pd.DataFrame()
sub_lgb['target']=pred_test_y

sub_xgb=pd.DataFrame()
sub_xgb['target']=xgb_pred_y

sub_cat = pd.DataFrame()
sub_cat['target'] = pred_test_cat

sub = pd.read_csv(filepath+'sample_submission.csv')
sub.head()
sub['target']=sub_lgb['target']*0.5 + sub_xgb['target']*0.3 + sub_cat['target']*0.2
sub.head()
sub.to_csv('/submission.csv', index=False)