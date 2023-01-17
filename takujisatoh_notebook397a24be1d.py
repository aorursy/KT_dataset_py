# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import scipy as sp
import pandas as pd
import time
import gc
import re
from pandas import DataFrame, Series
import datetime as datetime

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,ShuffleSplit,KFold,GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder
from tqdm.notebook import tqdm
from sklearn.preprocessing import quantile_transform
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier,LGBMRegressor
from hyperopt import fmin, tpe, hp, rand, Trials
from sklearn.ensemble import RandomForestRegressor

import random
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 乱数シード固定
seed_everything(2020)
#df_train = pd.read_csv('../input/exam-for-students20200923/train.csv', index_col=0, skiprows=lambda x: x%20!=0)
df_train = pd.read_csv('../input/exam-for-students20200923/train.csv', index_col=0)
df_test = pd.read_csv('../input/exam-for-students20200923/test.csv', index_col=0)
df_train['ConvertedSalary']
display(df_train.shape)
display(df_test.shape)
# engeenering
df_train['istrain'] = 1
df_test['istrain'] = 0
df_train_test = pd.concat([df_train, df_test], axis=0)
df_train_test=df_train_test.replace({'YearsCodingProf':{'0-2 years':11,
                                                        '3-5 years':10,
                                                        '6-8 years':9,
                                                        '9-11 years':8,
                                                        '12-14 years':7,
                                                        '15-17 years':6,
                                                        '18-20 years':5,
                                                        '21-23 years':4,
                                                        '24-26 years':3,
                                                        '27-29 years':2,
                                                        '30 or more years':1}})

df_train_test['YearsCodingProf'] = df_train_test['YearsCodingProf'].astype('object')


df_train_test=df_train_test.replace({'YearsCoding':{'0-2 years':11,
                                                        '3-5 years':10,
                                                        '6-8 years':9,
                                                        '9-11 years':8,
                                                        '12-14 years':7,
                                                        '15-17 years':6,
                                                        '18-20 years':5,
                                                        '21-23 years':4,
                                                        '24-26 years':3,
                                                        '27-29 years':2,
                                                        '30 or more years':1}})

df_train_test['YearsCoding'] = df_train_test['YearsCoding'].astype('object')



df_train_test=df_train_test.replace({'CompanySize':{'1,000 to 4,999 employees':3,
                                                    '5,000 to 9,999 employees':2,
                                                    '10 to 19 employees':7,
                                                    '10,000 or more employees':1,
                                                    '20 to 99 employees':6,
                                                    '100 to 499 employees':5,
                                                    '500 to 999 employees':4,
                                                    'Fewer than 10 employees':8}})

df_train_test['CompanySize'] = df_train_test['CompanySize'].astype('object')





df_train_test['Student_oh']=df_train_test['Student']
df_train_test['Hobby_oh']=df_train_test['Hobby']
df_train_test['OpenSource_oh']=df_train_test['OpenSource']
df_train_test = pd.get_dummies(df_train_test, columns=['Student_oh','Hobby_oh','OpenSource_oh'])
df_train_test['AssessJob_mean']=df_train_test['AssessJob1']+df_train_test['AssessJob2']+df_train_test['AssessJob3']+df_train_test['AssessJob4']+df_train_test['AssessJob5']+df_train_test['AssessJob6']+df_train_test['AssessJob7']+df_train_test['AssessJob8']+df_train_test['AssessJob9']+df_train_test['AssessJob10']
df_train_test['JobEmailPriorities_mean']=df_train_test['JobEmailPriorities1']+df_train_test['JobEmailPriorities2']+df_train_test['JobEmailPriorities3']+df_train_test['JobEmailPriorities4']+df_train_test['JobEmailPriorities5']+df_train_test['JobEmailPriorities6']+df_train_test['JobEmailPriorities7']
df_train_test['JobContactPriorities_mean']=df_train_test['JobContactPriorities1']+df_train_test['JobContactPriorities2']+df_train_test['JobContactPriorities3']+df_train_test['JobContactPriorities4']+df_train_test['JobContactPriorities5']
df_train_test['AssessBenefits_mean']=df_train_test['AssessBenefits1']+df_train_test['AssessBenefits2']+df_train_test['AssessBenefits3']+df_train_test['AssessBenefits4']+df_train_test['AssessBenefits5']+df_train_test['AssessBenefits6']+df_train_test['AssessBenefits7']+df_train_test['AssessBenefits8']+df_train_test['AssessBenefits9']+df_train_test['AssessBenefits10']+df_train_test['AssessBenefits11']
df_train_test
x_train = df_train_test[df_train_test['istrain'] == 1]
x_test  = df_train_test[df_train_test['istrain'] == 0]
x_train  = x_train.drop(['istrain'], axis=1)
x_test  = x_test.drop(['ConvertedSalary', 'istrain'], axis=1)

y_train = x_train.ConvertedSalary
x_train = x_train.drop(['ConvertedSalary'], axis=1)
# 不要な列を削除（重要度が0.0）

#delcols = ['acc_now_delinq','application_type','collections_12_mths_ex_med','pub_rec','delinq_2yrs','def_cn']


#x_train = x_train.drop(delcols, axis=1)
#x_test = x_test.drop(delcols, axis=1)
#gc.collect()
cats = []
for col in x_train.columns:
    if x_train[col].dtype == 'object':
        cats.append(col)
        
        print(col, x_train[col].nunique())
cats.remove('YearsCodingProf')
cats.remove('YearsCoding')
cats.remove('CompanySize')
oe = OrdinalEncoder(cols=cats, return_df=False)

x_train[cats] = oe.fit_transform(x_train[cats])
x_test[cats] = oe.transform(x_test[cats])
target = 'ConvertedSalary'
x_temp = pd.concat([x_train, y_train], axis=1)

for col in cats:

    # X_testはX_trainでエンコーディングする
    summary = x_temp.groupby([col])[target].mean()
    x_test[col] = x_test[col].map(summary) 
#    X_test[col + "_tgtenc"] = X_test[col].map(summary) 

    skf = KFold(n_splits=5, random_state=71, shuffle=True)
#    skf = KFold(n_splits=5, random_state=71, shuffle=True)
#    skf = ShuffleSplit(n_splits=5, random_state=71)
    
    enc_train = Series(np.zeros(len(x_train)), index=x_train.index)

    for i, (train_ix, val_ix) in enumerate((skf.split(x_train, y_train))):
        x_train_, _ = x_temp.iloc[train_ix], y_train.iloc[train_ix]
        x_val, _ = x_temp.iloc[val_ix], y_train.iloc[val_ix]

        summary = x_train_.groupby([col])[target].mean()
        enc_train.iloc[val_ix] = x_val[col].map(summary)
        
    x_train[col]  = enc_train
x_train.fillna(x_train.mean(), axis=0, inplace=True)
x_test.fillna(x_train.mean(), axis=0, inplace=True)
display(x_train.shape)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
scores = []
y_pred_hold = np.zeros(len(x_test))

skf = KFold(n_splits=5, random_state=71, shuffle=True)
        

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(x_train, y_train))):
    x_train_, y_train_ = x_train.values[train_ix], y_train.values[train_ix]
    x_val, y_val = x_train.values[test_ix], y_train.values[test_ix]
    
#    clf =  RandomForestRegressor()
    clf = LGBMRegressor(n_estimators=9999, colsample_bytree=0.9,
                        learning_rate=0.05, min_child_samples=20,min_child_weight=0.001, min_split_gain=0.0, num_leaves=15)   
    clf.fit(x_train_, y_train_)
    y_val_pred = clf.predict(x_val)
    print(rmsle(y_val,y_val_pred))
    
    y_pred_hold += clf.predict(x_test)
y_pred_1 = y_pred_hold/5

scores = []
y_pred_hold = np.zeros(len(x_test))

skf = KFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(x_train, y_train))):
    x_train_, y_train_ = x_train.values[train_ix], y_train.values[train_ix]
    x_val, y_val = x_train.values[test_ix], y_train.values[test_ix]
    
    clf =  RandomForestRegressor()
    
    clf.fit(x_train_, y_train_)
    y_val_pred = clf.predict(x_val)
    print(rmsle(y_val,y_val_pred))
    
    y_pred_hold += clf.predict(x_test)
y_pred_2 = y_pred_hold/5
ypred=(y_pred_1+y_pred_2)/2
submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)

submission.ConvertedSalary = ypred
submission.to_csv('submission.csv')
ypred