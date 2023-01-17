import numpy as np  

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/odi-match-winner/ODI_Participants_Data/Train.csv')

test = pd.read_csv('/kaggle/input/odi-match-winner/ODI_Participants_Data/Test.csv')

sub = pd.read_excel('/kaggle/input/odi-match-winner/ODI_Participants_Data/Sample_submission.xlsx')
train.shape, test.shape, sub.shape
train.head(5)
train.isnull().sum()
train.nunique()
train.dtypes
train['MatchWinner'].value_counts()
df = train.append(test,ignore_index=True)

df.shape
calc = df.groupby(['Team1'], axis=0).agg({'Team1':[('op1', 'count')]}).reset_index() 

calc.columns = ['Team1','Team1 Count']

df = df.merge(calc, on=['Team1'], how='left')



calc = df.groupby(['Team2'], axis=0).agg({'Team2':[('op1', 'count')]}).reset_index() 

calc.columns = ['Team2','Team2 Count']

df = df.merge(calc, on=['Team2'], how='left')



calc = df.groupby(['Stadium'], axis=0).agg({'Stadium':[('op1', 'count')]}).reset_index() 

calc.columns = ['Stadium','Stadium Count']

df = df.merge(calc, on=['Stadium'], how='left')



calc = df.groupby(['HostCountry'], axis=0).agg({'HostCountry':[('op1', 'count')]}).reset_index() 

calc.columns = ['HostCountry','HostCountry Count']

df = df.merge(calc, on=['HostCountry'], how='left')



calc = df.groupby(['Team1_Venue'], axis=0).agg({'Team1_Venue':[('op1', 'count')]}).reset_index() 

calc.columns = ['Team1_Venue','Team1_Venue Count']

df = df.merge(calc, on=['Team1_Venue'], how='left')



calc = df.groupby(['Team2_Venue'], axis=0).agg({'Team2_Venue':[('op1', 'count')]}).reset_index() 

calc.columns = ['Team2_Venue','Team2_Venue Count']

df = df.merge(calc, on=['Team2_Venue'], how='left')



calc = df.groupby(['Team1_Innings'], axis=0).agg({'Team1_Innings':[('op1', 'count')]}).reset_index() 

calc.columns = ['Team1_Innings','Team1_Innings Count']

df = df.merge(calc, on=['Team1_Innings'], how='left')



calc = df.groupby(['Team2_Innings'], axis=0).agg({'Team2_Innings':[('op1', 'count')]}).reset_index() 

calc.columns = ['Team2_Innings','Team2_Innings Count']

df = df.merge(calc, on=['Team2_Innings'], how='left')



calc = df.groupby(['MonthOfMatch'], axis=0).agg({'MonthOfMatch':[('op1', 'count')]}).reset_index() 

calc.columns = ['MonthOfMatch','MonthOfMatch Count']

df = df.merge(calc, on=['MonthOfMatch'], how='left')
agg_func = {

    'Team2': ['count','nunique'],

    'Stadium': ['count','nunique'],

    'HostCountry': ['count','nunique'],

    'MonthOfMatch': ['count','nunique'],

}

agg_func = df.groupby('Team1').agg(agg_func)

agg_func.columns = [ 'Team1_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Team1'], how='left')



agg_func = {

    'Team1': ['count','nunique'],

    'Stadium': ['count','nunique'],

    'HostCountry': ['count','nunique'],

    'MonthOfMatch': ['count','nunique'],

}

agg_func = df.groupby('Team2').agg(agg_func)

agg_func.columns = [ 'Team2_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Team2'], how='left')
cols = ['Team1', 'Team2', 'Stadium', 'HostCountry', 'Team1_Venue','Team2_Venue', 'Team1_Innings', 'Team2_Innings', 'MonthOfMatch']

df = pd.get_dummies(df, columns=cols, drop_first=True)
df.head(3)
df.head(2)
train_df = df[df['MatchWinner'].isnull()!=True]

test_df = df[df['MatchWinner'].isnull()==True]

test_df.drop(['MatchWinner'], axis=1, inplace=True)
X = train_df.drop(labels=['MatchWinner'], axis=1)

y = train_df['MatchWinner'].values



X.shape, y.shape
X.head(3)
Xtest = test_df
import lightgbm as lgb



param = {'objective': 'multiclass',

         'num_class': 16,

         'boosting': 'gbdt',  

         'metric': 'multi_logloss',

         'learning_rate': 0.1,

         'num_iterations': 500,

         'max_depth': -1,

         'min_data_in_leaf': 20,

         'bagging_fraction': 0.7,

         'bagging_freq': 1,

         'feature_fraction': 0.7

         }
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss



err_lgb = []

y_pred_tot_lgb = []



fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



for train_index, test_index in fold.split(X, y):

    

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    test_data = lgb.Dataset(X_test, label=y_test)

    

    clf = lgb.train(params=param, 

                    early_stopping_rounds=100,

                    verbose_eval=0,

                    train_set=train_data,

                    valid_sets=[test_data])



    y_pred = clf.predict(X_test)

         

    print("Log Loss:", log_loss(y_test, y_pred))



    err_lgb.append(log_loss(y_test, y_pred))

    p = clf.predict(Xtest)

    y_pred_tot_lgb.append(p)
np.mean(err_lgb,0) 
final = np.mean(y_pred_tot_lgb,0)

sub = pd.DataFrame(final)
sub.head()
sub.to_excel('Output.xlsx', index=False)