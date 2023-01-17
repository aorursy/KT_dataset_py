import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Standard plotly imports

#import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

import plotly.figure_factory as ff



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)



# Preprocessing, modelling and evaluating

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



import os

import gc

#print(os.listdir("../input"))
path = os.getcwd()
df_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

df_test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



df_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

df_test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')



sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True, on='TransactionID')

df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True, on='TransactionID')



print(df_train.shape)

print(df_test.shape)



# y_train = df_train['isFraud'].copy()

del df_trans, df_id, df_test_trans, df_test_id
## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
## REducing memory

df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 

          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',

          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 

          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',

          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',

          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 

          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 

          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',

          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',

          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 

          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 

          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 

          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 

          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',

          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}



us_emails = ['gmail', 'net', 'edu']



# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654

for c in ['P_emaildomain', 'R_emaildomain']:

    df_train[c + '_bin'] = df_train[c].map(emails)

    df_test[c + '_bin'] = df_test[c].map(emails)

    

    df_train[c + '_suffix'] = df_train[c].map(lambda x: str(x).split('.')[-1])

    df_test[c + '_suffix'] = df_test[c].map(lambda x: str(x).split('.')[-1])

    

    df_train[c + '_suffix'] = df_train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    df_test[c + '_suffix'] = df_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
df_train = df_train.fillna(-99)

df_test = df_test.fillna(-99)

# fill missing values with mean column values

#df_train = df_train.fillna(df_train.mean(), inplace=True)
for i in ['card1','card2','card3','card4','card5','card6','addr1','addr2']:

    df_train[i] = df_train[i].astype('object')

    df_test[i] = df_test[i].astype('object')
df_train['card1'].dtypes
dict(zip(list(df_train.columns),list(df_train.dtypes)))
df_train['card1'].dtype
categorical = []

for f in df_train.drop('isFraud', axis=1).columns:

    if df_train[f].dtype=='object' or df_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_train[f].values) + list(df_test[f].values))

        df_train[f] = lbl.transform(list(df_train[f].values))

        df_test[f] = lbl.transform(list(df_test[f].values)) 

        categorical.append(df_train[f].name)  

        print(df_train[f].name)
categorical
df_test['isFraud'] = 'test'

df = pd.concat([df_train, df_test], axis=0, sort=False )

df = df.reset_index()

df = df.drop('index', axis=1)
pd.set_option('display.max_columns',None)

pd.set_option("display.max_rows", 101)

df.columns.values.tolist()
def PCA_change(df, cols, n_components, prefix='PCA_', rand_seed=4):

    pca = PCA(n_components=n_components, random_state=rand_seed)



    principalComponents = pca.fit_transform(df[cols])



    principalDf = pd.DataFrame(principalComponents)



    df.drop(cols, axis=1, inplace=True)



    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)



    df = pd.concat([df, principalDf], axis=1)

    

    return df
mas_v = df_train.columns[55:394]
from sklearn.preprocessing import minmax_scale

from sklearn.decomposition import PCA

# from sklearn.cluster import KMeans



for col in mas_v:

    df[col] = df[col].fillna((df[col].min() - 2))

    df[col] = (minmax_scale(df[col], feature_range=(0,1)))



    

df = PCA_change(df, mas_v, prefix='PCA_V_', n_components=30)
df = reduce_mem_usage(df)
df_train, df_test = df[df['isFraud'] != 'test'], df[df['isFraud'] == 'test'].drop('isFraud', axis=1)
X_train = df_train.sort_values('TransactionID').drop(['isFraud', 

                                                      'TransactionID', 

                                                      #'Card_ID'

                                                     ],

                                                     axis=1)

y_train = df_train.sort_values('TransactionID')['isFraud'].astype(bool)



df_test = df_test.sort_values('TransactionID')



X_test = df_test.drop(['TransactionID'], axis=1)



del df_train

df_test = df_test[["TransactionID"]]
y_train = y_train.astype(int)
import lightgbm as lgbm

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split



params = {

    'num_leaves': [500,600,700,800],

    'feature_fraction': list(np.arange(0.1,0.5,0.1)),

    'bagging_fraction': list(np.arange(0.1,0.5,0.1)),

    'min_data_in_leaf': [100,120,140,160],

    'learning_rate': [0.1],

    'reg_alpha': list(np.arange(0.1,0.5,0.1)),

    'reg_lambda': list(np.arange(0.1,0.5,0.1)),

}

model = lgbm.LGBMClassifier(random_state=42,metric='auc',verbosity=-1,objective='binary',max_depth=-1)

grid = RandomizedSearchCV(model,param_distributions=params,n_iter=15,cv=3,scoring='roc_auc')
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)



grid.fit(X_train2,y_train2)



print(grid.best_params_)



print(grid.best_score_)
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_valid2,grid.predict(X_valid2)))
a = roc_auc_score(y_valid2,grid.predict_proba(X_valid2)[:,1])

print(a)
#df_test['isFraud'] = a
#df_test.to_csv('mysubmission_test.csv',index=False)