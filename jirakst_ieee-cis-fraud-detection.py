!pip install datatable==0.11.0
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datatable as dt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID', nrows=10**5)

train_t.shape



'''

#dt.fread("../input/riiid-test-answer-prediction/train.csv").to_jay("train.jay")

train_t = dt.fread("/kaggle/input/ieee-fraud-detection/train_transaction.csv").to_pandas()

train_t.set_index('TransactionID')

train_t.shape

'''
# Select frauds

train_t_f = train_t.loc[train_t['isFraud'] == 1]

train_t_f.shape
# Select not frauds

train_t_nf = train_t.loc[train_t['isFraud'] == 0]

train_t_nf = train_t_nf.sample(frac=0.1).iloc[0:2560]

train_t_nf.shape
# Create balanced dataset

train_t = pd.concat([train_t_f, train_t_nf], axis=0)

train_t.shape



del train_t_f, train_t_nf
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID', nrows=10**5)

train_id.shape

'''

train_id = dt.fread("/kaggle/input/ieee-fraud-detection/train_identity.csv").to_pandas()

train_id.set_index('TransactionID')

train_id.shape

'''
df_train = train_t.merge(train_id, on='TransactionID', how='left')

df_train.shape



# Release memory

del train_t

del train_id
'''

test_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

test_t.shape

'''



dt.fread("/kaggle/input/ieee-fraud-detection/test_transaction.csv").to_jay("test_transaction.jay")

test_t = dt.fread("test_transaction.jay").to_pandas()

test_t.set_index('TransactionID')

test_t.shape
'''

test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

test_id.shape

'''



dt.fread("/kaggle/input/ieee-fraud-detection/test_identity.csv").to_jay("test_identity.jay")

test_id = dt.fread("test_identity.jay").to_pandas()

test_id.set_index('TransactionID')

test_id.shape
df_test = test_t.merge(test_id, on='TransactionID', how='left')

df_test.shape
# Release memory

del test_t

del test_id
sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

sub.shape
df_train.head()
df_train.describe()
# Target distribution

import seaborn as sns



g = sns.countplot(x='isFraud', data=df_train, )

g.set_title("Fraud Transactions Distribution", fontsize=18)

g.set_xlabel("Is fraud?", fontsize=14)

g.set_ylabel('Count', fontsize=14)
df_train.info()
df_train.isnull().sum()
# Concat dataset for better manipulation

split = len(df_train)

target = df_train.isFraud

df = pd.concat([df_train, df_test], axis=0).drop('isFraud', axis=1)



del df_train, df_test
# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt

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
for c in ['P_emaildomain', 'R_emaildomain']:

    df[c + '_bin'] = df[c].map(emails)    

    df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])    

    df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
# Encoding

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



for f in df.columns:

    if df[f].dtype=='object':

        df[f] = le.fit_transform(list(df[f].values))
# Impute nulls

df = df.fillna(-1)
#TODO: Feature engineering
#TODO: Box-Cox?
#TODO: Scalling

from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

df_ss = ss.fit_transform(df.iloc[:,3:470].values)

df_ss = pd.DataFrame(df_ss, index=df.index, columns=df.iloc[:,3:470].columns)
# Dimensionality reduction

from sklearn.decomposition import PCA



pca = PCA(.999) # retain 95% of the variance #PCA(n_components=10)

df_pca = pca.fit_transform(df.iloc[:,3:470])

df_pca = pd.DataFrame(df_pca, index=df.index) # Convert to df

#print(pca.explained_variance_ratio_.sum())

print(pca.n_components_)
# TODO: Use df, df_c is for dev purpose

df_c = pd.concat([df.iloc[:,:3], df_pca, df.iloc[:,470:]], axis=1)

df_c.shape
df_train = df_c[:split]

df_test = df_c[split:]



#del df
# Get train and validation sub-datasets

from sklearn.model_selection import train_test_split



X = df_train

y = target



#Do train data splitting

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, random_state=42)
import xgboost as xgb



model = xgb.XGBClassifier(

    n_estimators=1000,

    tree_method='gpu_hist',

    eval_metric='auc'

)
model.fit(X_train, y_train, 

          eval_set=[(X_test, y_test)], 

          verbose=50, 

          early_stopping_rounds=300)
# Use whole training data

model.fit(df_train, target)
y_pred = model.predict_proba(df_test)[:,1] 

y_pred
# Create submission

sub['isFraud'] = y_pred

sub.to_csv('submission.csv', index=False)
sub.head()