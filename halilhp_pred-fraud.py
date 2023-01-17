# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
%matplotlib inline

# Standard plotly imports

import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff
import os
print(os.listdir("../input"))
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]




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

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

import os
import gc
print(os.listdir("../input"))
# Read train and test data with pd.read_csv():
train_id= pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
test_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
train_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
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
train_tr = reduce_mem_usage(train_tr)
train_id = reduce_mem_usage(train_id)
train_id.head()

test_id.head()
train_tr.head()
test_tr.head()
train_id.info()
test_id.info()
train_tr.info()
test_tr.info()
print('train_transaction shape is {}'.format(train_tr.shape))
print('test_transaction shape is {}'.format(test_tr.shape))
print('train_identity shape is {}'.format(train_id.shape))
print('test_identity shape is {}'.format(test_id.shape))

missing_values_count = train_tr.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_tr.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)
missing_values_count = train_id.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_id.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)
# Here we confirm that all of the transactions in `train_identity`
print(np.sum(train_tr['TransactionID'].isin(train_id['TransactionID'].unique())))
print(np.sum(test_tr['TransactionID'].isin(test_id['TransactionID'].unique())))

print('  {:.4f}% of Transactions that are fraud in train '.format(train_tr['isFraud'].mean() * 100))
train_tr.groupby('isFraud') \
    .count()['TransactionID'] \
    .plot(kind='barh',
          title='Is Fraud?',
          figsize=(15, 3))
plt.show()

train_tr['TransactionAmt'] \
    .apply(np.log) \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribution of Log Transaction Amt')
plt.show()
print('Mean transaction amt for fraud is {:.4f}'.format(train_tr.loc[train_tr['isFraud'] == 1]['TransactionAmt'].mean()))
print('Mean transaction amt for non-fraud is {:.4f}'.format(train_tr.loc[train_tr['isFraud'] == 0]['TransactionAmt'].mean()))
train_tr.groupby('ProductCD') \
    ['TransactionID'].count() \
    .sort_index() \
    .plot(kind='barh',
          figsize=(20, 5),
         title='Count of Observations by ProductCD')
plt.show()
train_tr.groupby('ProductCD')['isFraud'] \
    .mean() \
    .sort_index() \
    .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()

card_cols = [c for c in train_tr.columns if 'card' in c]
train_tr[card_cols].head()
color_idx = 0
for c in card_cols:
    if train_tr[c].dtype in ['float64','int64']:
        train_tr[c].plot(kind='hist',
                                      title=c,
                                      bins=50,
                                      figsize=(15, 2),
                                      color=color_pal[color_idx])
    color_idx += 1
    plt.show()
    
train_tr.head()
train=pd.merge(train_tr, train_id, on = "TransactionID",how='left',left_index=True, right_index=True)
train.head()
test=pd.merge(test_tr, test_id, on = "TransactionID",how='left',left_index=True, right_index=True)
test.head()
del train_id, train_tr, test_id, test_tr

!pip install missingno
import missingno as msno
msno.matrix(train.iloc[:,:46]);
msno.matrix(train.iloc[:,40:80]);
msno.matrix(train.iloc[:,80:120]);
msno.matrix(train.iloc[:,120:160]);
msno.matrix(train.iloc[:,160:200]);
msno.matrix(train.iloc[:,200:240]);
msno.matrix(train.iloc[:,240:280]);
msno.matrix(train.iloc[:,280:320]);
msno.matrix(train.iloc[:,320:360]);
msno.matrix(train.iloc[:,360:400]);




