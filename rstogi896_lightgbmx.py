import os

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

from sklearn import preprocessing

from wordcloud import WordCloud

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly import tools

from datetime import date

import lightgbm as lgb
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
train_transaction.head()
train_transaction.shape
import seaborn as sns

ax = train_transaction['isFraud'].value_counts().plot.bar(

    figsize = (12,6),

    color = 'mediumvioletred',

    fontsize = 16

)

ax.set_title('isFraud',fontsize = 20)

sns.despine(bottom = True, left = True)
ax = train_transaction['card4'].value_counts().plot.bar(

    figsize = (12,6),

    color = 'mediumvioletred',

    fontsize = 16

)

ax.set_title('Card Type',fontsize = 20)

sns.despine(bottom = True, left = True)
ax = train_transaction['card6'].value_counts().plot.bar(

    figsize = (12,6),

    color = 'mediumvioletred',

    fontsize = 16

)

ax.set_title('card6',fontsize = 20)

sns.despine(bottom = True, left = True)
ax = train_transaction['P_emaildomain'].value_counts().plot.bar(

    figsize = (12,6),

    color = 'mediumvioletred',

    fontsize = 16

)

ax.set_title('P_emaildomain',fontsize = 20)

sns.despine(bottom = True, left = True)
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)



y_train = train['isFraud'].copy()

del train_transaction, train_identity, test_transaction, test_identity



# Drop target, fill in NaNs

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()



del train, test



X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)
# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))   
%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
X_train.shape,X_test.shape
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=18)

lgb_train = lgb.Dataset(data=x_train, label=y_train)

lgb_eval = lgb.Dataset(data=x_val, label=y_val)

params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 

          'learning_rate': 0.05, 'num_leaves': 200, 'num_iteration': 450, 'verbose': 0 ,

          'colsample_bytree':.8, 'subsample':.9, 'max_depth':18, 'reg_alpha':.1, 'reg_lambda':.1, 

          'min_split_gain':.01, 'min_child_weight':1}

model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=60, verbose_eval=60)
preds = model.predict(X_test)
sample_submission['isFraud'] = preds

sample_submission.to_csv('deepak.csv')