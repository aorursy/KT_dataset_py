import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import time

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn.metrics import roc_auc_score, classification_report

import lightgbm as lgb



import gc

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

# test = pd.read_csv('../input/test.csv')

features = pd.read_csv('../input/user_features.csv')

sub = pd.read_csv('../input/sample_submission_only_headers.csv')
gc.collect()
def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
train, NAlist = reduce_mem_usage(train)
# test, NAlist = reduce_mem_usage(test)
features, NAlist = reduce_mem_usage(features)
train.head()
# test.head()
%%time

train = train.merge(features,how = 'left',left_on='node1_id',right_on='node_id')

train = train.merge(features,how = 'left',left_on='node2_id',right_on='node_id')
# %%time

# test = test.merge(features,how = 'left',left_on='node1_id',right_on='node_id')

# test = test.merge(features,how = 'left',left_on='node2_id',right_on='node_id')
train.drop(['node_id_x','node_id_y','node1_id','node2_id'], axis=1, inplace= True)

# test.drop(['node_id_x','node_id_y','node1_id','node2_id'], axis=1, inplace= True)
train.head()
# test.head()
# test_id = test.pop('id')

X = train.drop('is_chat', axis=1)

y = train.is_chat
del train
# shape of all the files

X.shape, y.shape
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=1999111000)
del X

del y

del x_train

del y_train
x_tr,x_val,y_tr,y_val = train_test_split(x_test,y_test, test_size = 0.1, random_state=1999111000)
del x_test

del y_test
gc.collect()
print(x_tr.shape)

print(y_tr.shape)

print(x_val.shape)

print(y_val.shape)
x_tr.columns
x_tr['f1_change'] = x_tr['f1_x'] - x_tr['f1_y']

x_tr['f2_change'] = x_tr['f2_x'] - x_tr['f2_y']

x_tr['f3_change'] = x_tr['f3_x'] - x_tr['f3_y']

x_tr['f4_change'] = x_tr['f4_x'] - x_tr['f4_y']

x_tr['f5_change'] = x_tr['f5_x'] - x_tr['f5_y']

x_tr['f6_change'] = x_tr['f6_x'] - x_tr['f6_y']

x_tr['f7_change'] = x_tr['f7_x'] - x_tr['f7_y']

x_tr['f8_change'] = x_tr['f8_x'] - x_tr['f8_y']

x_tr['f9_change'] = x_tr['f9_x'] - x_tr['f9_y']

x_tr['f10_change'] = x_tr['f10_x'] - x_tr['f10_y']

x_tr['f11_change'] = x_tr['f11_x'] - x_tr['f11_y']

x_tr['f12_change'] = x_tr['f12_x'] - x_tr['f12_y']

x_tr['f13_change'] = x_tr['f13_x'] - x_tr['f13_y']





x_val['f1_change'] = x_val['f1_x'] - x_val['f1_y']

x_val['f2_change'] = x_val['f2_x'] - x_val['f2_y']

x_val['f3_change'] = x_val['f3_x'] - x_val['f3_y']

x_val['f4_change'] = x_val['f4_x'] - x_val['f4_y']

x_val['f5_change'] = x_val['f5_x'] - x_val['f5_y']

x_val['f6_change'] = x_val['f6_x'] - x_val['f6_y']

x_val['f7_change'] = x_val['f7_x'] - x_val['f7_y']

x_val['f8_change'] = x_val['f8_x'] - x_val['f8_y']

x_val['f9_change'] = x_val['f9_x'] - x_val['f9_y']

x_val['f10_change'] = x_val['f10_x'] - x_val['f10_y']

x_val['f11_change'] = x_val['f11_x'] - x_val['f11_y']

x_val['f12_change'] = x_val['f12_x'] - x_val['f12_y']

x_val['f13_change'] = x_val['f13_x'] - x_val['f13_y']
x_tr, NAlist = reduce_mem_usage(x_tr)

x_val, NAlist = reduce_mem_usage(x_val)
param = {}

param['learning_rate'] = 0.1

param['boosting_type'] = 'gbdt'

param['objective'] = 'binary'

param['metric'] = 'auc'

param['sub_feature'] = 0.6

param['num_leaves'] = 31

param['feature_fraction'] = 0.8

param['bagging_fraction'] = 0.7

param['min_data'] = 100

param['max_depth'] = 10
trn_data = lgb.Dataset(x_tr, label=y_tr)

val_data = lgb.Dataset(x_val, label=y_val)
del features
import sys # These are the usual ipython objects, including this one you are creating 

ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars'] # Get a sorted list of the objects and their sizes 

sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
clf = lgb.train(param, trn_data, num_boost_round=2000, valid_sets = [trn_data, val_data], 

                verbose_eval=5, early_stopping_rounds = 50)
del x_tr

del y_tr

del x_val

del y_val
## All preprocessing on test data

test = pd.read_csv('../input/test.csv')

features = pd.read_csv('../input/user_features.csv')

test, NAlist = reduce_mem_usage(test)

features, NAlist = reduce_mem_usage(features)



test = test.merge(features,how = 'left',left_on='node1_id',right_on='node_id')

test = test.merge(features,how = 'left',left_on='node2_id',right_on='node_id')



test.drop(['node_id_x','node_id_y','node1_id','node2_id'], axis=1, inplace= True)



test['f1_change'] = test['f1_x'] - test['f1_y']

test['f2_change'] = test['f2_x'] - test['f2_y']

test['f3_change'] = test['f3_x'] - test['f3_y']

test['f4_change'] = test['f4_x'] - test['f4_y']

test['f5_change'] = test['f5_x'] - test['f5_y']

test['f6_change'] = test['f6_x'] - test['f6_y']

test['f7_change'] = test['f7_x'] - test['f7_y']

test['f8_change'] = test['f8_x'] - test['f8_y']

test['f9_change'] = test['f9_x'] - test['f9_y']

test['f10_change'] = test['f10_x'] - test['f10_y']

test['f11_change'] = test['f11_x'] - test['f11_y']

test['f12_change'] = test['f12_x'] - test['f12_y']

test['f13_change'] = test['f13_x'] - test['f13_y']



del features



test, NAlist = reduce_mem_usage(test)



test_id = test.pop('id')
predictions = clf.predict(test, num_iteration=clf.best_iteration)
sub["id"] = test_id

sub["is_chat"] = predictions

sub.to_csv("submission.csv", index=False)

sub.head()