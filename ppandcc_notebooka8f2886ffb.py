# ! _*_ coding:utf-8 _*_

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train_2016_v2.csv",parse_dates=["transactiondate"])

print(train_df.shape)

train_df.head()
plt.scatter(range(train_df.shape[0]),np.sort(train_df.logerror))

plt.show()
train_df = pd.read_csv("../input/properties_2016.csv")

print(train_df.shape)

train_df.head()
train_df = pd.read_csv("../input/properties_2017.csv")

print(train_df.shape)

train_df.head()
train_df = pd.read_csv("../input/sample_submission.csv")

print(train_df.shape)

train_df.head()
train_df = pd.read_csv("../input/train_2017.csv")

print(train_df.shape)

train_df.head()
train_df = pd.read_excel("../input/zillow_data_dictionary.xlsx")

print(train_df.shape)

train_df.head()
import numpy as np

import pandas as pd

import xgboost as xgb

import gc

print("Loading data ...")
"""

properties_2016.csv

properties_2017.csv

sample_submission.csv

train_2016_v2.csv

train_2017.csv

zillow_data_dictionary.xlsx

"""





train = pd.read_csv("../input/train_2016_v2.csv")

prop = pd.read_csv("../input/properties_2016.csv")

sample = pd.read_csv("../input/sample_submission.csv")

print("Binding to float32")

for c,dtype in zip(prop.columns,prop.dtypes):

    if dtype == np.float64:

        prop[c] = prop[c].astype(np.float32)
print("Creating training set ...")

df_train = train.merge(prop,how="left",on="parcelid")

x_train = df_train.drop(['parcelid','logerror','transactiondate','propertyzoningdesc',

                        'propertycountylandusecode'],axis=1)

y_train = df_train['logerror'].values

print(x_train.shape,y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes==object].index.values:

    x_train[c]=(x_train[c]==True)

del df_train;gc.collect()

split = 80000

x_train,y_train,x_valid,y_valid = x_trian[:split],y_train[:split],x_train[split:],y_train[split:]
print('Nuilding DMatrix ...')

d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train,x_valid;gc.collect()
# Parameters

XGB_WEIGHT = 0.6500

BASELINE_WEIGHT = 0.0056

BASELINE_PRED = 0.0115

import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import gc

##### READ IN RAW DATA

print( "\nReading data from disk ...")

prop = pd.read_csv('../input/properties_2016.csv')

train = pd.read_csv("../input/train_2016_v2.csv")

##### PROCESS DATA FOR LIGHTGBM

print( "\nProcessing data for LightGBM ..." )

for c, dtype in zip(prop.columns, prop.dtypes):	

    if dtype == np.float64:		

        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 

                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)

y_train = df_train['logerror'].values

print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)

##### RUN LIGHTGBM

params = {}

params['max_bin'] = 10

params['learning_rate'] = 0.0021 # shrinkage_rate

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'l1'          # or 'mae'

params['sub_feature'] = 0.5      # feature_fraction -- OK, back to .5, but maybe later increase this

params['bagging_fraction'] = 0.85 # sub_row

params['bagging_freq'] = 40

params['num_leaves'] = 512        # num_leaf

params['min_data'] = 500         # min_data_in_leaf

params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

params['verbose'] = 0

print("\nFitting LightGBM model ...")

clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()

del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")

print("   Read sample file ...")

sample = pd.read_csv('../input/sample_submission.csv')

print("   ...")

sample['parcelid'] = sample['ParcelId']

print("   Merge with property data ...")

df_test = sample.merge(prop, on='parcelid', how='left')

print("   ...")

del sample, prop; gc.collect()

print("   ...")

x_test = df_test[train_columns]

print("   ...")

del df_test; gc.collect()

print("   Preparing x_test...")

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)

print("   ...")

x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")

# num_threads > 1 will predict very slow in kernal

clf.reset_parameter({"num_threads":1})

p_test = clf.predict(x_test)

del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )

print( pd.DataFrame(p_test).head() )

##### RE-READ PROPERTIES FILE

##### (I tried keeping a copy, but the program crashed.)

print( "\nRe-reading properties file ...")

properties = pd.read_csv('../input/properties_2016.csv')

##### PROCESS DATA FOR XGBOOST

print( "\nProcessing data for XGBoost ...")

for c in properties.columns:

    properties[c]=properties[c].fillna(-1)

    if properties[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(properties[c].values))

        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')

x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

x_test = properties.drop(['parcelid'], axis=1)

# shape        

print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers

train_df=train_df[ train_df.logerror > -0.4 ]

train_df=train_df[ train_df.logerror < 0.418 ]

x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

y_train = train_df["logerror"].values.astype(np.float32)

y_mean = np.mean(y_train)

print('After removing outliers:')     

print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")

# xgboost params

xgb_params = {

    'eta': 0.037,

    'max_depth': 5,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'lambda': 0.8,   

    'alpha': 0.4, 

    'base_score': y_mean,

    'silent': 1

}

# Enough with the ridiculously overfit parameters.

# I'm going back to my version 20 instead of copying Jayaraman.

# I want a num_boost_rounds that's chosen by my CV,

# not one that's chosen by overfitting the public leaderboard.

# (There may be underlying differences between the train and test data

#  that will affect some parameters, but they shouldn't affect that.)

dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)

# cross-validation

#print( "Running XGBoost CV ..." )

#cv_result = xgb.cv(xgb_params, 

#                   dtrain, 

#                   nfold=5,

#                   num_boost_round=350,

#                   early_stopping_rounds=50,

#                   verbose_eval=10, 

#                   show_stdv=False

#                  )

#num_boost_rounds = len(cv_result)

# num_boost_rounds = 150

num_boost_rounds = 242

print("\nXGBoost tuned with CV in:")

print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")

print("num_boost_rounds="+str(num_boost_rounds))

# train model

print( "\nTraining XGBoost ...")

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")

xgb_pred = model.predict(dtest)

print( "\nXGBoost predictions:" )

print( pd.DataFrame(xgb_pred).head() )

##### COMBINE PREDICTIONS

print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )

lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT

pred = XGB_WEIGHT*xgb_pred + BASELINE_WEIGHT*BASELINE_PRED + lgb_weight*p_test

print( "\nCombined predictions:" )

print( pd.DataFrame(pred).head() )

##### WRITE THE RESULTS

print( "\nPreparing results for write ..." )

y_pred=[]

for i,predict in enumerate(pred):

    y_pred.append(str(round(predict,4)))

y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),

        '201610': y_pred, '201611': y_pred, '201612': y_pred,

        '201710': y_pred, '201711': y_pred, '201712': y_pred})

# set col 'ParceID' to first col

cols = output.columns.tolist()

cols = cols[-1:] + cols[:-1]

output = output[cols]

from datetime import datetime

print( "\nWriting results to disk ..." )

output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ..." )
"""

properties_2016.csv

properties_2017.csv

sample_submission.csv

train_2016_v2.csv

train_2017.csv

zillow_data_dictionary.xlsx

"""



import numpy as np

import pandas as pd

import lightgbm as lgb

import gc

print('Loading data ...')

train = pd.read_csv('../input/train_2016_v2.csv')

prop = pd.read_csv('../input/properties_2016.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	

    if dtype == np.float64:		

        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

y_train = df_train['logerror'].values

print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 90000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

x_train = x_train.values.astype(np.float32, copy=False)

x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)

d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}

params['learning_rate'] = 0.002

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'mae'

params['sub_feature'] = 0.5

params['num_leaves'] = 60

params['min_data'] = 500

params['min_hessian'] = 1

watchlist = [d_valid]

clf = lgb.train(params, d_train, 500, watchlist)

del d_train, d_valid; gc.collect()

del x_train, x_valid; gc.collect()

print("Prepare for the prediction ...")

sample = pd.read_csv('../input/sample_submission.csv')

sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(prop, on='parcelid', how='left')

del sample, prop; gc.collect()

x_test = df_test[train_columns]

del df_test; gc.collect()

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)

x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction ...")

# num_threads > 1 will predict very slow in kernal

clf.reset_parameter({"num_threads":1})

p_test = clf.predict(x_test)

del x_test; gc.collect()

print("Start write result ...")

sub = pd.read_csv('../input/sample_submission.csv')

for c in sub.columns[sub.columns != 'ParcelId']:

    sub[c] = p_test

sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')