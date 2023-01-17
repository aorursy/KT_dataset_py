# System pacakges

import os

import sys

import re

import gc

import time

import datetime

import warnings



# Data pacakges

import pickle

import numpy as np

import pandas as pd

from itertools import product

import seaborn as sns

import matplotlib.pyplot as plt



# Modelling pacakges

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import plot_importance



# Pacakge settings

%matplotlib inline

sns.set(style="darkgrid")

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

sys.version_info
print (os.listdir('../input'))
train = pd.read_csv('../input/sales_train.csv.gz')

items = pd.read_csv('../input/items.csv')

# Drop name to save space

items = items.drop('item_name', axis=1)

# Set index to ID to avoid droping it later

test  = pd.read_csv('../input/test.csv.gz').set_index('ID')

# Add date_block_num = 34

test['date_block_num'] = 34

#shops = pd.read_csv('../input/shops.csv')

#categories = pd.read_csv('../input/item_categories.csv')

# Get the ranges of each feature to select the most appropriate data size

print ('-------------------------------------------------------')

print ('train:')

for f in train.columns.values:

    print ('%s: %s ~ %s' %(f, train[f].min(), train[f].max()))

print ('-------------------------------------------------------')

print ('items:')

for f in items.columns.values:

    print ('%s: %s ~ %s' %(f, items[f].min(), items[f].max()))

print ('-------------------------------------------------------')

print ('test:')

for f in test.columns.values:

    print ('%s: %s ~ %s' %(f, test[f].min(), test[f].max()))
def compress_columns(df,columns,keyword,search_type,datatype):

    if search_type=='in':

        valid_features = [x for x in columns if keyword in x]

    elif search_type=='start':

        valid_features = [x for x in columns if x.startswith(keyword)]

    if len(valid_features):

        for f in valid_features:

            df[f] = df[f].round().astype(datatype)

    return df



def data_compression(df):

    features = df.columns.values

    # Original features

    if 'date_block_num' in features:

        df['date_block_num'] = df['date_block_num'].astype(np.int8)

    if 'shop_id' in features:

        df['shop_id'] = df['shop_id'].astype(np.int8)

    if 'item_category_id' in features:

        df['item_category_id'] = df['item_category_id'].astype(np.int8)

    if 'item_id' in features:

        df['item_id'] = df['item_id'].astype(np.int16)

    if 'item_price' in features:

        df['item_price'] = df['item_price'].astype(np.float32)

    if 'item_id_avg_item_price' in features:

        df['item_id_avg_item_price'] = df['item_id_avg_item_price'].astype(np.float32)

        

    # Mean encoded features & lag features

    df = compress_columns(df,features,'item_id_sum_item_cnt_day','in',np.int16)

    df = compress_columns(df,features,'item_id_avg_item_cnt_day','in',np.float16)

    

    df = compress_columns(df,features,'shop_id_avg_item_price','in',np.float16)

    df = compress_columns(df,features,'shop_id_sum_item_cnt_day','in',np.int16)

    df = compress_columns(df,features,'shop_id_avg_item_cnt_day','in',np.float16)

    

    df = compress_columns(df,features,'item_category_id_avg_item_price','in',np.float16)

    df = compress_columns(df,features,'item_category_id_sum_item_cnt_day','in',np.int32)

    df = compress_columns(df,features,'item_category_id_avg_item_cnt_day','in',np.float16)

    

    df = compress_columns(df,features,'item_cnt_day','start',np.int16)

    return df
# Compress features according to range

train = data_compression(train)

items = data_compression(items)

test = data_compression(test)
# Include Category id

train = pd.merge(train,items,on='item_id',how='left')

test = pd.merge(test,items, on='item_id', how='left')
# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
train.info()
def box_plot(df,f):

    plt.figure(figsize=(10,4))

    plt.title(f+' distribution')

    x_min = int(df[f].min() - (abs(df[f].min())*0.1))

    x_max = int(df[f].max() + (abs(df[f].max())*0.1))

    if x_min==0:

        x_min = -1

    if x_max==0:

        x_max = 1

    plt.xlim(x_min,x_max)

    sns.boxplot(x=df[f])



plot_features = [x for x in train.columns.values if train[x].dtype != 'object']

for f in plot_features:

    box_plot(train,f)
# Getting rid of the outliers & negative values

train = train[(train['item_price']<100000) & (train['item_price']>=0)]

train = train[(train['item_cnt_day']<1000) & (train['item_cnt_day']>=0)]



# distribution after outliers removal

plot_features = ['item_price','item_cnt_day']

for f in plot_features:

    box_plot(train,f)
# Create a grid with columns

index_cols = ['shop_id','item_id','date_block_num']



# For every month we create a grid for all shops/items pair

grid = []

for block_num in train['date_block_num'].unique():

    cur_shops = train.loc[train['date_block_num']==block_num,'shop_id'].unique()

    cur_items = train.loc[train['date_block_num']==block_num,'item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops,cur_items,[block_num]])),dtype='int32'))

grid = pd.DataFrame(np.vstack(grid),columns=index_cols,dtype=np.int32)

grid = data_compression(grid)

grid.head()
grid.info()
# Group items per month, per shop, per item, sum the sales of the item, mean the price

# There is a big difference between np.mean and pandas mean

train_m = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':'sum','item_price':np.mean}).reset_index()

train_m = pd.merge(grid,train_m,on=['date_block_num','shop_id','item_id'],how='left').fillna(0)

train_m = pd.merge(train_m,items,on='item_id',how='left')

train_m = data_compression(train_m)
# Making the mean encoded features

for type_id in ['item_id', 'shop_id', 'item_category_id']:

    for column_id, aggregator, aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:

        mean_df = train.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]

        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id,type_id,'date_block_num']

        train_m = pd.merge(train_m, mean_df, on=['date_block_num',type_id], how='left')

        del mean_df

        gc.collect()
del train

gc.collect()
for f in train_m.columns:

    if 'item_cnt' in f:

        train_m[f] = train_m[f].fillna(0)

    elif 'item_price' in f:

        train_m[f] = train_m[f].fillna(train_m[f].median())



# Compress data

train_m = data_compression(train_m)

train_m.info(verbose=False)
# Check the positions of the base lag features

train_m.columns.values[6:]
# Get all the monthly features, which means the Mean Encoded fatures are all monthly based

lag_features = list(train_m.columns[6:])+['item_cnt_day']

# The selected months from current month

lags = [1,2,3,6]
for lag in lags:

    train_new_df = train_m.copy()

    # Get the current month

    train_new_df['date_block_num'] += lag

    train_new_df = train_new_df[['date_block_num','shop_id','item_id']+lag_features]

    # Name the columns as lag features of the month

    train_new_df.columns = ['date_block_num','shop_id','item_id'] + [x+'_lag_'+str(lag) for x in lag_features]

    train_m = pd.merge(train_m,train_new_df,on=['date_block_num','shop_id','item_id'],how='left')

    del train_new_df

    gc.collect()

    print ('lag %s processed' %lag)
# Fill NaNs

for f in train_m.columns:

    if 'item_cnt' in f:

        train_m[f] = train_m[f].fillna(0)

    elif 'item_price' in f:

        train_m[f] = train_m[f].fillna(train_m[f].median())



train_m = data_compression(train_m)

train_m.info(verbose=False)
# Set the maximum clip value

max_clip = 30

train_m['item_cnt_day'] = train_m['item_cnt_day'].clip(0,max_clip).astype(np.float16)
# Add lag variables

for lag in lags:

    train_new_df = train_m.copy()

    # Get the current month

    train_new_df['date_block_num'] += lag

    train_new_df = train_new_df[['date_block_num','shop_id','item_id']+lag_features]

    # Name the columns as lag features of the month

    train_new_df.columns = ['date_block_num','shop_id','item_id'] + [x+'_lag_'+str(lag) for x in lag_features]

    test = pd.merge(test,train_new_df,on=['date_block_num','shop_id','item_id'],how='left')

    del train_new_df

    gc.collect()

    print ('lag %s processed' %lag)
# Fill NaNs

for f in test.columns:

    if 'item_cnt' in f:

        test[f] = test[f].fillna(0)

    elif 'item_price' in f:

        test[f] = test[f].fillna(test[f].median())



test = data_compression(test)
cols_to_drop = lag_features[:-1] + ['item_price']

print ('Columns to drop')

print (cols_to_drop)
train_cols = train_m.columns.values

test_cols = test.columns.values

for c in cols_to_drop:

    if c in train_cols:

        train_m = train_m.drop(c,axis=1)

    if c in test_cols:

        test = test.drop(c,axis=1)
# Month number

train_m['month'] = train_m['date_block_num']%12

train_m['month'] = train_m['month'].astype(np.int8)

# Number of days in a month, no leap years here

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

train_m['days'] = train_m['month'].map(days).astype(np.int8)



test['month'] = 11

test['month'] = test['month'].astype(np.int8)

test['days'] = 30

test['days'] = test['days'].astype(np.int8)
# Assert all the columns are the same except target column

set(train_m.columns.values) ^ set(test.columns.values)
train_m.head()
test.head()
test[['shop_id','item_id']+['item_cnt_day_lag_'+str(x) for x in [1,2,3]]].head()
print(train_m[train_m['shop_id'] == 5][train_m['item_id'] == 5037][train_m['date_block_num'] == 33]['item_cnt_day'])

print(train_m[train_m['shop_id'] == 5][train_m['item_id'] == 5037][train_m['date_block_num'] == 32]['item_cnt_day'])

print(train_m[train_m['shop_id'] == 5][train_m['item_id'] == 5037][train_m['date_block_num'] == 31]['item_cnt_day'])
train_m = train_m[train_m['date_block_num']>12]
train_set = train_m[train_m['date_block_num']<33]

val_set = train_m[train_m['date_block_num']==33]
print (train_set.shape)

print (val_set.shape)

print (test.shape)
# Save data

train_set.to_pickle('train.pkl')

val_set.to_pickle('val.pkl')

test.to_pickle('test.pkl')
del train_m

gc.collect()
# divide data into x & y

train_x = train_set.drop(['item_cnt_day'],axis=1)

train_y = train_set['item_cnt_day']

val_x = val_set.drop(['item_cnt_day'],axis=1)

val_y = val_set['item_cnt_day']



features = list(train_x.columns.values)
# Check if the data sets have equal amount of features

print (train_x.shape)

print (train_y.shape)

print (val_x.shape)

print (val_y.shape)

print (test.shape)
del train_set

del val_set

gc.collect()
# For saving data & output results / models

def post_processing(model,model_name,train_x,val_x,test_x,train_y,val_y,test):

    # Here we once again clip the output to 0~20

    train_pred = model.predict(train_x).clip(0, 20)

    val_pred = model.predict(val_x).clip(0, 20)

    test_pred = model.predict(test_x).clip(0, 20)



    # Get rmse scores

    train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))

    print("Train RMSE: %f" % (train_rmse))

    val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))

    print("Val RMSE: %f" % (val_rmse))

    

    # Export submission

    submission = pd.DataFrame({'ID':test.index,'item_cnt_month': test_pred})

    submission.to_csv('%s_submission.csv'%model_name,index=False)



    # save model to file

    pickle.dump(lm, open("%s_model.pickle" %model_name, "wb"))

    return train_pred,val_pred,test_pred
# For plotting feature importance

def plot_feature_importances(importances,indices,features,title,dimensions):

    plt.figure(figsize=dimensions)

    plt.title(title)

    plt.barh(range(len(indices)), importances[indices], color='b', align='center')

    plt.yticks(range(len(indices)), [features[i] for i in indices])

    plt.xlabel('Relative Importance')

    plt.show()   
# Normalise data

scaler = preprocessing.StandardScaler()

scaler.fit(train_x.values)

train_x_norm = scaler.transform(train_x.values)

val_x_norm = scaler.transform(val_x.values)

test_norm = scaler.transform(test.values)
# Training

gc.collect()

ts = time.time()

# Training

lm = linear_model.Ridge()

lm.fit(train_x_norm,train_y)

print ('Training time: %s' %(time.time() - ts))
# Performance and test predictions

train_pred1,val_pred1,test_pred1 = post_processing(lm,'ridge',train_x_norm,val_x_norm,test_norm,train_y,val_y,test)
# Feature Importance

importances = abs(lm.coef_)

indices = np.argsort(importances)

title = 'Linear Regression Feature Importances'

plot_feature_importances(importances,indices,features,title,(8,16))
del train_x_norm

del val_x_norm

del test_norm

gc.collect()
# Training

gc.collect()

ts = time.time()

xgbtrain = xgb.DMatrix(train_x.values, train_y.values)



param = {'max_depth':8, 

         'subsample':1,

         'min_child_weight':0.5,

         'eta':0.3, 

         'num_round':1000, 

         'seed':1,

         'verbosity':2,

         'eval_metric':'rmse'} # random parameters



bst = xgb.train(param, xgbtrain)

print ('Training time: %s' %(time.time() - ts))
# Performance and test predictions

train_pred2,val_pred2,test_pred2 = post_processing(bst,'xgboost',xgb.DMatrix(train_x.values),xgb.DMatrix(val_x.values),xgb.DMatrix(test.values),train_y,val_y,test)
# Feature Importance

import operator

importance = sorted(bst.get_score().items(), key=operator.itemgetter(1))

importance_v = np.asarray([x[1] for x in importance],dtype=np.int16)

indices = np.asarray([int(x[0].replace('f','')) for x in importance],dtype=np.int8)

title = 'xgboost Feature Importances'



plt.figure(figsize=(8,16))

plt.title(title)

plt.barh(range(len(indices)), importance_v, color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()   
# Training

gc.collect()

ts = time.time()

rf = RandomForestRegressor(

    bootstrap=True,

    max_depth=30,

    max_features=3,

    min_samples_leaf=5,

    min_samples_split=12,

    n_estimators=200,

    random_state=42,

    verbose=1,

    n_jobs=-1

)

rf.fit(train_x.values,train_y.values)

print ('Training time: %s' %(time.time() - ts))
# Performance and test predictions

train_pred3,val_pred3,test_pred3 = post_processing(rf,'rf',train_x,val_x,test,train_y,val_y,test)
# Feature Importance

importances = rf.feature_importances_

indices = np.argsort(importances)

title = 'Random Forest Feature Importances'

plot_feature_importances(importances,indices,features,title,(8,16))
# Making data

train_stack = pd.DataFrame({'lm':train_pred1,'xgboost':train_pred2,'rf':train_pred3})

val_stack = pd.DataFrame({'lm':val_pred1,'xgboost':val_pred2,'rf':val_pred3})

test_stack = pd.DataFrame({'lm':test_pred1,'xgboost':test_pred2,'rf':test_pred3})
# Normalise data

scaler = preprocessing.StandardScaler()

scaler.fit(train_stack.values)

train_stack = scaler.transform(train_stack.values)

val_stack = scaler.transform(val_stack.values)

test_stack = scaler.transform(test_stack.values)
# Training

gc.collect()

ts = time.time()

# Training

stack_lm = linear_model.Ridge()

stack_lm.fit(train_stack,train_y)

print ('Training time: %s' %(time.time() - ts))
# Performance and test predictions

post_processing(stack_lm,'stack',train_stack,val_stack,test_stack,train_y,val_y,test)
# Feature Importance

importances = abs(stack_lm.coef_)

indices = np.argsort(importances)

title = 'Stack Model Feature Importances'

plot_feature_importances(importances,indices,features,title,(8,6))