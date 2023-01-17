# General imports

import numpy as np

import pandas as pd

import os, sys, gc, time, warnings, pickle, psutil, random



import time



warnings.filterwarnings('ignore')
## Simple "Memory profilers" to see memory usage

def get_memory_usage():

    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

        

def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)
########################### Vars

#################################################################################

TARGET = 'sales'         # Our main target

END_TRAIN = 1913         # Last day in train set

MAIN_INDEX = ['id','d']  # We can identify item by these columns
########################### Load Data

#################################################################################

print('Load Main Data')



# We will need only train dataset

# to show lags concept

train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')



# To make all calculations faster

# we will limit dataset by 'CA' state

train_df = train_df[train_df['state_id']=='CA']
########################### Data Representation

#################################################################################



# Let's check our shape

print('Shape', train_df.shape)
## Horizontal representation



# If we feed directly this data to model

# our label will be values in column 'd_1913'

# all other columns will be our "features"



# In lag terminology all d_1->d_1912 columns

# are our lag features 

# (target values in previous time period)



# Good thing that we have a lot of features here

# Bad thing is that we have just 12196 "train rows"

# Note: here and after all numbers are limited to 'CA' state

train_df.iloc[:10]
## Vertical representation



# In other hand we can think of d_ columns

# as additional labels and can significantly 

# scale up our training set to 23330948 rows



# Good thing that our model will have 

# greater input for training

# Bad thing that we are losing lags that we had

# in horizontal representation and

# also new data set consumes much more memory



index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

train_df = pd.melt(train_df, 

                  id_vars = index_columns, 

                  var_name = 'd', 

                  value_name = TARGET)



train_df[train_df['id']=='HOBBIES_1_001_CA_1_validation'].iloc[:10]
## Some minification

train_df['d'] = train_df['d'].apply(lambda x: x[2:]).astype(np.int16)



icols = ['id','item_id','dept_id','cat_id','store_id','state_id']

for col in icols:

    train_df[col] = train_df[col].astype('category')
########################### Lags creation

#################################################################################



# We have several "code" solutions here

# As our dataset is allready sorted by d values

# we can simply shift() values

# also we have to keep in mind that 

# we need to aggregate values on 'id' level



# group and shift in loop

temp_df = train_df[['id','d',TARGET]]



start_time = time.time()

for i in range(1,8):

    print('Shifting:', i)

    temp_df['lag_'+str(i)] = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(i))

    

print('%0.2f min: Time for loops' % ((time.time() - start_time) / 60))





# Or same in "compact" manner

LAG_DAYS = [col for col in range(1,8)]

temp_df = train_df[['id','d',TARGET]]



start_time = time.time()

temp_df = temp_df.assign(**{

        '{}_lag_{}'.format(col, l): temp_df.groupby(['id'])[col].transform(lambda x: x.shift(l))

        for l in LAG_DAYS

        for col in [TARGET]

    })



print('%0.2f min: Time for bulk shift' % ((time.time() - start_time) / 60))
# The result

temp_df[temp_df['id']=='HOBBIES_1_001_CA_1_validation'].iloc[:10]



# You can notice many NaNs values - it's normal

# because there is no data for day 0,-1,-2

# (out of dataset time periods)



# Same works for test set

# be careful to make lag features:

# for day 1920 there is no data about day 1919 (until 1913)

# So if you want to predict day 1915 your 

# lag features have to start from 2 

# (1915(predicting day) - 1913(last day with label in dataset))

# and so on.



# There are few options to work 

# with NaNs in train set

## 1. drop it train_df[train_df['d']>MAX_LAG_DAY] 

## 1.1 in our case we already dropped some lines by release date

##     so you have find d.min() for each id

##     and drop train_df[train_df['d']>(train_df['d_min']+MAX_LAG_DAY)] 

## 2. If you want to keep it you can 

##    fill with '-1' to be able to convert to int

## 3. Leave as it is

## 4. Fill with mean -> not recommended
########################### Rolling lags

#################################################################################



# We restored some day sales values from horizontal representation

# as lag features but just few of them (last 7 days or less)

# because of memory limits we can't have many lag features

# How we can get additional information from other days?



## Rolling aggragations



temp_df = train_df[['id','d','sales']]



start_time = time.time()



for i in [14,30,60]:

    print('Rolling period:', i)

    temp_df['rolling_mean_'+str(i)] = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).mean())

    temp_df['rolling_std_'+str(i)]  = temp_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).std())



# lambda x: x.shift(1)

# 1 day shift will serve only to predict day 1914

# for other days you have to shift PREDICT_DAY-1913



# Such aggregations will help us to restore

# at least part of the information for our model

# and out of 14+30+60->104 columns we can have just 6

# with valuable information (hope it is sufficient)

# you can also aggregate by max/skew/median etc 

# also you can try other rolling periods 180,365 etc

print('%0.2f min: Time for loop' % ((time.time() - start_time) / 60))

# The result

temp_df[temp_df['id']=='HOBBIES_1_002_CA_1_validation'].iloc[:20]



# Same for NaNs values - it's normal

# because there is no data for 

# 0*(rolling_period),-1*(rolling_period),-2*(rolling_period)
########################### Memory ussage

#################################################################################

# Let's check our memory usage

print("{:>20}: {:>8}".format('Original rolling df',sizeof_fmt(temp_df.memory_usage(index=True).sum())))



# can we minify it?

# 1. if our dataset are aligned by index 

#    you don't need 'id' 'd' 'sales' columns

temp_df = temp_df.iloc[:,3:]

print("{:>20}: {:>8}".format('Values rolling df',sizeof_fmt(temp_df.memory_usage(index=True).sum())))



# can we make it even smaller?

# carefully change dtype and/or

# use sparce matrix to minify 0s

# Also note that lgbm accepts matrixes as input

# that is good for memory reducion 

from scipy import sparse 

temp_matrix = sparse.csr_matrix(temp_df)



# restore to df

temp_matrix_restored = pd.DataFrame(temp_matrix.todense())

restored_cols = ['roll_' + str(i) for i in list(temp_matrix_restored)]

temp_matrix_restored.columns = restored_cols
########################### Remove old objects

#################################################################################

del temp_df, train_df, temp_matrix, temp_matrix_restored
########################### Apply on grid_df

#################################################################################

# lets read grid from 

# https://www.kaggle.com/kyakovlev/m5-simple-fe

# to be sure that our grids are aligned by index

grid_df = pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')



# We need only 'id','d','sales'

# to make lags and rollings

grid_df = grid_df[['id','d','sales']]

SHIFT_DAY = 28



# Lags

# with 28 day shift

start_time = time.time()

print('Create lags')



LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+15)]

grid_df = grid_df.assign(**{

        '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))

        for l in LAG_DAYS

        for col in [TARGET]

    })



# Minify lag columns

for col in list(grid_df):

    if 'lag' in col:

        grid_df[col] = grid_df[col].astype(np.float16)



print('%0.2f min: Lags' % ((time.time() - start_time) / 60))



# Rollings

# with 28 day shift

start_time = time.time()

print('Create rolling aggs')



for i in [7,14,30,60,180]:

    print('Rolling period:', i)

    grid_df['rolling_mean_'+str(i)] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)

    grid_df['rolling_std_'+str(i)]  = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)



# Rollings

# with sliding shift

for d_shift in [1,7,14]: 

    print('Shifting period:', d_shift)

    for d_window in [7,14,30,60]:

        col_name = 'rolling_mean_tmp_'+str(d_shift)+'_'+str(d_window)

        grid_df[col_name] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)

    

    

print('%0.2f min: Lags' % ((time.time() - start_time) / 60))
########################### Export

#################################################################################

print('Save lags and rollings')

grid_df.to_pickle('lags_df_'+str(SHIFT_DAY)+'.pkl')
########################### Final list of new features

#################################################################################

grid_df.info()