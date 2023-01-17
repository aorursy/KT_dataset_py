## In this kernel I would like to show: 

## 1. FE creation approaches

## 2. Sequential fe validation

## 3. Dimension reduction

## 4. FE validation by Permutation importance

## 5. Mean encodings

## 6. Parallelization for FE
import numpy as np 

import pandas as pd 

import os, sys, gc, warnings, psutil, random



warnings.filterwarnings('ignore')
########################### Load data

########################### Basic features were created here:

########################### https://www.kaggle.com/kyakovlev/m5-simple-fe

#################################################################################



# Read data

grid_df = pd.concat([pd.read_pickle('../input/featureextraction/grid_part_1.pkl'),

                     pd.read_pickle('../input/featureextraction/grid_part_2.pkl').iloc[:,2:],

                     pd.read_pickle('../input/featureextraction/grid_part_3.pkl').iloc[:,2:]],

                     axis=1)



# Subsampling

# to make all calculations faster.

# Keep only 5% of original ids.

keep_id = np.array_split(list(grid_df['id'].unique()), 20)[0]

grid_df = grid_df[grid_df['id'].isin(keep_id)].reset_index(drop=True)



# Let's "inspect" our grid DataFrame

grid_df.info()
########################### Baseline model

#################################################################################



# We will need some global VARS for future



SEED = 42             # Our random seed for everything

random.seed(SEED)     # to make all tests "deterministic"

np.random.seed(SEED)

N_CORES = psutil.cpu_count()     # Available CPU cores



TARGET = 'sales'      # Our Target

END_TRAIN = 1941      # And we will use last 28 days as validation



# Drop some items from "TEST" set part (1914...)

grid_df = grid_df[grid_df['d']<=END_TRAIN].reset_index(drop=True)



# Features that we want to exclude from training

remove_features = ['id','d',TARGET]



# Our baseline model serves

# to do fast checks of

# new features performance 



# We will use LightGBM for our tests

import lightgbm as lgb

lgb_params = {

                    'boosting_type': 'gbdt',         # Standart boosting type

                    'objective': 'regression',       # Standart loss for RMSE

                    'metric': ['rmse'],              # as we will use rmse as metric "proxy"

                    'subsample': 0.8,                

                    'subsample_freq': 1,

                    'learning_rate': 0.05,           # 0.5 is "fast enough" for us

                    'num_leaves': 2**7-1,            # We will need model only for fast check

                    'min_data_in_leaf': 2**8-1,      # So we want it to train faster even with drop in generalization 

                    'feature_fraction': 0.8,

                    'n_estimators': 5000,            # We don't want to limit training (you can change 5000 to any big enough number)

                    'early_stopping_rounds': 30,     # We will stop training almost immediately (if it stops improving) 

                    'seed': SEED,

                    'verbose': -1,

                } 



## RMSE

def rmse(y, y_pred):

    return np.sqrt(np.mean(np.square(y - y_pred)))



# Small function to make fast features tests

# estimator = make_fast_test(grid_df)

# it will return lgb booster for future analisys

def make_fast_test(df):



    features_columns = [col for col in list(df) if col not in remove_features]



    tr_x, tr_y = df[df['d']<=(END_TRAIN-28)][features_columns], df[df['d']<=(END_TRAIN-28)][TARGET]              

    vl_x, v_y = df[df['d']>(END_TRAIN-28)][features_columns], df[df['d']>(END_TRAIN-28)][TARGET]

    

    train_data = lgb.Dataset(tr_x, label=tr_y)

    valid_data = lgb.Dataset(vl_x, label=v_y)

    

    estimator = lgb.train(

                            lgb_params,

                            train_data,

                            valid_sets = [train_data,valid_data],

                            verbose_eval = 500,

                        )

    

    return estimator



# Make baseline model

baseline_model = make_fast_test(grid_df)
########################### Lets test our normal Lags (7 days)

########################### Some more info about lags here:

########################### https://www.kaggle.com/kyakovlev/m5-lags-features

#################################################################################



# Small helper to make lags creation faster

from multiprocessing import Pool                # Multiprocess Runs



## Multiprocessing Run.

# :t_split - int of lags days                   # type: int

# :func - Function to apply on each split       # type: python function

# This function is NOT 'bulletproof', be carefull and pass only correct types of variables.

## Multiprocess Runs

def df_parallelize_run(func, t_split):

    num_cores = np.min([N_CORES,len(t_split)])

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, t_split), axis=1)

    pool.close()

    pool.join()

    return df



def make_normal_lag(lag_day):

    lag_df = grid_df[['id','d',TARGET]] # not good to use df from "global space"

    col_name = 'sales_lag_'+str(lag_day)

    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(lag_day)).astype(np.float16)

    return lag_df[[col_name]]



# Launch parallel lag creation

# and "append" to our grid

LAGS_SPLIT = [col for col in range(1,1+7)]

grid_df = pd.concat([grid_df, df_parallelize_run(make_normal_lag,LAGS_SPLIT)], axis=1)



# Make features test

test_model = make_fast_test(grid_df)
########################### Permutation importance Test

########################### https://www.kaggle.com/dansbecker/permutation-importance @dansbecker

#################################################################################



# Let's creat validation dataset and features

features_columns = [col for col in list(grid_df) if col not in remove_features]

validation_df = grid_df[grid_df['d']>(END_TRAIN-28)].reset_index(drop=True)



# Make normal prediction with our model and save score

validation_df['preds'] = test_model.predict(validation_df[features_columns])

base_score = rmse(validation_df[TARGET], validation_df['preds'])

print('Standart RMSE', base_score)





# Now we are looping over all our numerical features

for col in features_columns:

    

    # We will make validation set copy to restore

    # features states on each run

    temp_df = validation_df.copy()

    

    # Error here appears if we have "categorical" features and can't 

    # do np.random.permutation without disrupt categories

    # so we need to check if feature is numerical

    if temp_df[col].dtypes.name != 'category':

        temp_df[col] = np.random.permutation(temp_df[col].values)

        temp_df['preds'] = test_model.predict(temp_df[features_columns])

        cur_score = rmse(temp_df[TARGET], temp_df['preds'])

        

        # If our current rmse score is less than base score

        # it means that feature most probably is a bad one

        # and our model is learning on noise

        print(col, np.round(cur_score - base_score, 4))



# Remove Temp data

del temp_df, validation_df



# Remove test features

# As we will compare performance with baseline model for now

keep_cols = [col for col in list(grid_df) if 'sales_lag_' not in col]

grid_df = grid_df[keep_cols]





# Results:

## Lags with 1 days shift (nearest past) are important

## Some other features are not important and probably just noise

## Better make several Permutation runs to confirm useless of the feature

## link again https://www.kaggle.com/dansbecker/permutation-importance @dansbecker



## price_nunique -0.002 : strong negative values are most probably noise

## price_max -0.0002 : values close to 0 need deeper investigation
########################### Lets test far away Lags (7 days with 56 days shift)

########################### and check permutation importance

#################################################################################



LAGS_SPLIT = [col for col in range(56,56+7)]

grid_df = pd.concat([grid_df, df_parallelize_run(make_normal_lag,LAGS_SPLIT)], axis=1)

test_model = make_fast_test(grid_df)



features_columns = [col for col in list(grid_df) if col not in remove_features]

validation_df = grid_df[grid_df['d']>(END_TRAIN-28)].reset_index(drop=True)

validation_df['preds'] = test_model.predict(validation_df[features_columns])

base_score = rmse(validation_df[TARGET], validation_df['preds'])

print('Standart RMSE', base_score)



for col in features_columns:

    temp_df = validation_df.copy()

    if temp_df[col].dtypes.name != 'category':

        temp_df[col] = np.random.permutation(temp_df[col].values)

        temp_df['preds'] = test_model.predict(temp_df[features_columns])

        cur_score = rmse(temp_df[TARGET], temp_df['preds'])

        print(col, np.round(cur_score - base_score, 4))



del temp_df, validation_df

        

# Remove test features

# As we will compare performance with baseline model for now

keep_cols = [col for col in list(grid_df) if 'sales_lag_' not in col]

grid_df = grid_df[keep_cols]





# Results:

## Lags with 56 days shift (far away past) are not as important

## as nearest past lags

## and at some point will be just noise for our model
########################### PCA

#################################################################################



# The main question here - can we have 

# almost same rmse boost with less features

# less dimensionality?



# Lets try PCA and make 7->3 dimensionality reduction



# PCA is "unsupervised" learning

# and with shifted target we can be sure

# that we have no Target leakage

from sklearn.decomposition import PCA



def make_pca(df, pca_col, n_days):

    print('PCA:', pca_col, n_days)

    

    # We don't need any other columns to make pca

    pca_df = df[[pca_col,'d',TARGET]]

    

    # If we are doing pca for other series "levels" 

    # we need to agg first

    if pca_col != 'id':

        merge_base = pca_df[[pca_col,'d']]

        pca_df = pca_df.groupby([pca_col,'d'])[TARGET].agg(['sum']).reset_index()

        pca_df[TARGET] = pca_df['sum']

        del pca_df['sum']

    

    # Min/Max scaling

    pca_df[TARGET] = pca_df[TARGET]/pca_df[TARGET].max()

    

    # Making "lag" in old way (not parallel)

    LAG_DAYS = [col for col in range(1,n_days+1)]

    format_s = '{}_pca_'+pca_col+str(n_days)+'_{}'

    pca_df = pca_df.assign(**{

            format_s.format(col, l): pca_df.groupby([pca_col])[col].transform(lambda x: x.shift(l))

            for l in LAG_DAYS

            for col in [TARGET]

        })

    

    pca_columns = list(pca_df)[3:]

    pca_df[pca_columns] = pca_df[pca_columns].fillna(0)

    pca = PCA(random_state=SEED)

    

    # You can use fit_transform here

    pca.fit(pca_df[pca_columns])

    pca_df[pca_columns] = pca.transform(pca_df[pca_columns])

    

    print(pca.explained_variance_ratio_)

    

    # we will keep only 3 most "valuable" columns/dimensions 

    keep_cols = pca_columns[:3]

    print('Columns to keep:', keep_cols)

    

    # If we are doing pca for other series "levels"

    # we need merge back our results to merge_base df

    # and only than return resulted df

    # I'll skip that step here

    

    return pca_df[keep_cols]





# Make PCA

grid_df = pd.concat([grid_df, make_pca(grid_df,'id',7)], axis=1)



# Make features test

test_model = make_fast_test(grid_df)



# Remove test features

# As we will compare performance with baseline model for now

keep_cols = [col for col in list(grid_df) if '_pca_' not in col]

grid_df = grid_df[keep_cols]

########################### Mean/std target encoding

#################################################################################



# We will use these three columns for test

# (in combination with store_id)

icols = ['item_id','cat_id','dept_id']



# But we can use any other column or even multiple groups

# like these ones

#            'state_id',

#            'store_id',

#            'cat_id',

#            'dept_id',

#            ['state_id', 'cat_id'],

#            ['state_id', 'dept_id'],

#            ['store_id', 'cat_id'],

#            ['store_id', 'dept_id'],

#            'item_id',

#            ['item_id', 'state_id'],

#            ['item_id', 'store_id']



# There are several ways to do "mean" encoding

## K-fold scheme

## LOO (leave one out)

## Smoothed/regularized 

## Expanding mean

## etc 



# You can test as many options as you want

# and decide what to use

# Because of memory issues you can't 

# use many features.



# We will use simple target encoding

# by std and mean agg

for col in icols:

    print('Encoding', col)

    temp_df = grid_df[grid_df['d']<=(1941-28)] # to be sure we don't have leakage in our validation set

    

    temp_df = temp_df.groupby([col,'store_id']).agg({TARGET: ['std','mean']})

    joiner = '_'+col+'_encoding_'

    temp_df.columns = [joiner.join(col).strip() for col in temp_df.columns.values]

    temp_df = temp_df.reset_index()

    grid_df = grid_df.merge(temp_df, on=[col,'store_id'], how='left')

    del temp_df



# Make features test

test_model = make_fast_test(grid_df)



# Remove test features

keep_cols = [col for col in list(grid_df) if '_encoding_' not in col]

grid_df = grid_df[keep_cols]



# Bad thing that for some items  

# we are using past and future values.

# But we are looking for "categorical" similiarity

# on a "long run". So future here is not a big problem.
########################### Apply on grid_df

#################################################################################

# lets read grid from 

# https://www.kaggle.com/kyakovlev/m5-simple-fe

# to be sure that our grids are aligned by index

grid_df = pd.read_pickle('../input/featureextraction/grid_part_1.pkl')

grid_df[TARGET][grid_df['d']>(1941-28)] = np.nan

base_cols = list(grid_df)



icols =  [

            ['state_id'],

            ['store_id'],

            ['cat_id'],

            ['dept_id'],

            ['state_id', 'cat_id'],

            ['state_id', 'dept_id'],

            ['store_id', 'cat_id'],

            ['store_id', 'dept_id'],

            ['item_id'],

            ['item_id', 'state_id'],

            ['item_id', 'store_id']

            ]



for col in icols:

    print('Encoding', col)

    col_name = '_'+'_'.join(col)+'_'

    grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float16)

    grid_df['enc'+col_name+'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float16)



keep_cols = [col for col in list(grid_df) if col not in base_cols]

grid_df = grid_df[['id','d']+keep_cols]
########################### Apply on grid_df

#################################################################################

# lets read grid from 

# https://www.kaggle.com/kyakovlev/m5-simple-fe

# to be sure that our grids are aligned by index

grid_df = pd.read_pickle('../input/featureextraction/grid_part_1.pkl')

grid_df[TARGET][grid_df['d']>(1941-28)] = np.nan

base_cols = list(grid_df)



icols =  [

            ['state_id'],

            ['store_id'],

            ['cat_id'],

            ['dept_id'],

            ['state_id', 'cat_id'],

            ['state_id', 'dept_id'],

            ['store_id', 'cat_id'],

            ['store_id', 'dept_id'],

            ['item_id'],

            ['item_id', 'state_id'],

            ['item_id', 'store_id']

            ]



for col in icols:

    print('Encoding', col)

    col_name = '_'+'_'.join(col)+'_'

    grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float16)

    grid_df['enc'+col_name+'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float16)



keep_cols = [col for col in list(grid_df) if col not in base_cols]

grid_df = grid_df[['id','d']+keep_cols]
#################################################################################

print('Save Mean/Std encoding')

grid_df.to_pickle('mean_encoding_df.pkl')
########################### Final list of new features

#################################################################################

grid_df.info()