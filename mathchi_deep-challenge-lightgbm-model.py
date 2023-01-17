# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')





import os, gc, sys, warnings, random, math, psutil, pickle



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
##### Helpers ######





## Seeder

# :seed to make all processes deterministic     # type: int

def seed_everything(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    

## Simple "Memory profilers" to see memory usage

def get_memory_usage():

    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

        

def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)



####### Vars ######



SEED = 42

LOCAl_TEST = False

seed_everything(SEED)

TARGET = 'meter_reading'
train_df = pd.read_pickle('/kaggle/input/data-minification-ashrae/train.pkl')

test_df = pd.read_pickle('/kaggle/input/data-minification-ashrae/test.pkl')



building_df = pd.read_pickle('/kaggle/input/data-minification-ashrae/building_metadata_metadata.pkl')



train_weather_df = pd.read_pickle('/kaggle/input/data-minification-ashrae/weather_train.pkl')

test_weather_df = pd.read_pickle('/kaggle/input/data-minification-ashrae/weather_test.pkl')
################ Building DF merge through concat ########################

# Benefits of concat:

## Faster for huge datasets (columns number)

## No dtype change for dataset

## Consume less memmory 



temp_df = train_df[['building_id']]

temp_df = temp_df.merge(building_df, on=['building_id'], how='left')

del temp_df['building_id']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['building_id']]

temp_df = temp_df.merge(building_df, on=['building_id'], how='left')

del temp_df['building_id']

test_df = pd.concat([test_df, temp_df], axis=1)



del building_df, temp_df
################ Weather DF merge over concat (to not lose type) ###########################

# Benefits of concat:

## Faster for huge datasets (columns number)

## No dtype change for dataset

## Consume less memmory 



temp_df = train_df[['site_id','timestamp']]

temp_df = temp_df.merge(train_weather_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['site_id','timestamp']]

temp_df = temp_df.merge(test_weather_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

test_df = pd.concat([test_df, temp_df], axis=1)



del train_weather_df, test_weather_df, temp_df
############# Trick to use kernel hdd to store results #############



# You can save just test_df or both if have sufficient space

train_df.to_pickle('train_df.pkl')

test_df.to_pickle('test_df.pkl')

   

del train_df, test_df

gc.collect()
########################### Check memory usage #################################



for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),

                         key= lambda x: -x[1])[:10]:

    print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))

print('Memory in Gb', get_memory_usage())
#Model params



import lightgbm as lgb

lgb_params = {

                    'objective':'regression',

                    'boosting_type':'gbdt',

                    'metric':'rmse',

                    'n_jobs':-1,

                    'learning_rate':0.05,

                    'num_leaves': 2**8,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.7,

                    'subsample_freq':1,

                    'subsample':0.7,

                    'n_estimators':800,

                    'max_bin':255,

                    'verbose':-1,

                    'seed': SEED,

                    'early_stopping_rounds':100, 

                } 
# Models saving

model_filename = 'lgbm'

models = []



# Load train_df from hdd

train_df = pd.read_pickle('train_df.pkl')



remove_columns = ['timestamp',TARGET]

features_columns = [col for col in list(train_df) if col not in remove_columns]



if LOCAl_TEST:

    tr_data = lgb.Dataset(train_df.iloc[:15000000][features_columns], label=np.log1p(train_df.iloc[:15000000][TARGET]))

    vl_data = lgb.Dataset(train_df.iloc[15000000:][features_columns], label=np.log1p(train_df.iloc[15000000:][TARGET]))

    eval_sets = [tr_data,vl_data]

else:

    tr_data = lgb.Dataset(train_df[features_columns], label=np.log1p(train_df[TARGET]))

    eval_sets = [tr_data]



# Remove train_df from hdd

os.system('rm train_df.pkl')



# Lets make 5 seeds mix model

for cur_seed in [42,43,44,45,46]:

    

    # Seed everything

    seed_everything(cur_seed)

    lgb_params['seed'] = cur_seed

    

    estimator = lgb.train(

                lgb_params,

                tr_data,

                valid_sets = eval_sets,

                verbose_eval = 100,

            )



    # For CV you may add fold number

    # pickle.dump(estimator, open(model_filename + '__fold_' + str(i) + '.bin', "wb"))

    pickle.dump(estimator, open(model_filename + '__seed_' + str(cur_seed)  + '.bin', 'wb'))

    models.append(model_filename + '__seed_' + str(cur_seed)  + '.bin')



if not LOCAl_TEST:

    del tr_data, train_df

    gc.collect()
######### Predict #############



if not LOCAl_TEST:

    

    # Load test_df from hdd

    test_df = pd.read_pickle('test_df.pkl')

    

    # Remove unused columns

    test_df = test_df[features_columns]

    

    # Remove test_df from hdd

    os.system('rm test_df.pkl')

    

    # Read submission file

    submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')



    # Remove row_id for a while

    del submission['row_id']

    

    for model_path in models:

        print('Predictions for', model_path)

        estimator = pickle.load(open(model_path, 'rb'))



        predictions = []

        batch_size = 2000000

        for batch in range(int(len(test_df)/batch_size)+1):

            print('Predicting batch:', batch)

            predictions += list(np.expm1(estimator.predict(test_df[features_columns].iloc[batch*batch_size:(batch+1)*batch_size])))

            

        submission['meter_reading'] += predictions

        

    # Average over models

    submission['meter_reading'] /= len(models)

    

    # Delete test_df

    del test_df

     

    # Fix negative values

    submission['meter_reading'] = submission['meter_reading'].clip(0,None)



    # Restore row_id

    submission['row_id'] = submission.index

    

    

    

    #### Check ####

    

    print(submission.iloc[:20])

    print(submission['meter_reading'].describe())