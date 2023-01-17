# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from pathlib import Path

import os

import os, gc

import random

import datetime



from tqdm import tqdm_notebook as tqdm



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")

#import xgboost as xgb
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16 or not. feather format does not support float16.

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

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
def dataset_reader():

    lists=['weather_test.feather'

          ,'weather_train.feather'

          ,'test.feather'

          ,'train.feather'

          ,'building_metadata.feather']

    Input = Path('/kaggle/input/ashrae-feather-format-for-fast-loading/')

    

    wtest = pd.read_feather(Input/lists[0])

    wtrain = pd.read_feather(Input/lists[1])

    test = pd.read_feather(Input/lists[2])

    train = pd.read_feather(Input/lists[3])

    bmdata = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/building_metadata.feather')

    

    #train = train.sort_values(by=['building_id','meter','timestamp']).reset_index()



    gc.collect()

    #wfull = pd.concat([wtrain,wtest],sort=True,ignore_index = True)

    train = reduce_mem_usage(train)

    test = reduce_mem_usage(test)

    wtrain = reduce_mem_usage(wtrain)

    wtest = reduce_mem_usage(wtest)

    bmdata = reduce_mem_usage(bmdata)



    return train,test,wtrain,wtest,bmdata

train,test,wtrain,wtest,bmdata = dataset_reader()
bm1 = train.groupby('building_id').meter_reading.mean().to_dict()

bm2 = train.groupby('building_id').meter_reading.median().to_dict()

bm3 = train.groupby('building_id').meter_reading.min().to_dict()

bm4 = train.groupby('building_id').meter_reading.max().to_dict()

bsd = train.groupby('building_id').meter_reading.std().to_dict()

bmdata['bm1'] = bmdata.building_id.map(bm1)

bmdata['bm2'] = bmdata.building_id.map(bm2)

bmdata['bm3'] = bmdata.building_id.map(bm3)

bmdata['bm4'] = bmdata.building_id.map(bm4)

bmdata['bsd'] = bmdata.building_id.map(bsd)



primary_use = {0: 'Religious worship',

  1: 'Warehouse/storage',

  2: 'Technology/science',

  3: 'Other',

  4: 'Retail',

  5: 'Parking',

  6: 'Lodging/residential',

  7: 'Manufacturing/industrial',

  8: 'Public services',

  9: 'Food sales and service',

  10: 'Entertainment/public assembly',

  11: 'Utility',

  12: 'Office',

  13: 'Healthcare',

  14: 'Services',

  15: 'Education'}



inv_map = {v: k for k, v in primary_use.items()}

floor_avg = bmdata.groupby('primary_use').floor_count.mean().apply(np.ceil).to_dict()



def addTime(df):

    df['Month']= df.timestamp.dt.month.astype(np.uint8)

    df['Day']= df.timestamp.dt.day.astype(np.uint8)

    df['Hour'] = df.timestamp.dt.hour.astype(np.uint8)

    df['Weekday'] = df.timestamp.dt.weekday.astype(np.uint8)

    #df['Date'] = df.timestamp.dt.date

    return df



#Rolling!



lists = ['air_temperature','cloud_coverage'

            ,'dew_temperature','precip_depth_1_hr'

            , 'sea_level_pressure','wind_direction','wind_speed']



def get_rolling(df,f,p):

    

    if f == 'meter_reading':

        df[f'{f}_{p}_mean'] = df.groupby('building_id')[f].rolling(p).mean().reset_index().astype(np.float16)[f]

        df[f'{f}_{p})_std'] = df.groupby('building_id')[f].rolling(p).std().reset_index().astype(np.float16)[f]

    else:

        #df[f'{f}_{p}'] =

        gp= df.groupby('site_id')[f].rolling(p).mean().reset_index().astype(np.float16)[f]

        gp2 = df.groupby('site_id')[f].rolling(p).std().reset_index().astype(np.float16)[f]

        gp3 = df.groupby('site_id')[f].rolling(p).min().reset_index().astype(np.float16)[f]

        for i in f:

            df[f'{i}_{p}_mean'] = gp[i]

            df[f'{i}_{p}_std'] = gp2[i]

            df[f'{i}_{p}_min'] = gp3[i]

    #for i in [24,24*7,24*7*4,24*7*4*4]:

    #df[f'{f}_{p}'] = train.groupby('building_id')['meter_reading'].rolling(24).mean().reset_index()

    return df
gc.collect()

gc.collect()
def Create_train_feature(df,wdf,meter,smooth,types=None):

    

    new_df = df.loc[df.meter==meter]

    #new_df = get_rolling(new_df,'meter_reading',smooth)

    new_df = new_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    new_df = new_df.query('not(building_id>=1325 & timestamp >= "2016-02-01" & timestamp <= "2016-04-20")')



    bmdata['log_sqf'] = np.log(bmdata.square_feet)

    if types!='test':

        new_df['m_log1p'] = np.log1p(new_df.meter_reading)

    ##

    bmdata['floor_count'] = bmdata.primary_use.map(floor_avg)

    bmdata['primary_use'] = bmdata['primary_use'].map(inv_map)

    ##

    new_df = new_df.merge(bmdata, on='building_id',how='left')

    #use it later

    new_wdf = get_rolling(wdf,lists,smooth)

    new_wdf = new_wdf.query('not(site_id==15 & timestamp >= "2016-02-01" & timestamp <= "2016-04-20")')

    

    #Temporary

    #wdf = wdf.query('not(site_id==15 & timestamp >= "2016-02-01" & timestamp <= "2016-04-20")')

    

    new_df = new_df.merge(wdf, on=['site_id','timestamp'],how='left')

    new_df = addTime(new_df)

    

    return new_df
import lightgbm as lgb

import optuna

from optuna import Trial

from sklearn.metrics import mean_squared_error



def objective(trial,meter,rounds):

    Losses = 0

    folds = 4

    shuffle = True

    seed = 7

    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)

    selective_train = Create_train_feature(train,wtrain,meter,30)

    selective_train = reduce_mem_usage(selective_train)

    target= 'm_log1p'



    do_not_use = ['meter_reading','m_log1p'

                     ,'is_train'

                    ,'row_id'

                    ,'square_feet'

                    ,'timestamp'

                  ,'index'

                     ]



    feature_columns = [c for c in selective_train.columns if c not in do_not_use ]



  

    for train_idx, valid_idx in tqdm(kf.split(selective_train,selective_train['building_id']),total=folds):

        print(f'###############Starting_meter :{meter}###############')

        print(f'Training and predicting for target {target}')

        Xtr = selective_train[feature_columns].iloc[train_idx]

        Xv = selective_train[feature_columns].iloc[valid_idx]

        ytr = selective_train[target].iloc[train_idx].values

        yv = selective_train[target].iloc[valid_idx].values



        dtrain = lgb.Dataset(Xtr, label=ytr)

        dvalid = lgb.Dataset(Xv, label=yv ,reference= dtrain)



        print('Train_size: ',Xtr.shape[0],'Validation_size: ', Xv.shape[0])



  

        Loss = opt_lgbm(trial,rounds,dtrain,dvalid,Xv,yv)

        Losses = Losses + Loss

    print(print(f'Cumulative MSLE: {Losses}'))

    return Losses



def opt_lgbm(trial,rounds,dtrain,dvalid,Xv,yv,on_gpu=0):



    param = {"objective": "regression",

        'metric': 'rmse',

        'verbosity': -1,

        'boosting_type': 'gbdt',

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        }

    if on_gpu==1:

        print('Use_GPU')

        gpu={'device': 'gpu'}

        param.update(gpu)



    print(param)

    model = lgb.train(param,

              dtrain,

              num_boost_round=rounds,

              valid_sets=(dtrain, dvalid),

              early_stopping_rounds=20,

              verbose_eval = 200)

    valid_prediction = model.predict(Xv, num_iteration=model.best_iteration)

    #Loss = np.sqrt(np.mean((np.log1p(valid_prediction)-np.log1p(yv))**2))

    #Loss = np.sqrt( np.mean(np.subtract( np.log1p( valid_prediction ), np.log1p( yv ) )**2 ) )

    Loss = np.sqrt(mean_squared_error(valid_prediction, yv))

    gc.collect()

    print(print(f'RMSLE: {Loss}'))

    return Loss

#Find The Best Hyper Parameter 'minimize' RMSE(RMSLE)

rounds = 1

meters = [0,1,2,3]

n_trials = 1

#Meter 0

study0 = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner())

study0.optimize(lambda trial: objective(trial, meters[0],rounds), n_trials=n_trials)

#Meter 1

study1 = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner())

study1.optimize(lambda trial: objective(trial, meters[1],rounds), n_trials=n_trials)

#Meter 2

study2 = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner())

study2.optimize(lambda trial: objective(trial, meters[2],rounds), n_trials=n_trials)

#Meter 3

study3 = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner())

study3.optimize(lambda trial: objective(trial, meters[3],rounds), n_trials=n_trials)
#Extract Best Hyper Parameters For Each Meters

BestHPMeter_0 = study0.best_trial.params

BestHPMeter_1 = study1.best_trial.params

BestHPMeter_2 = study2.best_trial.params

BestHPMeter_3 = study3.best_trial.params
print(BestHPMeter_0)

print(BestHPMeter_1)

print(BestHPMeter_2)

print(BestHPMeter_3)
import lightgbm as lgb

def lgbm(df,wdf,meter,rounds,on_gpu=0):

    Losses = []

    models = []

    selective_train = Create_train_feature(df,wdf,meter,30)



    folds = 4

    shuffle = True

    seed = 7

    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)



    target= 'm_log1p'



    do_not_use = ['meter_reading','m_log1p'

                     ,'is_train'

                    ,'row_id'

                    ,'square_feet'

                    ,'timestamp'

                  ,'index'

                     ]



    feature_columns = [c for c in selective_train.columns if c not in do_not_use ]

    

    print(f'Feature Colums : {len(feature_columns)}')



    best_params = {"objective": "regression",

          "boosting_type": "gbdt",

          "metric": "rmse",

          "verbose": 0,

         }

    if on_gpu==1:

        print('Use_GPU')

        gpu={'device': 'gpu'}

        best_params.update(gpu)



    if meter == 0:

        best_params.update(BestHPMeter_0)

    elif meter == 1:

        best_params.update(BestHPMeter_1)

    elif meter == 2:

        best_params.update(BestHPMeter_2)

    else:

        best_params.update(BestHPMeter_3)



  

    for train_idx, valid_idx in tqdm(kf.split(selective_train,selective_train['building_id']),total=folds):



        print(f'###############Starting_meter :{meter}###############')

        print(f'Training and predicting for target {target}')

        Xtr = selective_train[feature_columns].iloc[train_idx]

        Xv = selective_train[feature_columns].iloc[valid_idx]

        ytr = selective_train[target].iloc[train_idx].values

        yv = selective_train[target].iloc[valid_idx].values



        dtrain = lgb.Dataset(Xtr, label=ytr)

        dvalid = lgb.Dataset(Xv, label=yv ,reference= dtrain)



        print('Train_size: ',Xtr.shape[0],'Validation_size: ', Xv.shape[0])

        print(f'Train with {best_params}')

        model = lgb.train(best_params,

              dtrain,

              num_boost_round=rounds,

              valid_sets=(dtrain, dvalid),

              early_stopping_rounds=20,

              verbose_eval = 50)

        models.append(model)

        gc.collect()



    return models



  

def Lgbm_Training_LOOP(df,wdf,meter,rounds):

    

    print("LGBM Main Training Meter_{meter}")



    models  =  lgbm(df,wdf,meter,rounds)

    

    print(models)

    print('End')

    gc.collect()



    return models

    

models0 = Lgbm_Training_LOOP(train,wtrain,0,1)

models1 = Lgbm_Training_LOOP(train,wtrain,1,1)

models2 = Lgbm_Training_LOOP(train,wtrain,2,1)

models3 = Lgbm_Training_LOOP(train,wtrain,3,1)

gc.collect()
def prediction(meter,models):

    meter_reading = []

    selective_test = Create_train_feature(test,wtest,meter,30,types='test')

    selective_test = reduce_mem_usage(selective_test)

    set_size = len(selective_test)

    print(f'meter {meter} : test_size {set_size}')

    iterations = 100

    batch_size = int(np.ceil(set_size / iterations))

    do_not_use = ['meter_reading','m_log1p'

                     ,'is_train'

                    ,'row_id'

                    ,'square_feet'

                    ,'timestamp'

                  ,'index'

                     ]

    feature_columns = [c for c in selective_test.columns if c not in do_not_use ]

    

    print(f'Feature Columns : {len(feature_columns)}')



    for i in tqdm(range(iterations)):

        pos = i*batch_size

        fold_preds = [np.expm1(model.predict(selective_test[feature_columns].iloc[pos : pos+batch_size],num_iteration=model.best_iteration)) for model in models]

        meter_reading.extend(np.mean(fold_preds, axis=0))

    print(np.mean(meter_reading))

    gc.collect()

    assert len(meter_reading) == set_size

    return meter_reading
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

gc.collect()
prediction0 = prediction(0,models0)

submission.loc[test['meter'] == 0, 'meter_reading'] = np.clip(prediction0, a_min=0, a_max=None)

prediction1 = prediction(1,models1)

submission.loc[test['meter'] == 1, 'meter_reading'] = np.clip(prediction1, a_min=0, a_max=None)

prediction2 = prediction(2,models2)

submission.loc[test['meter'] == 2, 'meter_reading'] = np.clip(prediction2, a_min=0, a_max=None)

prediction3 = prediction(3,models3)

submission.loc[test['meter'] == 3, 'meter_reading'] = np.clip(prediction3, a_min=0, a_max=None)
submission.head(10)
submission.describe()
submission.to_csv('submission.csv', index=False, float_format='%.4f')