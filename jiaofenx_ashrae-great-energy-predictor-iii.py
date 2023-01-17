import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, sys, warnings, random, math, datetime, psutil, pickle



from tqdm import tqdm_notebook as tqdm



from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb



train = pd.read_pickle('../input/ashraedataminification/train_df.pkl')

test = pd.read_pickle('../input/ashraedataminification/test_df.pkl')

target = 'meter_reading'



## Simple "Memory profilers" to see memory usage

def get_memory_usage():

    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

print('Memory in Gb', get_memory_usage())
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        if col!=target:

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



train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

gc.collect()

print('Memory in Gb', get_memory_usage())
# force the model to use the weather data instead of dates, to avoid overfitting to the past history

features = [col for col in train.columns if col not in [target, 'DT_Y', 'DT_M', 'DT_W', 'DT_D', 'DT_day_month', 'DT_week_month']]

folds = 4

seed = 42

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

models = []



## stratify data by building_id

for tr_idx, val_idx in tqdm(kf.split(train, train['building_id']), total=folds):

    tr_x, tr_y = train[features].iloc[tr_idx], train[target][tr_idx]

    vl_x, vl_y = train[features].iloc[val_idx], train[target][val_idx]

    print({'train size':len(tr_x), 'eval size':len(vl_x)})



    tr_data = lgb.Dataset(tr_x, label=tr_y)

    vl_data = lgb.Dataset(vl_x, label=vl_y)  

    clf = lgb.LGBMRegressor(n_estimators=6000,

                            learning_rate=0.28,

                            feature_fraction=0.9,

                            subsample=0.2,  # batches of 20% of the data

                            subsample_freq=1,

                            num_leaves=20,

                            metric='rmse')

    clf.fit(tr_x, tr_y,

            eval_set=[(vl_x, vl_y)],

            early_stopping_rounds=50,

            verbose=200)

    models.append(clf)

    

gc.collect()

# split test data into batches

set_size = len(test)

iterations = 50

batch_size = set_size // iterations



meter_reading = []

for i in tqdm(range(iterations)):

    pos = i*batch_size

    fold_preds = [np.expm1(model.predict(test[features].iloc[pos : pos+batch_size])) for model in models]

    meter_reading.extend(np.mean(fold_preds, axis=0))

submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')

submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None) # clip min at zero

submission.to_csv('submission.csv', index=False)