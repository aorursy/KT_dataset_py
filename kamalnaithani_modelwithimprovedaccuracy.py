# General imports

import numpy as np

import pandas as pd

import os, sys, gc, time, warnings, pickle, psutil, random



# custom imports

from multiprocessing import Pool        # Multiprocess Runs



warnings.filterwarnings('ignore')


def seed_everything(seed=0):

    random.seed(seed)

    np.random.seed(seed)



    

## Multiprocess Runs

def df_parallelize_run(func, t_split):

    num_cores = np.min([N_CORES,len(t_split)])

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, t_split), axis=1)

    pool.close()

    pool.join()

    return df


def get_data_by_store(store):

    

    # Read and contact basic feature

    df = pd.concat([pd.read_pickle(BASE),

                    pd.read_pickle(PRICE).iloc[:,2:],

                    pd.read_pickle(CALENDAR).iloc[:,2:]],

                    axis=1)

    

    df = df[df['store_id']==store]

    df2 = pd.read_pickle(MEAN_ENC)[mean_features]

    df2 = df2[df2.index.isin(df.index)]

    

    df3 = pd.read_pickle(LAGS).iloc[:,3:]

    df3 = df3[df3.index.isin(df.index)]

    

    df = pd.concat([df, df2], axis=1)

    del df2 # to not reach memory limit 

    

    df = pd.concat([df, df3], axis=1)

    del df3 # to not reach memory limit 

    

    # Create features list

    features = [col for col in list(df) if col not in remove_features]

    df = df[['id','d',TARGET]+features]

    

    # Skipping first n rows

    df = df[df['d']>=START_TRAIN].reset_index(drop=True)

    

    return df, features



# Recombine Test set after training

def get_base_test():

    base_test = pd.DataFrame()



    for store_id in STORES_IDS:

        temp_df = pd.read_pickle('test_'+store_id+'.pkl')

        temp_df['store_id'] = store_id

        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

    

    return base_test





def make_lag(LAG_DAY):

    lag_df = base_test[['id','d',TARGET]]

    col_name = 'sales_lag_'+str(LAG_DAY)

    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)

    return lag_df[[col_name]]





def make_lag_roll(LAG_DAY):

    shift_day = LAG_DAY[0]

    roll_wind = LAG_DAY[1]

    lag_df = base_test[['id','d',TARGET]]

    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)

    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())

    return lag_df[[col_name]]
########################### Model params

import lightgbm as lgb

lgb_params = {

                    'boosting_type': 'gbdt',

                    'objective': 'tweedie',

                    'tweedie_variance_power': 1.1,

                    'metric': 'rmse',

                    'subsample': 0.5,

                    'subsample_freq': 1,

                    'learning_rate': 0.03,

                    'num_leaves': 2**11-1,

                    'min_data_in_leaf': 2**12-1,

                    'feature_fraction': 0.5,

                    'max_bin': 100,

                    'n_estimators': 1400,

                    'boost_from_average': False,

                    'verbose': -1,

                } 





#prices.info(memory_usage="deep")


#prices.memory_usage(deep=True) * 1e-6
#prices.memory_usage(deep=True).sum() * 1e-6
########################### Vars

#################################################################################

VER = 2                          # Our model version

SEED = 42                        

seed_everything(SEED)            

lgb_params['seed'] = SEED       

N_CORES = psutil.cpu_count()     



#LIMITS and const

TARGET      = 'sales'           

START_TRAIN = 0                  

END_TRAIN   = 1913               

P_HORIZON   = 28                

USE_AUX     = True               



#FEATURES to remove

## These features lead to overfit

remove_features = ['id','state_id','store_id',

                   'date','wm_yr_wk','d',TARGET]

mean_features   = ['enc_cat_id_mean','enc_cat_id_std',

                   'enc_dept_id_mean','enc_dept_id_std',

                   'enc_item_id_mean','enc_item_id_std'] 





ORIGINAL = '../input/m5-forecasting-accuracy/'

BASE     = '../input/featureextraction/grid_part_1.pkl'

PRICE    = '../input/featureextraction/grid_part_2.pkl'

CALENDAR = '../input/featureextraction/grid_part_3.pkl'

LAGS     = '../input/lag-rollingfeature/lags_df_28.pkl'

MEAN_ENC = '../input/other-features/mean_encoding_df.pkl'

# AUX(pretrained) Models paths

AUX_MODELS = '../input/m5modelwithestimator1300/'







STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']

STORES_IDS = list(STORES_IDS.unique())









#SPLITS for lags creation

SHIFT_DAY  = 28

N_LAGS     = 15

LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]

ROLS_SPLIT = []

for i in [1,7,14]:

    for j in [7,14,30,60]:

        ROLS_SPLIT.append([i,j])
STORES_IDS
gc.collect()
#Model Training


if USE_AUX:

    lgb_params['n_estimators'] = 2

    

########################### Train Models

#################################################################################

for store_id in STORES_IDS:

    print('Train', store_id)

    

    # Get grid for current store

    grid_df, features_columns = get_data_by_store(store_id)

    

    train_mask = grid_df['d']<=END_TRAIN

    valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))

    preds_mask = grid_df['d']>(END_TRAIN-100)

    



    train_data = lgb.Dataset(grid_df[train_mask][features_columns], 

                       label=grid_df[train_mask][TARGET])

    train_data.save_binary('train_data.bin')

    train_data = lgb.Dataset('train_data.bin')

    

    valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 

                       label=grid_df[valid_mask][TARGET])

    

    grid_df = grid_df[preds_mask].reset_index(drop=True)

    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]

    grid_df = grid_df[keep_cols]

    grid_df.to_pickle('test_'+store_id+'.pkl')

    del grid_df

    

    # Launch seeder again to make lgb training 100% deterministic

    # with each "code line" np.random "evolves" 

    # so we need (may want) to "reset" it

    seed_everything(SEED)

    estimator = lgb.train(lgb_params,

                          train_data,

                          valid_sets = [valid_data],

                          verbose_eval = 100,

                          )

    

    model_name = 'lgb_model_'+store_id+'_v'+str(VER)+'.bin'

    pickle.dump(estimator, open(model_name, 'wb'))



    !rm train_data.bin

    del train_data, valid_data, estimator

    gc.collect()

    

    # "Keep" models features for predictions

    MODEL_FEATURES = features_columns
#size=os.path.getsize("../input/m5-custom-features/mean_encoding_df.pkl")


all_preds = pd.DataFrame()

base_test = get_base_test()

main_time = time.time()



for PREDICT_DAY in range(1,29):    

    print('Predict | Day:', PREDICT_DAY)

    start_time = time.time()



    grid_df = base_test.copy()

    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)

        

    for store_id in STORES_IDS:

        

        model_path = 'lgb_model_'+store_id+'_v'+str(VER)+'.bin' 

        if USE_AUX:

            model_path = AUX_MODELS + model_path

        

        estimator = pickle.load(open(model_path, 'rb'))

        

        day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)

        store_mask = base_test['store_id']==store_id

        

        mask = (day_mask)&(store_mask)

        base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])

    

   

    temp_df = base_test[day_mask][['id',TARGET]]

    temp_df.columns = ['id','F'+str(PREDICT_DAY)]

    if 'id' in list(all_preds):

        all_preds = all_preds.merge(temp_df, on=['id'], how='left')

    else:

        all_preds = temp_df.copy()

        

    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),

                  ' %0.2f min total |' % ((time.time() - main_time) / 60),

                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))

    del temp_df

    

all_preds = all_preds.reset_index(drop=True)

all_preds


submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]

submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)

submission.to_csv('submission_v'+str(VER)+'.csv', index=False)
gc.collect()