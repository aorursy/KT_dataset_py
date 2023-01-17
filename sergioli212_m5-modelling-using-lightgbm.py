import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle, gc
import lightgbm as lgb
grid_df = pd.read_pickle('../input/grid_df_evaluation.pkl')
for col in list(grid_df):
    num_na = grid_df[col].isnull().sum()
    if num_na != 0:
        print(col, str(num_na))
# As said before, there are 30490 null value for price_momentum in prices_df, because of 3049 products and 10 stores in data(3049*10 records per day)
# after joining with grid_df, there are 30490*7day=213430 null
# similar for others
grid_df.dropna(inplace=True)
grid_df.info()
# Useless for training : ["id", "date", "sales","d", "wm_yr_wk"]  
# since train model for each store, ['state_id','store_id'] can be removed(But now 'store_id' saved for getting data to build model)
identifier = ['id', 'd']
target = ['sales']
TARGET = target[0]

features_columns = [col for col in list(grid_df) if col not in  identifier+target+['id', 'd', 'store_id','state_id', "date", "sales", "wm_yr_wk"] ]
grid_df.d.unique()
features_columns
# Run gpu-version LGBM: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89004
# !rm -r /opt/conda/lib/python3.6/site-packages/lightgbm
# !git clone --recursive https://github.com/Microsoft/LightGBM
    
# !apt-get install -y -qq libboost-all-dev

# %%bash
# cd LightGBM
# rm -r build
# mkdir build
# cd build
# cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
# make -j$(nproc)

# !cd LightGBM/python-package/;python3 setup.py install --precompile

# !mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# !rm -r LightGBM
eval_end_day = 1941
valid_end_day = 1913
# get data by store
# 'CA_1','CA_2'.'CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3'
for store_id in [  'TX_1','TX_2','TX_3','WI_1','WI_2','WI_3','CA_1','CA_2', 'CA_3','CA_4']:
    print(store_id)
    df = grid_df[grid_df['store_id']==store_id]
    
    # select for train, validation and testing
    train_mask = (df['d']<=eval_end_day) # all 1913 days data
    valid_mask = (df['d']>(eval_end_day-28)) # the last 28 days for validation, 
    ## the last 100 days for constructing features
    ## Test (All data greater than 1913 day, with some gap for recursive features
    preds_mask = df['d']>(eval_end_day-100) 

    df[preds_mask].reset_index(drop=True).to_pickle('test_'+store_id+'.pkl')
    

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
#         'device': 'gpu', 'gpu_platform_id': 0,  'gpu_device_id': 0
                    } 


    train_data = lgb.Dataset(df[train_mask][features_columns], label=df[train_mask][TARGET])

    valid_data = lgb.Dataset(df[valid_mask][features_columns], 
                       label=df[valid_mask][TARGET])

    np.random.seed(44)
    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets = [valid_data],
                          verbose_eval = 100,
                          )



    # Save model - it's not real '.bin' but a pickle file
    model_name = 'lgb_model_'+store_id+'_v2'+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))

    # Remove temporary files and objects 
    # to free some hdd space and ram memory
    del train_data, valid_data, 
    gc.collect()
