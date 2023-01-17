!pip install fastai2 


## Kaggle paths ##########################################



path = "/kaggle/input/m5-forecasting-accuracy"

path_fe ="/kaggle/input/m5-simple-fe-eval"

path_lag ="/kaggle/input/m5-lags-features-eval"

path_o = '../output/tabular_pred'



#PATHS for Features

ORIGINAL = path





#test_path  = f'{path}/dm_files'

BASE     = f'{path_fe}/grid_part_1.pkl'

PRICE    = f'{path_fe}/grid_part_2.pkl'

CALENDAR = f'{path_fe}/grid_part_3.pkl'

LAGS     = f'{path_lag}/lags_df_28.pkl'

CS   = f'{path_lag}/cumsum.pkl'



#########################################################



# AUX(pretrained) Models paths

AUX_MODELS = path
%reload_ext autoreload

%autoreload 2
import fastai2

from fastai2.tabular.all import *

from fastai2.basics import *

from fastai2.callback.all import *



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

# Read data

def get_data_by_store(store, START_TRAIN):

    print('Start train at Day ', START_TRAIN, '; End train at Day ', END_TRAIN)

    # Read and contact basic feature

    df = pd.concat([pd.read_pickle(BASE),

                    pd.read_pickle(PRICE).iloc[:,2:],

                    pd.read_pickle(CALENDAR).iloc[:,2:]],

                    axis=1)

    

    # Leave only relevant store

    df = df[df['store_id']==store]

    df = df[df['d']>=START_TRAIN]







    # keep 'd' to select by days

    lag_feat = ['sales_lag_28', 'sales_lag_56', 'sales_lag_84', 'sales_lag_168', 'sales_lag_364',

              'roll_mean_lag_28_7', 'roll_mean_lag_28_14', 'roll_mean_lag_28_28', 

            'roll_mean_lag_56_7', 'roll_mean_lag_56_14', 'roll_mean_lag_56_28', 

            'roll_mean_lag_84_7', 'roll_mean_lag_84_14', 'roll_mean_lag_84_28', 

            'roll_mean_lag_168_7', 'roll_mean_lag_168_14', 'roll_mean_lag_168_28',

            'roll_mean_lag_364_7', 'roll_mean_lag_364_14', 'roll_mean_lag_364_28',

            ]



    #df3 = pd.read_pickle(LAGS).iloc[:, 3:]

    df3 = pd.read_pickle(LAGS)[lag_feat]

    df3 = df3[df3.index.isin(df.index)]



  

    df = pd.concat([df, df3], axis=1)

    del df3 # to not reach memory limit 

    gc.collect()

    

    

    cum_feat = [ 'sales_lag_28_cum']  # 'price_sales',

    df4 = pd.read_pickle(CS)[cum_feat]

    #df4 = pd.read_pickle(CS).iloc[:, 3:]

    df4 = df4[df4.index.isin(df.index)]



  

    df = pd.concat([df, df4], axis=1)

    del df4 # to not reach memory limit 

    gc.collect()

    



    # Create features list

    features = [col for col in list(df) if col not in remove_features]

    #df = df[['id','d',TARGET]+features]

    df = df[features]

    

    # Skipping first n rows

    df = df[df['d']>=START_TRAIN].reset_index(drop=True)

    

    return df, features
def create_dbunch(train_df):

    global df_test, submit_store, y_max



    dep_var = TARGET





    print('max of sales_lag_28_cum=',  train_df.sales_lag_28_cum.max() ) # train_df.ps_lag_1.max()





    feats = list(train_df)

    for feat in feats:

      if '_lag_' in feat:

        train_df[feat]=train_df[feat].fillna(0.0)  #inplace can't work



    #train_df['log_sales']= np.log1p(train_df.sales.values)  #convert to log then can fit

    #train_df[TARGET]= train_df.sales.values  #try no Log

    train_df[TARGET]= train_df['sales']*train_df['sell_price']  #tgt=price*sales

    

    print('max, Target; max, min Sales ',train_df[TARGET].max(), train_df['sales'].max(), train_df['sales'].min())

    y_max = 1.1 * train_df[TARGET].max()

    #print('train_df null=', train_df.isnull().sum())



    all_vars = train_df.columns.tolist()

    all_vars.remove(dep_var)

    #all_vars.remove('weekday')



    cat_vars = ['item_id', 'dept_id', 'cat_id', 'd', 'price_nunique', 'item_nunique',  

                'event_name_1', 'event_name_2', 'event', 'snap_CA', 'snap_TX', 'snap_WI', 'release',

                  'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end',  ]



    xtra_vars = ['sales', 'id' ] # move 'id' here, else embedding ERROR

    cont_vars = [col for col in all_vars if col not in cat_vars+xtra_vars]





    train_mask = train_df['d']<= END_TRAIN  # all data b4 end of trg set

    df = train_df[train_mask][cat_vars + cont_vars + [dep_var]].copy()



    preds_mask = train_df['d']> (START_PRED - 400) # need 364+28days b4 predict to calc lag_roll

    df_test = train_df[preds_mask][cat_vars + cont_vars + [dep_var]].copy()  #add Target to test for recursive predict



    submit_mask = train_df['d']== START_PRED  #mask 1-store all 3049 products, for only 1 day

    submit_store = train_df[submit_mask][['id']].copy()



    #procs=[FillMissing, Categorify, Normalize]

    procs=[Categorify, Normalize]



    cut = df['d'][(df['d'] >= (END_TRAIN - P_HORIZON) )].index.min() # find smallest index

    last = df['d'][(df['d'] == END_TRAIN )].index.max() #find biggest trg index



    valid_idx = list(range(cut, last))  

    print(cut, last)

    #print (valid_idx)





    dls = TabularDataLoaders.from_df(df, path=path, procs=procs, cat_names=cat_vars, cont_names=cont_vars, 

                   y_names=TARGET, valid_idx=valid_idx, bs=2048)



    dls.show_batch()

    return dls
def create_learner(dls):

     

    #y_max = 800.0  #use 1.2*y_max

    print('set y_max at ', y_max)



    learn = tabular_learner(dls,  loss_func= nn.PoissonNLLLoss(log_input=False), layers=[500, 100], ps=[0.001, 0.01], 

                            emb_drop=0.04, y_range=[0.0, y_max], path=path_o) #define path for Kaggle 

    

    

    print('Loss fn= ', learn.loss_func)

    #learn.model



    learn.lr_find(end_lr=8)

    learn.fit_one_cycle(10, max_lr=5e-3, wd=0.01,  cbs=SaveModelCallback() ) 

    print('show sample result:')

    learn.show_results()

      

    return learn
def predict_store(learn):

    global preds, tgt



    dl = learn.dls.test_dl(df_test) #can provide Tgt y or not

    preds, tgt = learn.get_preds(dl=dl)

    #test_preds = (np.expm1(preds)).numpy().squeeze()  #inv log(x)-1.0

    test_preds = preds.numpy().squeeze() 

    

    #df_test['sales_p']=test_preds

    df_test['pricesales_p']=test_preds

    df_test['sales_p']= df_test['pricesales_p'] / df_test['sell_price']



    for day_id in range(1, 29):

    #for day_id in range(29, 57):

      submit_mask = df_test['d']== (START_PRED -1 + day_id) # 1942 - 1

      submit_store[f'F{day_id}'] = df_test[submit_mask][['sales_p']].values



    submit_store.to_pickle(f'{path_o}/{store_id}_pred.pkl')



    return
VER = 1                          # Our model version

SEED = 42                        # We want all things

seed_everything(SEED)            # to be as deterministic 

#lgb_params['seed'] = SEED        # as possible

N_CORES = psutil.cpu_count()     # Available CPU cores



#TARGET      = 'sales'            # Our target 'sales'

TARGET      = 'pricesales'



#remove_features = ['id','state_id','store_id', 'date','wm_yr_wk','d', TARGET]

remove_features = ['state_id','store_id', 'date', 'wm_yr_wk', ]

                        



#STORES_IDS = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

STORES_IDS = ['CA_1',  ]





#LIMITS and const

START_TRAIN = 1200                  # We can skip some rows (Nans/faster training)

END_TRAIN   = 1941               # End day of our train set

START_PRED  = 1942        # sid --> Decouple start_pred & end_train; Gap !!

P_HORIZON   = 28                 # Prediction horizon

USE_AUX     = False               # Use or not pretrained models



for store_id in STORES_IDS:

    print('Training Store ', store_id)

    

    # Get grid for current store

    grid_df, features_columns = get_data_by_store(store_id, START_TRAIN)

    

    

    ## Create databunch

    dbunch = create_dbunch(grid_df)



    del grid_df

    gc.collect()



    # Launch seeder again to make training 100% deterministic

    seed_everything(SEED)



    learner = create_learner(dbunch)



    predict_store(learner)

      
#df_test

df_test[(df_test['d']>=1907) & (df_test['item_id']=='FOODS_3_827')]  #.iloc[:, 20:]
tst_feat = list(df_test)

len(tst_feat), tst_feat
#path_o = f'{path}/tabular_pred'

all_preds = pd.DataFrame()



for store_id in STORES_IDS:

  temp_df = pd.read_pickle(f'{path_o}/{store_id}_pred.pkl')

  if 'id' in list(all_preds):

    #all_preds = all_preds.merge(temp_df, on=['id'], how='left')

    all_preds = pd.concat([all_preds, temp_df], axis=0, sort=False)

  else:

    all_preds = temp_df.copy()



  del temp_df

    

all_preds = all_preds.reset_index(drop=True)

all_preds
feats = list(all_preds)

feats.remove('id')

for feat in feats :

    all_preds[feat] = np.round(all_preds[feat].values * 1.00, 4)    
sample = pd.read_csv(ORIGINAL+'/sample_submission.csv')

subm_eval = sample[sample['id'].str.contains("validation")].copy()  #validation is now dummy

submission = pd.concat([subm_eval, all_preds ], axis=0, sort=False) 

submission.to_csv('submission.csv', index=False)

submission.id.nunique()
# no rows = 3049 x # stores + 30490 dummy stores

submission
submission[submission['id'].str.contains("evaluation")]