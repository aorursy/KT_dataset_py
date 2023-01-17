from sklearn.preprocessing import LabelEncoder

import datetime

import gc

import lightgbm as lgb

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

import time

import warnings



from contextlib import contextmanager

from pandas.core.common import SettingWithCopyWarning

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, StratifiedKFold



import pandas as pd

import time

import numpy as np

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import gc

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor





import pandas as pd

import numpy as np

debug=False

if debug:

    df = pd.read_csv('../input/train.csv')[:100]

    test = pd.read_csv('../input/test.csv')[:100]

    sub = pd.read_csv('../input/sample_submission.csv')[:100]

else:

    

    df = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    sub = pd.read_csv('../input/sample_submission.csv')

import pandas as pd

pd.set_option('display.max_columns', 500)



df.checkin_date = pd.to_datetime(df.checkin_date, format="%d/%m/%y")



df.checkout_date = pd.to_datetime(df.checkout_date, format="%d/%m/%y")



df.booking_date= pd.to_datetime(df.booking_date, format="%d/%m/%y")



test.checkin_date = pd.to_datetime(test.checkin_date, format="%d/%m/%y")



test.checkout_date = pd.to_datetime(test.checkout_date, format="%d/%m/%y")



test.booking_date= pd.to_datetime(test.booking_date, format="%d/%m/%y")









def dt_feat(train,date_col):

    print(train.shape)

    train[date_col+'week_day'] = train[date_col].dt.dayofweek

    train[date_col+'month'] = train[date_col].dt.month

    train[date_col+'is_month_end'] = train[date_col].dt.is_month_end

    train[date_col+'dayofyear'] = train[date_col].dt.dayofyear

    train[date_col+'day'] = train[date_col].dt.day

    train[date_col+'year'] = train[date_col].dt.year

    train[date_col+'weekyear'] = train[date_col].dt.weekofyear

    return train





train1 =dt_feat(df,'checkin_date')

train1=dt_feat(train1,'checkout_date')

train1=dt_feat(train1,'booking_date')

df = train1.copy()

train1 =dt_feat(test,'checkin_date')

train1=dt_feat(train1,'checkout_date')

train1=dt_feat(train1,'booking_date')

test = train1.copy()

df.columns







cat_cols = [f for f in df.columns if (df[f].dtype == 'object' and f not in ['reservation_id'])]

train = df.copy()

for col in cat_cols:

    lbl = LabelEncoder()

    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))

    train[col] = lbl.transform(list(train[col].values.astype('str')))

    test[col] = lbl.transform(list(test[col].values.astype('str')))



train['in_book']=train.checkin_date-train.booking_date

test['in_book']=test.checkin_date-test.booking_date

train['in_out'] = train.checkout_date-train.checkin_date

test['in_out'] = test.checkout_date-test.checkin_date

def day_get(x):

    return x.days

train.in_out = train.in_out.apply(day_get)

test.in_out = test.in_out.apply(day_get)

train.in_book = train.in_book.apply(day_get)

test.in_book = test.in_book.apply(day_get)



train_bkp=train.copy()

test_bkp=test.copy()



merge = pd.concat([train_bkp,test_bkp])



merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).reservation_id.count()).reset_index(),suffixes=('','res_mem'),on=['resort_id','memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).reservation_id.count()).reset_index(),suffixes=('','_x_res'),on='resort_id',how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','checkout_dateyear','checkout_datemonth']).reservation_id.count()).reset_index(),suffixes=('','_res_month'),on=['resort_id','checkout_dateyear','checkout_datemonth'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).reservation_id.count()).reset_index(),suffixes=('','mem'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid','checkout_dateyear']).reservation_id.count()).reset_index(),suffixes=('','mem_year'),on=['memberid','checkout_dateyear'],how='left')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).roomnights.median()).reset_index(),suffixes=('','res_amount_median'),on=['resort_id'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).roomnights.max()).reset_index(),suffixes=('','res_amount_max'),on=['resort_id'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).roomnights.min()).reset_index(),suffixes=('','res_amount_min'),on=['resort_id'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).roomnights.std()).reset_index(),suffixes=('','res_amount_std'),on=['resort_id'],how='left')



#0.9651884690956651





merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).roomnights.median()).reset_index(),suffixes=('','res_mem_amount_median'),on=['resort_id','memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).roomnights.max()).reset_index(),suffixes=('','res_mem_amount_max'),on=['resort_id','memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).roomnights.min()).reset_index(),suffixes=('','res_mem_amount_min'),on=['resort_id','memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).roomnights.std()).reset_index(),suffixes=('','res_mem_amount_std'),on=['resort_id','memberid'],how='left')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).roomnights.median()).reset_index(),suffixes=('','mem_amount_median'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).roomnights.max()).reset_index(),suffixes=('','mem_amount_max'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).roomnights.min()).reset_index(),suffixes=('','mem_amount_min'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).roomnights.std()).reset_index(),suffixes=('','mem_amount_std'),on=['memberid'],how='left')





merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','room_type_booked_code']).reservation_id.count()).reset_index(),suffixes=('','_res_type'),on=['resort_id','room_type_booked_code'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','checkout_datemonth','room_type_booked_code']).reservation_id.count()).reset_index(),suffixes=('','_res_code_month_x'),on=['resort_id','checkout_datemonth','room_type_booked_code'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','main_product_code']).reservation_id.count()).reset_index(),suffixes=('','_res_main_code'),on=['resort_id','main_product_code'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['channel_code','checkout_dateyear','checkout_datemonth']).reservation_id.count()).reset_index(),suffixes=('','channel_check_yearmonth'),on=['channel_code','checkout_dateyear','checkout_datemonth'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['checkout_dateyear','checkout_datemonth']).reservation_id.count()).reset_index(),suffixes=('','check_Date_month'),on=['checkout_dateyear','checkout_datemonth'],how='left')



#0.9651884690956651





merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).in_out.median()).reset_index(),suffixes=('','res_meminoutt_median'),on=['resort_id','memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).in_out.max()).reset_index(),suffixes=('','res_mem_inout_max'),on=['resort_id','memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).in_out.min()).reset_index(),suffixes=('','res_mem_inout_min'),on=['resort_id','memberid'],how='left')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).in_out.std()).reset_index(),suffixes=('','res_meminoutt_std'),on=['resort_id','memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).in_out.median()).reset_index(),suffixes=('','mem_inout_median'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).in_out.max()).reset_index(),suffixes=('','mem_inout_max'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).in_out.min()).reset_index(),suffixes=('','mem_inout_min'),on=['memberid'],how='left')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).in_out.std()).reset_index(),suffixes=('','mem_inout_std'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).total_pax.median()).reset_index(),suffixes=('','mem_inout_median_pax'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).total_pax.max()).reset_index(),suffixes=('','mem_inout_max_pax'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).total_pax.min()).reset_index(),suffixes=('','mem_inout_min_pax'),on=['memberid'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid']).total_pax.std()).reset_index(),suffixes=('','mem_inout_std_pax'),on=['memberid'],how='left')







merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).total_pax.median()).reset_index(),suffixes=('','res_inout_median_pax'),on=['resort_id'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).total_pax.max()).reset_index(),suffixes=('','res_inout_max_pax'),on=['resort_id'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).total_pax.min()).reset_index(),suffixes=('','res_inout_min_pax'),on=['resort_id'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id']).total_pax.std()).reset_index(),suffixes=('','res_inout_std_pax'),on=['resort_id'],how='left')







# merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).total_pax.median()).reset_index(),suffixes=('','mem_res_inout_median_pax'),on=['resort_id','memberid'],how='left')

# merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).total_pax.max()).reset_index(),suffixes=('','mem_res_inout_max_pax'),on=['resort_id','memberid'],how='left')

# merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).total_pax.min()).reset_index(),suffixes=('','mem_res_inout_min_pax'),on=['resort_id','memberid'],how='left')

# merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','memberid']).total_pax.std()).reset_index(),suffixes=('','mem_res_inout_std_pax'),on=['resort_id','memberid'],how='left')









merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','main_product_code']).reservation_id.count()).reset_index(),suffixes=('','res_memmain_product_code'),on=['resort_id','main_product_code'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['main_product_code']).reservation_id.count()).reset_index(),suffixes=('','_x_resmain_product_code'),on='main_product_code',how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid','resort_id','main_product_code']).reservation_id.count()).reset_index(),suffixes=('','memjknknk'),on=['memberid','resort_id','main_product_code'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid','main_product_code']).reservation_id.count()).reset_index(),suffixes=('','mem_yearmain_product_code'),on=['memberid','main_product_code'],how='left')







#merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','checkout_dateyear','checkin_dateweekyear']).total_pax.count()).reset_index(),suffixes=('','res_inout_std_pax','week_year'),on=['resort_id','checkout_dateyear','checkin_dateweekyear'],how='left')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','checkout_dateyear','checkin_dateweekyear']).reservation_id.count()).reset_index(),suffixes=('','_res_month_week_year'),on=['resort_id','checkout_dateyear','checkin_dateweekyear'],how='left')







merge = merge.merge(pd.DataFrame(merge.groupby(by=['memberid','checkout_dateyear','checkin_dateweekyear']).reservation_id.count()).reset_index(),suffixes=('','_res_month_week_year_member'),on=['memberid','checkout_dateyear','checkin_dateweekyear'],how='left')







import datetime



merge['booking_dayes'] = (datetime.datetime.today()-merge['booking_date']).dt.days

merge['checkin_dayes'] = (datetime.datetime.today()-merge['checkin_date']).dt.days

merge['checkout_dayes'] = (datetime.datetime.today()-merge['checkout_date']).dt.days







merge['rooms_booked_per_night'] = merge.in_out/merge.roomnights

merge['rooms_bookinf_per_night'] = merge.in_book/merge.roomnights





merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).total_pax.median()).reset_index(),suffixes=('','mem_inout_median_pax_state_code'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).total_pax.max()).reset_index(),suffixes=('','mem_inout_max_pax_state_code'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).total_pax.min()).reset_index(),suffixes=('','mem_inout_min_pax_state_code'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).total_pax.std()).reset_index(),suffixes=('','mem_inout_std_pax_state_code'),on=['state_code_residence'],how='left')











merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).roomnights.median()).reset_index(),suffixes=('','res_mem_amount_medianstate_code_resid'),on=['resort_id','state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).roomnights.max()).reset_index(),suffixes=('','res_mem_amount_maxstate_code_resid'),on=['resort_id','state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).roomnights.min()).reset_index(),suffixes=('','res_mem_amount_minstate_code_resid'),on=['resort_id','state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).roomnights.std()).reset_index(),suffixes=('','res_mem_amount_stdstate_code_resid'),on=['resort_id','state_code_residence'],how='left')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).roomnights.median()).reset_index(),suffixes=('','mem_amount_medianstate_code_resid'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).roomnights.max()).reset_index(),suffixes=('','mem_amount_maxstate_code_resid'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).roomnights.min()).reset_index(),suffixes=('','mem_amount_minstate_code_resid'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).roomnights.std()).reset_index(),suffixes=('','mem_amount_stdstate_code_resid'),on=['state_code_residence'],how='left')







merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).in_out.median()).reset_index(),suffixes=('','res_meminoutt_medianstate_code_resid'),on=['resort_id','state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).in_out.max()).reset_index(),suffixes=('','res_mem_inout_maxstate_code_resid'),on=['resort_id','state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).in_out.min()).reset_index(),suffixes=('','res_mem_inout_minstate_code_resid'),on=['resort_id','state_code_residence'],how='left')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).in_out.std()).reset_index(),suffixes=('','res_meminoutt_stdstate_code_resid'),on=['resort_id','state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).in_out.median()).reset_index(),suffixes=('','mem_inout_medianstate_code_resid'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).in_out.max()).reset_index(),suffixes=('','mem_inout_maxstate_code_resid'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).in_out.min()).reset_index(),suffixes=('','mem_inout_minstate_code_resid'),on=['state_code_residence'],how='left')





predictors=[]



def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):



    print(f">> \nExtracting {agg_suffix} time calculation features...\n")

    

    GROUP_BY_NEXT_CLICKS = [

    

     {'groupby': ['resort_id', 'memberid']},

     {'groupby': ['resort_id']},

        

    ]



    # Calculate the time to next click for each group

    for spec in GROUP_BY_NEXT_CLICKS:

    

       # Name of new feature

        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    

    

        # Unique list of features to select

        all_features = spec['groupby'] + ['booking_date']



        # Run calculation

        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")

        df[new_feature] = (df.booking_date - df[all_features].groupby(spec[

                'groupby']).booking_date.shift(+1) ).dt.days.astype(agg_type)

        

        predictors.append(new_feature)

        gc.collect()

    return (df)    





def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):

    

    print(f">> \nExtracting {agg_suffix} time calculation features...\n")

    

    GROUP_BY_NEXT_CLICKS = [

    

    # V3

   

     {'groupby': ['resort_id', 'memberid']},

     {'groupby': ['resort_id']},

    # {'groupby': ['memberid']}

        ]



    # Calculate the time to next click for each group

    for spec in GROUP_BY_NEXT_CLICKS:

    

       # Name of new feature

        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    

    

        # Unique list of features to select

        all_features = spec['groupby'] + ['booking_date']



        # Run calculation

        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")

        df[new_feature] = (df[all_features].groupby(spec[

            'groupby']).booking_date.shift(-1) - df.booking_date).dt.days.astype(agg_type)

        

        #predictors.webpage_idend(new_feature)

        gc.collect()

    return (df)



              



merge = do_prev_Click(merge)

merge = do_next_Click(merge)



categor_col =['member_age_buckets','memberid','resort_id',

 'cluster_code',

 'reservationstatusid_code',

 'resort_id',

 'channel_code',

 'main_product_code',

 'persontravellingid',

 'resort_region_code',

 'resort_type_code',

 'room_type_booked_code',

 'season_holidayed_code',

 'state_code_residence',

 'state_code_resort',

 'member_age_buckets',

 'booking_type_code']



for i in categor_col:

    print(i,train[i].nunique())



for i in categor_col:

    merge[i] = merge[i].astype('category')



merge = merge.merge(pd.DataFrame(merge.groupby(by=['state_code_residence']).reservation_id.count()).reset_index(),suffixes=('','_x_res_mem_state_score'),on=['state_code_residence'],how='left')

merge = merge.merge(pd.DataFrame(merge.groupby(by=['resort_id','state_code_residence']).reservation_id.count()).reset_index(),suffixes=('','_x_res_mem_state_scoreresort_id'),on=['resort_id','state_code_residence'],how='left')





train = merge.loc[merge.amount_spent_per_room_night_scaled.notnull()]

test = merge.loc[merge.amount_spent_per_room_night_scaled.isnull()]



X= train.drop(['reservation_id','booking_date','checkin_date','checkout_date','amount_spent_per_room_night_scaled','memberid'],axis=1)



y=train.amount_spent_per_room_night_scaled




def run_single_lgb(X,y):

    print("Making predictions on Single LGB model")

    clf=LGBMRegressor(n_estimators=717,learning_rate=.05,subsample=0.9,reg_alpha=25,min_child_weight=49)

    clf.fit(X,y)

    pred_test_lgb_single_model = clf.predict(test[X.columns])

    return pred_test_lgb_single_model



MAX_ROUNDS = 1231

OPTIMIZE_ROUNDS = True

LEARNING_RATE = 0.05

model = CatBoostRegressor(verbose=100,

    learning_rate=LEARNING_RATE, 

    l2_leaf_reg = 8, 

    iterations = MAX_ROUNDS,

#    verbose = True,

    loss_function='RMSE',

)

def run_single_catboost(X,y):

    fit_model = model.fit(X,y)

    pred_catboost_model=model.predict(test[X.columns])

    return pred_catboost_model
def folded_catboost(X,y):

    try:

        X.drop('target',axis=1,inplace=True)

    except:

        print("Not there")



    K = 5

    x_test = test[X.columns]

    kf = KFold(n_splits = K, random_state = 1, shuffle = True)

    y_valid_pred = 0*y

    y_test_pred = 0

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        # Create data for this fold

        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]

        print( "\nFold ", i)



        # Run model for this fold

        if OPTIMIZE_ROUNDS:

            fit_model = model.fit(X_train, y_train, 

                                   eval_set=[(X_valid, y_valid)],

                                   use_best_model=True

                                 )

            print( "  N trees = ", model.tree_count_ )

        else:

            fit_model = model.fit( X_train, y_train )



        # Generate validation predictions for this fold

        pred = fit_model.predict(X_valid)

        print( "  Gini = ", np.sqrt(mean_squared_error(y_valid, pred) ))

        y_valid_pred.iloc[test_index] = pred



        # Accumulate test set predictions

        y_test_pred += fit_model.predict(x_test)



    y_test_pred /= K  # Average test set predictions

    return y_test_pred
FEATS_EXCLUDED=['target']

categor_col =cat_cols+[

       'channel_code',

            'main_product_code', 

      'persontravellingid', 'resort_region_code',

       'resort_type_code', 'room_type_booked_code', 

       'season_holidayed_code', 'state_code_residence', 'state_code_resort',

       'member_age_buckets', 'booking_type_code', 'memberid',

       ]



categor_col.remove('memberid')

categor_col.remove('memberid')





def kfold_lightgbm(train_df, test_df, num_folds, y,stratified = False, debug= False):

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    train_df['target']=y

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=326)



    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]



    # k-fold

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]



        # set data structure

        lgb_train = lgb.Dataset(train_x,

                                label=train_y,

                                free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,

                               label=valid_y,

                               free_raw_data=False)

        params = {"objective" : "regression", "metric" : "rmse", 'n_estimators':3000, 'early_stopping_rounds':200,

              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.9,

               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.8,"lambda_l1": 25,"min_child_weight" : 44.97

          ,"min_child_samples": 5,

             }







        reg = lgb.train(

                        params,

                        lgb_train,

                        valid_sets=[lgb_train, lgb_test],

                        valid_names=['train', 'test'],

                        num_boost_round=10000,

                        early_stopping_rounds= 200,

                        verbose_eval=100,

            categorical_feature=categor_col

                        )



        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)

        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))

        del reg, train_x, train_y, valid_x, valid_y

        gc.collect()



    # display importances

    display_importances(feature_importance_df)



    if not debug:

        # save submission file

        return sub_preds





def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]



    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances.png')





def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))



FEATS_EXCLUDED=['target']

pred_test_lgb_single_model=run_single_lgb(X,y)

pred_catboost_model  = run_single_catboost(X,y)

y_test_fold_cat = folded_catboost(X,y)

pred_lgb_5_fold = kfold_lightgbm(X,test,5,train.amount_spent_per_room_night_scaled)

pred_ensem= pred_test_lgb_single_model*0.3+pred_catboost_model*0.2+pred_lgb_5_fold*0.3+y_test_fold_cat*0.2
pred_ensem
try:

    X.drop('target',axis=1,inplace=True)

except:

    print("Not there")

pred_ensem= pred_test_lgb_single_model*0.3+pred_catboost_model*0.2+pred_lgb_5_fold*0.3+y_test_fold_cat*0.2
sub.amount_spent_per_room_night_scaled=pred_ensem

sub.to_csv('submission_18_rewrite_parallel.csv',index=None)
# for i in categor_col:

#     X[i]=X[i].astype(int)

#     test[i]=test[i].astype(int)    

features = X.columns



import pandas as pd

import numpy as np

from scipy.stats import skew

import xgboost as xgb

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from math import sqrt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

import gc



NFOLDS = 3

SEED = 0

NROWS = None

categorical_feats =categor_col



x_train = X[features]

x_test = test[features]

ntrain = x_train.shape[0]

ntest = x_test.shape[0]

y_train=y



kf = KFold(n_splits = NFOLDS, shuffle=True, random_state=SEED)









class SklearnWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        print(x_train.isnull().sum().sum())

        self.clf.fit(x_train.fillna(-999).astype(float), y_train)



    def predict(self, x):

        return self.clf.predict(x.fillna(-999).astype(float))



class CatboostWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_seed'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

        

class LightGBMWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['feature_fraction_seed'] = seed

        params['bagging_seed'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)





class XgbWrapper(object):

    def __init__(self, seed=0, params=None):

        self.param = params

        self.param['seed'] = seed

        self.nrounds = params.pop('nrounds', 250)



    def train(self, x_train, y_train):

        for i in categor_col:

            

            try:

                x_train[i]=x_train[i].astype(int)

            except:

                print(i)

        dtrain = xgb.DMatrix(x_train.drop(['season_holidayed_code','state_code_residence'],axis=1,inplace=True), label=y_train)

        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)



    def predict(self, x):

        for i in categor_col:

            try:

                x[i]=x[i].astype(int)    

            except:

                print(i)

                

        return self.gbdt.predict(xgb.DMatrix(x.drop(['season_holidayed_code','state_code_residence'],axis=1,inplace=True)))





def get_oof(clf):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train.loc[train_index]

        y_tr = y_train.loc[train_index]

        x_te = x_train.loc[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



def get_oof_et(clf):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train.loc[train_index]

        y_tr = y_train.loc[train_index]

        x_te = x_train.loc[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test.fillna(-999))



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)







et_params = {

    'n_jobs': 16,

    'n_estimators': 200,

    'max_features': 0.5,

    'max_depth': 12,

    'min_samples_leaf': 2,

}



rf_params = {

    'n_jobs': 16,

    'n_estimators': 200,

    'max_features': 0.2,

    'max_depth': 12,

    'min_samples_leaf': 2,

}



xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.7,

    'silent': 1,

    'subsample': 0.7,

    'learning_rate': 0.075,

    'objective': 'reg:linear',

    'max_depth': 4,

    'num_parallel_tree': 1,

    'min_child_weight': 1,

    'nrounds': 400

}



catboost_params = {

    'iterations': 1400,

    'learning_rate': 0.05,

    'depth': 8,

    'l2_leaf_reg': 17,



    'eval_metric': 'RMSE',

    'od_type': 'Iter',

    'allow_writing_files': False

}



lightgbm_params  = {"objective" : "regression", "metric" : "rmse", 

                    

              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.9,

               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.8,"lambda_l1": 25,"min_child_weight" : 44.97

          ,"min_child_samples": 5,'iterations': 500,

             }







xg = XgbWrapper(seed=SEED, params=xgb_params)

cb = CatboostWrapper(clf= CatBoostRegressor, seed = SEED, params=catboost_params)

lg = LightGBMWrapper(clf = LGBMRegressor, seed = SEED, params = lightgbm_params)



#xg_oof_train, xg_oof_test = get_oof(xg)

print(2)

cb_oof_train, cb_oof_test = get_oof(cb)

print(3)

lg_oof_train, lg_oof_test = get_oof(lg)

print(4)

#print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))

#print("ET-CV: {}".format(sqrt(mean_squared_error(y_train.head(100), et_oof_train))))

#print("RF-CV: {}".format(sqrt(mean_squared_error(y_train.head(100), rf_oof_train))))

print("CB-CV: {}".format(sqrt(mean_squared_error(y_train, cb_oof_train))))

print("LG-CV: {}".format(sqrt(mean_squared_error(y_train, lg_oof_train))))

x_train = np.concatenate(( lg_oof_train, cb_oof_train), axis=1)

x_test = np.concatenate((lg_oof_test, cb_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

logistic_regression = LinearRegression()

logistic_regression.fit(x_train,y_train)

pred = logistic_regression.predict(x_test)



sub.amount_spent_per_room_night_scaled=pred_ensem*0.8+pred*0.2

sub.to_csv('xgb_cat_ensemb.csv',index=None)
