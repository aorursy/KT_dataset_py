FILENO= 1 #To distinguish the output file name.
debug=0  #Whethere or not in debuging mode
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
import matplotlib.pyplot as plt
import os
predictors=[]
def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    
    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
#     {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.fillna(2**32-1).astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)

def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):

    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},
    
    # V3
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1) ).dt.seconds.fillna(2**32-1).astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)
def do_agg( df, group_cols, agg_type='uint8', show_max=False, show_agg=True ):
    agg_name='{}_agg'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
def do_count( df, group_cols, counted, agg_type='uint8', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_count'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].count().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
def do_countuniq( df, group_cols, counted, agg_type='uint8', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
def do_cumcount( df, group_cols, counted,agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
def do_mean( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
def do_var( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')
def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint8',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('\nloading test data...\n')
    if debug:
        test_df = pd.read_csv("../input/test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    
    del test_df
    gc.collect()
    
    print('\nExtracting new features...\n')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('int8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('int8')
    train_df['minute'] = pd.to_datetime(train_df.click_time).dt.minute.astype('int8')
#     train_df['second'] = pd.to_datetime(train_df.click_time).dt.second.astype('int8')
    
    train_df = do_next_Click( train_df,agg_suffix='nextClick', agg_type='uint32'  ); gc.collect()
    train_df = do_prev_Click( train_df,agg_suffix='prevClick', agg_type='uint32'  ); gc.collect()  ## Removed temporarily due RAM sortage. 
    train_df = do_countuniq( train_df, ['ip'], 'channel' ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'os' ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'hour' ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'minute' ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app'); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'device'); gc.collect()
    train_df = do_countuniq( train_df, ['app'], 'channel'); gc.collect()
#     train_df = do_countuniq( train_df, ['ip', 'day'], 'hour' ); gc.collect()
#     train_df = do_countuniq( train_df, ['ip', 'app'], 'os'); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device'], 'channel' ); gc.collect()
    
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'channel'); gc.collect()
#     train_df = do_countuniq( train_df, ['ip','day','hour'], 'channel' ); gc.collect()
    train_df = do_countuniq( train_df, ['ip','app', 'os'], 'channel' ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
    train_df = do_countuniq( train_df, ['ip','app', 'os', 'device'], 'channel' ); gc.collect()
 

    train_df = do_cumcount( train_df, ['ip'], 'os'); gc.collect()
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
    
#     train_df = do_count( train_df, ['ip','day','hour'], 'channel' ); gc.collect()
    train_df = do_count( train_df, ['ip','app', 'os'], 'channel' ); gc.collect()
    train_df = do_count( train_df, ['ip','app', 'os', 'device'], 'channel' ); gc.collect()
    
    train_df = do_agg( train_df, ['ip', 'day', 'hour'] ); gc.collect()
    train_df = do_agg( train_df, ['ip', 'app']); gc.collect()
    train_df = do_agg( train_df, ['ip', 'app', 'os']); gc.collect()
#     train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour'); gc.collect()
#     train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour'); gc.collect()
#     train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day'); gc.collect()
#     train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour' ); gc.collect()
    del train_df['day']
    gc.collect()
    
    del train_df['minute']
    gc.collect()
   
    print('\n\nBefore appending predictors...\n\n', predictors )
    print('\n\nBefore appending predictors length...', len(predictors) )
    target = 'is_attributed'
    word= ['app','device','os', 'channel', 'hour']
    for feature in word:
        if feature not in predictors:
            predictors.append(feature)
    ##### Removing less important feature as they will change in test set         
#     for x in ['day','minute']:      
#         predictors.remove(x) # Day is 
    ################################    
    predictors_sorted= sorted(list(set(predictors))) # to remove any dublicate items
  
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    print('\n\nAfter appending predictors...\n\n',predictors_sorted )
    print('\nAfter appending predictors length...', len(predictors_sorted) )

    test_df = train_df[len_train:]
    gc.collect()
    val_df = train_df[(len_train-val_size):len_train]
    gc.collect()
    train_df = train_df[:(len_train-val_size)] 
    gc.collect()
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("\n\nTraining...")
    start_time = time.time()

    xgb_params ={'eta': 0.3,
                'max_depth': 6,
                'learning_rate': 0.3,
                'n_estimators': 500,
                'silent': False,
                'objective': 'binary:logistic',
                'eval_metric': 'auc', 
                'nthread':24,
                 'njobs': -1,
                'gamma': 5.103973694670875e-08,
                'max_delta_step': 20,
                'min_child_weight': 4,
                'subsample': 0.7,
                'colsample_bylevel': 0.1,
                'colsample_bytree': 0.7,
                'reg_alpha': 1e-09,
                'reg_lambda': 1000.0,
                'scale_pos_weight': 499.99999999999994,
                'random_state': 84,
                ' tree_method':'approx'
                } 
    
    xgtrain = xgb.DMatrix(train_df[predictors_sorted].values, label=train_df[target].values)
    xgvalid = xgb.DMatrix(val_df[predictors_sorted].values, label=val_df[target].values)
    del train_df
    del val_df
    gc.collect()
    
    trained_model = xgb.train(xgb_params,xgtrain, 300,[(xgvalid, 'valid')],
                              maximize=True, early_stopping_rounds=50,verbose_eval=10)

    xgbtest=xgb.DMatrix(test_df[predictors_sorted].values)
    print("\nModel Report")
    print("bst1.best_iteration: ", trained_model.best_iteration)
    
    print('[{}]: Training time for  XGB'.format(time.time() - start_time))
    
    ax = xgb.plot_importance(trained_model, max_num_features=300)
    plt.gcf().savefig('test%d.png'%(fileno), dpi=600)
    plt.show()

    print("Predicting...")
    sub['is_attributed'] = trained_model.predict(xgbtest,ntree_limit=trained_model.best_ntree_limit)
#     if not debug:
#         print("writing...")
    sub.to_csv('sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    print("done...")
    return sub

nrows=184903891-1
nchunk=12000000
val_size=1000000

frm=nrows-84903891
if debug:
    frm=0
    nchunk=100000
    val_size=10000

to=frm+nchunk

sub=DO(frm,to,FILENO)
