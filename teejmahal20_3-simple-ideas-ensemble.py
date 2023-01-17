import numpy as np 

import pandas as pd

from sklearn import *

from sklearn.metrics import f1_score

import lightgbm as lgb

import xgboost as xgb

from catboost import Pool,CatBoostRegressor

import matplotlib.pyplot as plt

import seaborn as sns

import time

import datetime



sns.set_style("whitegrid")



from sklearn.model_selection import KFold



#Constants

ROW_PER_BATCH = 500000
#Loading data

train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')





print('Shape of train is ',train.shape)

print('Shape of test is ',test.shape)
train['batch'] = 0



for i in range(0, train.shape[0]//ROW_PER_BATCH):

    train.iloc[i * ROW_PER_BATCH: (i+1) * ROW_PER_BATCH,3] = i
plt.figure(figsize=(26, 22))

plt.subplots_adjust(top=1.2, hspace = 0.8)

for i, b in enumerate(train['batch'].unique()):

    plt.subplot(5, 2, i + 1)

    plt.plot(train.loc[train['batch'] == b, ['signal']], color='b')

    plt.title(f'Batch: {b}')

    plt.plot(train.loc[train['batch'] == b, ['open_channels']], color='r')

    plt.legend(['signal', 'open_channels'], loc=(0.875, 0.9))

    plt.grid(False)
plt.figure(figsize=(26, 22))

colors = ['red','red','blue','pink','gold','brown','blue','pink','brown','gold']

for i, b in enumerate(train['batch'].unique()):

    plt.subplot(5, 2, i + 1)

    train.iloc[i*ROW_PER_BATCH : (i+1)*ROW_PER_BATCH]['open_channels'].value_counts().plot(kind='bar',color= colors[i])

    plt.title(f'Batch: {b}')
df = pd.DataFrame()

plt.figure(figsize=(26, 22))

plt.subplots_adjust(top=1.2, hspace = 0.8)

colors = ['red','red','blue','pink','gold','brown','blue','pink','brown','gold']

for i in train['batch'].unique():

    df[ f'batch:{i}' ] = train.iloc[i*ROW_PER_BATCH : (i+1)*ROW_PER_BATCH].reset_index().signal 

    plt.subplot(5, 2, i + 1)

    

    sns.distplot(df[f'batch:{i}'], color= colors[i]).set_title(f"(Signal) median: {df[f'batch:{i}'].median():.2f}")



    

plt.figure(figsize=(16,10))



sns.boxplot(x="variable",y="value",data=pd.melt(df))
test['batch'] = 0



for i in range(0, test.shape[0]//ROW_PER_BATCH):

    test.iloc[i * ROW_PER_BATCH: (i+1) * ROW_PER_BATCH,2] = i
mean_signal = []

df = pd.DataFrame()

templis = {}

plt.figure(figsize=(26, 22))

plt.subplots_adjust(top=1.2, hspace = 0.8)

colors = ['red','red','blue','pink','gold','brown','blue','pink','brown','gold']

for i in test['batch'].unique():

    df[ f'batch:{i}' ] = test.iloc[i*ROW_PER_BATCH : (i+1)*ROW_PER_BATCH].reset_index().signal 

    plt.subplot(5, 2, i + 1)

    sns.distplot(df[f'batch:{i}'], color= colors[i]).set_title(f"(Signal) median: {df[f'batch:{i}'].median():.2f}")
plt.figure(figsize=(8,5))

sns.boxplot(x="variable",y="value",data=pd.melt(df))
train['type'] = 0

for i in range(train['batch'].nunique()):

    median = train.iloc[i*ROW_PER_BATCH : (i+1) * ROW_PER_BATCH].signal.median()

    if (median < 0):

        train.iloc[i*ROW_PER_BATCH : (i+1) * ROW_PER_BATCH, train.columns.get_loc('type')] = 0

    else:

        train.iloc[i*ROW_PER_BATCH : (i+1) * ROW_PER_BATCH, train.columns.get_loc('type')] = 1

        

test['type'] = 0



ROW_PER_BATCH = 100000



for i in range(test['batch'].nunique()):

    median = test.iloc[i*ROW_PER_BATCH : (i+1)*ROW_PER_BATCH].signal.median()

    if (median < 0):

        test.iloc[i*ROW_PER_BATCH : (i+1) * ROW_PER_BATCH, test.columns.get_loc('type')] = 0

    else:

        test.iloc[i*ROW_PER_BATCH : (i+1) * ROW_PER_BATCH, test.columns.get_loc('type')] = 1    
plt.figure(figsize=(20,5))

plt.plot( train.signal[500000:1000000][::100] )

plt.show()
a = 500000; b = a *2

print( 'Before: mean: {} std: {} median: {}'.format( train.signal[a:b].mean(), train.signal[a:b].std(),train.signal[a:b].median() ) )



a=500000; b=600000

train['signal_undrifted'] = train.signal

train.loc[train.index[a:b],'signal_undrifted'] = train.signal[a:b].values - 3*(train.time.values[a:b] - 50)/10.



a = 500000; b = a *2

print( 'After: mean: {} std: {} median: {}'.format( train.signal_undrifted[a:b].mean(), train.signal_undrifted[a:b].std(),train.signal_undrifted[a:b].median() ) )
plt.figure(figsize=(20,6))

sns.distplot(train.signal[500000:1000000],color='r')

sns.distplot(train.signal_undrifted[500000:1000000],color='g' ).set(xlabel="Signal")

plt.legend(labels=['Original Signal','Undrifted Signal'])



def f(x,low,high,mid): return -((-low+high)/625)*(x-mid)**2+high -low



# CLEAN TRAIN BATCH 7

batch = 7; a = 500000*(batch-1); b = 500000*batch

train.loc[train.index[a:b],'signal_undrifted'] = train.signal.values[a:b] - f(train.time[a:b].values,-1.817,3.186,325)

# CLEAN TRAIN BATCH 8

batch = 8; a = 500000*(batch-1); b = 500000*batch

train.loc[train.index[a:b],'signal_undrifted'] = train.signal.values[a:b] - f(train.time[a:b].values,-0.094,4.936,375)

# CLEAN TRAIN BATCH 9

batch = 9; a = 500000*(batch-1); b = 500000*batch

train.loc[train.index[a:b],'signal_undrifted'] = train.signal.values[a:b] - f(train.time[a:b].values,1.715,6.689,425)

# CLEAN TRAIN BATCH 10

batch = 10; a = 500000*(batch-1); b = 500000*batch

train.loc[train.index[a:b],'signal_undrifted'] = train.signal.values[a:b] - f(train.time[a:b].values,3.361,8.45,475)
plt.figure(figsize=(20,5))

sns.lineplot(train.time[::1000],train.signal[::2000],color='r').set_title('Training Batches 7-10 with Parabolic Drift')

#plt.figure(figsize=(20,5))

g = sns.lineplot(train.time[::1000],train.signal_undrifted[::2000],color='g').set_title('Training Batches 7-10 without Parabolic Drift')

plt.legend(title='Train Data',loc='upper left', labels=['Original Signal', 'UnDrifted Signal'])

plt.show(g)



test['signal_undrifted'] = test.signal



# REMOVE BATCH 1 DRIFT

start=500

a = 0; b = 100000

test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

start=510

a = 100000; b = 200000

test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

start=540

a = 400000; b = 500000

test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.



# REMOVE BATCH 2 DRIFT

start=560

a = 600000; b = 700000

test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

start=570

a = 700000; b = 800000

test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

start=580

a = 800000; b = 900000

test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.



# REMOVE BATCH 3 DRIFT

def f(x):

    return -(0.00788)*(x-625)**2+2.345 +2.58

a = 1000000; b = 1500000

test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - f(test.time[a:b].values)
plt.figure(figsize=(20,5))

sns.lineplot(test.time[::1000],test.signal[::1000],color='r').set_title('Training Batches 7-10 with Parabolic Drift')

#plt.figure(figsize=(20,5))

g = sns.lineplot(test.time[::1000],test.signal_undrifted[::1000],color='g').set_title('Training Batches 7-10 without Parabolic Drift')

plt.legend(title='Test Data',loc='upper right', labels=['Original Signal', 'UnDrifted Signal'])

plt.show(g)



%%time



def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 50_000

    df['batch_index'] = df.index  - (df.batch * 50_000)

    df['batch_slices'] = df['batch_index']  // 5_000

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal_undrifted'].mean()

        d['median'+c] = df.groupby([c])['signal_undrifted'].median()

        d['max'+c] = df.groupby([c])['signal_undrifted'].max()

        d['min'+c] = df.groupby([c])['signal_undrifted'].min()

        d['std'+c] = df.groupby([c])['signal_undrifted'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.min(np.abs(x)))

        for v in d:

            df[v] = df[c].map(d[v].to_dict())

        df['range'+c] = df['max'+c] - df['min'+c]

        df['maxtomin'+c] = df['max'+c] / df['min'+c]

        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2

    

    #add shifts

    df['signal_shift_+1'] = [0,] + list(df['signal_undrifted'].values[:-1])

    df['signal_shift_-1'] = list(df['signal_undrifted'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan



    # add shifts_2 - https://www.kaggle.com/vbmokin/ion-switching-advanced-fe-lgb-confmatrix

    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal_undrifted'].values[:-2])

    df['signal_shift_-2'] = list(df['signal_undrifted'].values[2:]) + [0] + [1]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==1].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-2'][i] = np.nan

    for i in df[df['batch_index']==49998].index:

        df['signal_shift_-2'][i] = np.nan 

        

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal_undrifted', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal_undrifted']

        

    return df



train = features(train)

test = features(test)
def f1_score_calc(y_true, y_pred):

    return f1_score(y_true, y_pred, average="macro")



def lgb_Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = f1_score(labels, preds, average="macro")

    return ('KaggleMetric', score, True)





def train_model_classification(X, X_test, y, params, model_type='lgb', eval_metric='f1score',

                               columns=None, plot_feature_importance=False, model=None,

                               verbose=50, early_stopping_rounds=200, n_estimators=2000):



    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {

                    'f1score': {'lgb_metric_name': lgb_Metric,}

                   }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros(len(X) )

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test)))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    '''for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]'''

            

    if True:        

        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.3, random_state=7)    

            

        if model_type == 'lgb':

            #model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)

            #model.fit(X_train, y_train, 

            #        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

            #       verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            model = lgb.train(params, lgb.Dataset(X_train, y_train),

                              n_estimators,  lgb.Dataset(X_valid, y_valid),

                              verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds, feval=lgb_Metric)

            

            

            preds = model.predict(X, num_iteration=model.best_iteration) #model.predict(X_valid) 



            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            

        if model_type == 'xgb':

            train_set = xgb.DMatrix(X_train, y_train)

            val_set = xgb.DMatrix(X_valid, y_valid)

            model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 

                                     verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)

            

            preds = model.predict(xgb.DMatrix(X)) 



            y_pred = model.predict(xgb.DMatrix(X_test))

            



        if model_type == 'cat':

            # Initialize CatBoostRegressor

            model = CatBoostRegressor(params)

            # Fit model

            model.fit(X_train, y_train)

            # Get predictions

            y_pred_valid = np.round(np.clip(preds, 0, 10)).astype(int)



            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)



 

        oof = preds

        

        scores.append(f1_score_calc(y, np.round(np.clip(preds,0,10)).astype(int) ) )



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    #prediction /= folds.n_splits

    

    print('FINAL score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    result_dict['model'] = model

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict
#Let's uses 'signal_undrifted' instead of 'signal'

good_columns = [c for c in train.columns if c not in ['time', 'signal','open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]



X = train[good_columns].copy()

y = train['open_channels']

X_test = test[good_columns].copy()
del train, test
params_xgb = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',

          'eval_metric':'rmse'}



result_dict_xgb = train_model_classification(X=X[0:500000*8-1], X_test=X_test, y=y[0:500000*8-1], params=params_xgb, model_type='xgb', eval_metric='f1score', plot_feature_importance=False,

                                                      verbose=50, early_stopping_rounds=250)
#params_cat = {'task_type': "CPU",'iterations':1000,'learning_rate':0.1,'random_seed': 42,'depth':2}



#result_dict_cat = train_model_classification(X=X[0:500000*8-1], X_test=X_test, y=y[0:500000*8-1], params=params_cat, model_type='cat', eval_metric='f1score', plot_feature_importance=False)
params_lgb = {'learning_rate': 0.1, 'max_depth': 7, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1}



result_dict_lgb = train_model_classification(X=X[0:500000*8-1], X_test=X_test, y=y[0:500000*8-1], params=params_lgb, model_type='lgb', eval_metric='f1score', plot_feature_importance=False,

                                                      verbose=50, early_stopping_rounds=250, n_estimators=3000)
booster = result_dict_lgb['model']



fi = pd.DataFrame()

fi['importance'] = booster.feature_importance(importance_type='gain')

fi['feature'] = booster.feature_name()



best_features = fi.sort_values(by='importance', ascending=False)[:20]





plt.figure(figsize=(16, 12));

sns.barplot(x="importance", y="feature", data=best_features);

plt.title('LGB Features (avg over folds)');
def weight_opt(oof_lgb, oof_xgb, y_true):

    weight_lgb = np.inf

    best_f1 = np.inf

    

    for i in np.arange(0, 1.01, 0.10):

        combined_oof = i * oof_lgb + (1-i) * oof_xgb

        blend = np.round(np.clip(combined_oof,0,10)).astype(int) 

        f1_blend = metrics.f1_score(y_true, blend, average = 'macro' )

        if np.mean(f1_blend) < best_f1:

            best_f1 = np.mean(f1_blend)

            weight_lgb = round(i, 2)

            

        print(str(round(i, 2)) + ' : mean F1 (Blend) is ', round(np.mean(f1_blend), 6))

        

    print('-'*36)

    print('Best weight for LGB: ', weight_lgb)

    print('Best weight for XGB: ', round(1-weight_lgb, 2))

    print('Best mean F1 (Blend): ', round(best_f1, 6))

    

    return weight_lgb, round(1-weight_lgb, 2)
weight_lgb, weight_xgb = weight_opt(result_dict_lgb['oof'], result_dict_xgb['oof'], y[0:500000*8-1])
preds_ensemble = 0.50 * result_dict_lgb['prediction'] + 0.50 * result_dict_xgb['prediction']
sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sub['open_channels'] =  np.array(np.round(preds_ensemble,0), np.int) 



sub.to_csv('submission_unshifted_70p.csv', index=False, float_format='%.4f')

sub.head(10)