import numpy as np

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_absolute_error

import lightgbm as lgb

import matplotlib.pyplot as plt
np.random.seed(42)



train = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv',

                    parse_dates=['week_start_date'])

test = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv',

                   parse_dates=['week_start_date'])

train['cases'] = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv', usecols=[3])
train['cases'] = np.log1p(train['cases'])
#NULL ANALYSIS

if train.isnull().sum().any():

    null_cnt = train.isnull().sum().sort_values()

    print('TRAIN null count:', null_cnt[null_cnt > 0])



if test.isnull().sum().any():

    null_cnt = test.isnull().sum().sort_values()

    print('TEST null count:', null_cnt[null_cnt > 0])
train.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

test.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
ltrain = train.shape[0]

df = pd.concat([train,test], sort=False)  

print('Combined df shape:{}'.format(df.shape))
# drop constant column

constant_column = [col for col in df.columns if df[col].nunique() == 1]

print('drop CONSTANT columns:', constant_column)

df.drop(constant_column, axis=1, inplace=True)



corr_matrix = df.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]

del upper



print('drop SIMILAR columns:', to_drop)

df.drop(to_drop,1,inplace=True)
df['month'] = df.week_start_date.dt.month
train = df[:ltrain].copy()

test = df[ltrain:].copy()

del df
y_sj = train[train.city=='sj']['cases']

y_iq = train[train.city=='iq']['cases']



train_sj = train[train.city=='sj'].drop('city',1)

train_iq = train[train.city=='iq'].drop('city',1)



test_sj = test[test.city=='sj'].drop('city',1)

test_iq = test[test.city=='iq'].drop('city',1)
def train_and_predict(tr,te,y_):

    

    def create_features(tr_,te_,idx=[]):

        df = pd.concat([tr_,te_], sort=False).set_index('week_start_date')

        if len(idx)>0:

            means_w = tr_.iloc[idx].groupby(['weekofyear'])['cases'].mean().to_frame('meanw').reset_index()

            medians_w = tr_.iloc[idx].groupby(['weekofyear'])['cases'].median().to_frame('medw').reset_index()

            means_m = tr_.iloc[idx].groupby(['month'])['cases'].mean().to_frame('meanm').reset_index()

            medians_m = tr_.iloc[idx].groupby(['month'])['cases'].median().to_frame('medm').reset_index()

        else:

            means_w = tr_.groupby(['weekofyear'])['cases'].mean().to_frame('meanw').reset_index()

            medians_w = tr_.groupby(['weekofyear'])['cases'].median().to_frame('medw').reset_index()

            means_m = tr_.groupby(['month'])['cases'].mean().to_frame('meanm').reset_index()

            medians_m = tr_.groupby(['month'])['cases'].median().to_frame('medm').reset_index()

        df = pd.merge(df, means_w, how='left', on=['weekofyear'])

        df = pd.merge(df, medians_w, how='left', on=['weekofyear'])

        df = pd.merge(df, means_m, how='left', on=['month'])

        df = pd.merge(df, medians_m, how='left', on=['month'])

        df = df.drop(['year','month','cases'],1)

        return df[:tr_.shape[0]], df[tr_.shape[0]:]



    x_train, x_test = create_features(tr,te)

    y_train = y_.clip(0,4).values



    params = {

        'max_depth': 4,

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'regression_l1',

        'metric': 'mae',

        'num_leaves': 9,

        'learning_rate': 0.05,

        'feature_fraction': 0.8,

        'bagging_fraction': 0.8,

        'bagging_freq': 10,

        'lambda_l1': 0.06,

        'lambda_l2': 0.1,

        'random_state': 42,

        'verbose': 0

    }





    n_f = 12 # Folds



    tscv = TimeSeriesSplit(n_splits=n_f)

    fds = tscv.split(x_train)

    lgbtrain = lgb.Dataset(x_train,y_train)

    

    hist = lgb.cv(params,

                lgbtrain,

                num_boost_round=2000,

                folds=fds,

                shuffle=False,

                metrics=['l1'],

                early_stopping_rounds=100,

                verbose_eval=0, 

                show_stdv=False, 

                seed=42)

    

    best_iter = np.array(hist['l1-mean']).argmin()

    

    gbm = lgb.train(params, lgbtrain, best_iter)    

    te['cases'] = gbm.predict(x_test, num_iteration=gbm.best_iteration)



    lgb.plot_importance(gbm,importance_type='gain',figsize=(10, 15))

    plt.show()



    preds = np.zeros([x_test.shape[0]])

    oof = np.zeros([x_train.shape[0]])

    

    for tr_i, vl_i in tscv.split(x_train):   

        x_train, x_test = create_features(tr,te,tr_i)

        lgb_tr = lgb.Dataset(x_train.iloc[tr_i],y_train[tr_i])

        lgb_vl = lgb.Dataset(x_train.iloc[vl_i],y_train[vl_i])

        gbm = lgb.train(params, lgb_tr, best_iter, valid_sets=[lgb_vl], verbose_eval=0)    

        oof[vl_i] = gbm.predict(x_train.iloc[vl_i], num_iteration=best_iter)

        preds += gbm.predict(x_test, num_iteration=best_iter) / n_f



    te['cases'] = preds



    lt = int(len(tr)/200+0.5)

    for t in range(lt):

        preds = np.zeros([x_test.shape[0]])    

        x_train, x_test = create_features(tr,te)        

        preds = gbm.predict(x_test, num_iteration=gbm.best_iteration)

        print('Erro preds :',mean_absolute_error(te['cases'].values,np.floor(preds)))

        te['cases'] = 0.5*te['cases'] + 0.5*preds



    lf = int(len(tr)/n_f)    

    print('oof:',mean_absolute_error(y_[lf:],np.floor(oof[lf:])))



    return preds, oof

preds_sj, oof_sj = train_and_predict(train_sj,test_sj,y_sj)
test_sj['cases'] = preds_sj

plot_all = pd.concat([train_sj, test_sj], sort=True).set_index('week_start_date')

fig, ax = plt.subplots(1, 1)

_ = plot_all.iloc[:train_sj.shape[0]].plot(y='cases',ax=ax,color='b',legend=False, figsize=(15, 5))

_ = plot_all.iloc[train_sj.shape[0]:].plot(y='cases',ax=ax,color='r',legend=False, figsize=(15, 5))
preds_iq, oof_iq = train_and_predict(train_iq,test_iq,y_iq)
test_iq['cases'] = preds_iq

plot_all = pd.concat([train_iq, test_iq], sort=True).set_index('week_start_date')

fig, ax = plt.subplots(1, 1)

_ = plot_all.iloc[:train_iq.shape[0]].plot(y='cases',ax=ax,color='b',legend=False, figsize=(15, 5))

_ = plot_all.iloc[train_iq.shape[0]:].plot(y='cases',ax=ax,color='r',legend=False, figsize=(15, 5))
preds = np.concatenate((preds_sj, preds_iq), axis=0)
test['total_cases'] = np.round(np.expm1(preds),0)

test[['city','year','weekofyear','total_cases']].to_csv('submission_lgb.csv', float_format='%.0f', index=False)