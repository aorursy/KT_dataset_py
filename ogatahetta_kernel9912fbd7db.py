# ライブラリのインポート

import numpy as np

import pandas as pd



import matplotlib as mpl

import matplotlib.pyplot as plt



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.datasets import make_classification

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GroupShuffleSplit

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_boston

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



from tqdm import tqdm_notebook as tqdm
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)
train = pd.read_csv('../input/train.csv',parse_dates=['earliest_cr_line','issue_d'])

test = pd.read_csv('../input/test.csv',parse_dates=['earliest_cr_line','issue_d'])

spi = pd.read_csv('../input/spi.csv',parse_dates=['date'])

gdp = pd.read_csv('../input/US_GDP_by_State.csv')

ll = pd.read_csv('../input/statelatlong.csv')
pd.DataFrame({'dtype':train.dtypes, 'unique':train.nunique(),'count': train.notnull().sum(),'nulls':train.isnull().sum()})
#カラム名の修正

train_col = train.copy()

train_col.columns = ['ID', 'loan_amnt', 'installment', 'grade', 'sub_grade', 'emp_title',

       'emp_length', 'home_ownership', 'annual_inc', 'issue_d', 'purpose',

       'title', 'zip_code', 'state', 'dti', 'delinq_2yrs',

       'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq',

       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',

       'revol_util', 'total_acc', 'initial_list_status',

       'collections_12_mths_ex_med', 'mths_since_last_major_derog',

       'application_type', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal','loan_condition']

test_col = test.copy()

test_col.columns = ['ID', 'loan_amnt', 'installment', 'grade', 'sub_grade', 'emp_title',

       'emp_length', 'home_ownership', 'annual_inc', 'issue_d', 'purpose',

       'title', 'zip_code', 'state', 'dti', 'delinq_2yrs',

       'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq',

       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',

       'revol_util', 'total_acc', 'initial_list_status',

       'collections_12_mths_ex_med', 'mths_since_last_major_derog',

       'application_type', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal']
date_cols = ['earliest_cr_line','issue_d']

txt_cols = ['emp_title','title']

cat_cols = list(test_col.drop(date_cols+txt_cols,axis=1).select_dtypes(include=[object]).columns)

num_cols = list(test_col.drop('ID',axis=1).select_dtypes(include=[int,float]).columns)
#prep

target = train_col.loan_condition.copy()

df0 = train_col.drop(['loan_condition'],axis = 1).copy()

df0['year'],df0['month'] = df0.issue_d.dt.year, df0.issue_d.dt.month

df0['ecl_year'],df0['ecl_month'] = df0.earliest_cr_line.dt.year, df0.earliest_cr_line.dt.month



dt0 = test_col.copy()

dt0['year'],dt0['month'] = dt0.issue_d.dt.year,dt0.issue_d.dt.month

dt0['ecl_year'],dt0['ecl_month'] = dt0.earliest_cr_line.dt.year, dt0.earliest_cr_line.dt.month



ds0 = spi.copy()

ds0['year'], ds0['month'] = ds0.date.dt.year,ds0.date.dt.month



dg0 = gdp.merge(ll[['State', 'City']],left_on = 'State',right_on='City',how='left').drop(['State_x','City'],axis = 1)

dg0.columns = ['state_and_local_spending', 'gross_state_product','real_state_growth_percent', 'population_million', 'year', 'state']



dl0 = ll.drop('City',axis = 1).copy()

dl0.columns = ['state', 'lat', 'lon']
#want to know trend by region, purpose

df0_agg = pd.concat([df0[['issue_d','year','month','state','purpose']],target],axis = 1).groupby(['issue_d','year','month','state','purpose'])['loan_condition'].agg({'sum':np.sum,'count':"count"}).reset_index()

df0_agg.columns = ['issue_d','year', 'month', 'state', 'purpose', 'bad_count','app_count']



ds0_agg = ds0.groupby(['year','month'])['close'].agg({'spi_mean': np.mean, 'spi_median':np.median, 'spi_var':np.var, 'spi_max': np.max, 'spi_min': np.min}).reset_index()

ds0_agg.columns = ['year','month','spi_mean','spi_median','spi_var','spi_max','spi_min']



target_agg = df0_agg.merge(ds0_agg, on=['year','month'] ,how='left').merge(dg0, on=['year','state'] ,how='left')
temp = target_agg[['year','month','state','purpose','bad_count','app_count']].groupby(['year','month','state','purpose']).sum().reset_index()
by_year = temp.groupby(['year'])[['bad_count','app_count']].sum().reset_index()

by_year['bad_rate'] = by_year.bad_count/by_year.app_count

by_year
by_state = temp.groupby(['state'])[['bad_count','app_count']].sum().reset_index()

by_state['bad_rate'] = by_state.bad_count/by_state.app_count

by_state
by_year_state = temp.groupby(['state','year'])[['bad_count','app_count']].sum().reset_index()

by_year_state['bad_rate'] = (by_year_state.bad_count/by_year_state.app_count)
set(dt0.state.unique()) - set(by_year_state[by_year_state.year >= 2015].state.unique())
dt0[dt0.state == 'ID'].describe()
def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series=None, 

                  tst_series=None, 

                  target=None, 

                  min_samples_leaf=1, 

                  smoothing=1,

                  noise_level=0):

    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """ 

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data

    prior = target.mean()

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
def missing_value_imputation(df,date_cols,cat_cols,num_cols):

    df['date_null_count'] = df[date_cols].isnull().sum(axis = 1)

    df['cat_null_count'] = df[cat_cols].isnull().sum(axis = 1)  

    df['num_null_count'] = df[num_cols].isnull().sum(axis = 1)  

    df['null_count'] = df.isnull().sum(axis = 1)

    for c in date_cols:

        df[c+'_is_missing'] = df[c].isnull()

    for c in cat_cols:

        df[c+'_is_missing'] = df[c].isnull()    

    for c in num_cols:

        df[c+'_is_missing'] = df[c].isnull()

        

    df[cat_cols] = df[cat_cols].fillna('NaN')

    df[txt_cols] = df[txt_cols].fillna("NaN")

    

    return df

    

def month_encode(df, col):

    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())

    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())

    

    return df



def merge_tables(df,dl,ds,dg):

    ds_agg = ds.groupby(['year','month'])['close'].agg({'spi_mean': np.mean, 'spi_median':np.median, 'spi_var':np.var, 'spi_max': np.max, 'spi_min': np.min}).reset_index()

    ds_agg.columns = ['year','month','spi_mean','spi_median','spi_var','spi_max','spi_min']

    df = df.merge(ds_agg, on=['year','month'],how='left').merge(dg, on=['year','state'],how='left').merge(dl, on=['state'],how='left')

#    df = df.merge(dl, on=['state'],how='left')

    

    return df



def enc_grades(df,glist,sglist):

    df['grade_num'] = df.grade.apply(lambda x:glist.index(x))

    df['sub_grade_num'] = df.sub_grade.apply(lambda x:sglist.index(x))

    

    return df



def num_feature(df, base, base_name, rank_target):

    df['loan_amnt_times_sub_grade_num'] = df.loan_amnt * df.sub_grade_num

    df['loan_amnt_times_grade_num'] = df.loan_amnt * df.grade_num

    df['installment_times_sub_grade_num'] = df.installment * df.sub_grade_num

    df['monthly_debt'] = (df.dti/100)*df.annual_inc

    df['total_monthly_debt'] = df.monthly_debt + df.installment

    df['total_dti'] = df.total_monthly_debt / df.annual_inc

    df['debt_ratio'] = df.monthly_debt / df.installment

    df['monthly_debt_times_grade'] = df.monthly_debt * df.grade_num

    df['total_monthly_debt_times_grade'] = df.total_monthly_debt * df.grade_num

    

    for b in base:

        for r in rank_target:

            df[r+'_rank_by_'+base_name[base.index(b)]] = df.groupby(b)[r].rank() 

    

    return df



def cat_feature(df,dt,enc_cols):

    for enc_col in enc_cols:

        df[enc_col + '_target_enc'], dt[enc_col + '_target_enc'] = target_encode(trn_series=df[enc_col], 

                      tst_series=dt[enc_col], 

                      target=df['sub_grade_num'], 

                      min_samples_leaf=1, 

                      smoothing=1,

                      noise_level=0)

    return df, dt



def txt_feature(df, txt_col):

    df[txt_col + '_phrase_len'] = [len(t) for t in df[txt_col]]

    df[txt_col + '_phrase_len'] = [t.count(' ')+1 for t in df[txt_col]]



    return df
df,dt,ds,dg,dl = df0.copy(), dt0.copy(), ds0.copy(), dg0.copy(), dl0.copy()
df,dt = missing_value_imputation(df,date_cols,cat_cols,num_cols),missing_value_imputation(dt,date_cols,cat_cols,num_cols)
df,dt = month_encode(df, 'month'), month_encode(dt, 'month')

df,dt = month_encode(df, 'ecl_month'), month_encode(dt, 'ecl_month')
glist = sorted(df.grade.unique())

sglist = sorted(df.sub_grade.unique())



df,dt = enc_grades(df,glist,sglist) ,enc_grades(dt,glist,sglist)
base = ['state','year',['year','state']]

base_name = ['state','year','ys']

rank_target = ['loan_amnt','annual_inc','installment','dti']

df,dt = num_feature(df, base, base_name, rank_target) ,num_feature(dt, base, base_name, rank_target)
#target enc

enc_cols = [x for x in cat_cols if x not in ['grade','sub_grade']]

df,dt = cat_feature(df,dt,enc_cols)
df,dt = merge_tables(df,dl,ds,dg),merge_tables(dt,dl,ds,dg)
txt_col = txt_cols[0]

df,dt = txt_feature(df,txt_col),txt_feature(dt,txt_col)



txt_col = txt_cols[1]

df,dt = txt_feature(df,txt_col),txt_feature(dt,txt_col)
tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True)

emp = tv.fit_transform(pd.concat([df[txt_cols[0]],dt[txt_cols[0]]]))

title = tv.fit_transform(pd.concat([df[txt_cols[1]],dt[txt_cols[1]]]))

from scipy.sparse import hstack

sparse = hstack([emp, title]).tocsr()



clft = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

   importance_type='split', learning_rate=0.1, max_depth=-1,

   min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

   n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

   random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

   subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

df['tfidf'] = cross_val_predict(clft, sparse[:len(df)], target.as_matrix(), cv = 5, method='predict_proba')[:,1]



clft.fit(sparse[:len(df)], target.as_matrix())

dt['tfidf'] = clft.predict_proba(sparse[len(df):])[:,1]
# orig_cols_to_use = num_cols + ['state']

# cols_use = orig_cols_to_use + list(df.columns[train.shape[1]-1:])

# df_av, dt_av = df[cols_use], dt[cols_use]

df_av, dt_av = df, dt

dv = pd.concat([df_av, dt_av])

dv['in_train'] = np.concatenate([

    np.zeros(df_av.shape[0]), 

    np.ones(dt_av.shape[0])

])

dv['ys'] = dv.year.astype(str) + dv.state



#yearの情報はどうしようもないので避ける。testにありそうな確率を割り振りたいので、なるべくオリジナルに近い特徴量のみ投入する

in_use = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',

       'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',

       'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',

       'collections_12_mths_ex_med', 'mths_since_last_major_derog',

       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal',

          'month', 'ecl_year', 'ecl_month',

          'grade_num','sub_grade_num',

          'emp_length_target_enc', 'home_ownership_target_enc',

       'purpose_target_enc', 'zip_code_target_enc', 'state_target_enc',

       'initial_list_status_target_enc', 'application_type_target_enc',

         'lat', 'lon',

       'emp_title_phrase_len', 'title_phrase_len']



X_dv = dv[in_use]

y_dv = dv.in_train

X_train, y_train = X_dv, y_dv



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
%%time

X_train, y_train = X_dv, y_dv

scores = []



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_ = X_train.iloc[train_ix,:]

    y_train_  = y_train.iloc[train_ix]

    X_val = X_train.iloc[test_ix,:]

    y_val = y_train.iloc[test_ix]

    

    clfv = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

       importance_type='split', learning_rate=0.1, max_depth=-1,

       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

       n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

       random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

       subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clfv.fit(X_train_, y_train_)

    y_pred = clfv.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
import eli5

eli5.show_weights(clfv)
%%time



clfa = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

   importance_type='split', learning_rate=0.1, max_depth=-1,

   min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

   n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

   random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

   subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



y_pred = cross_val_predict(clfa, X_train, y_train, cv = 5, method='predict_proba')[:,1]

df['alike_test'] = y_pred[:len(df)]

dt['alike_test'] = y_pred[len(df):]
orig_cols_to_use = num_cols + ['state']



cols_use = orig_cols_to_use + list(df.columns[train.shape[1]-1:])

df_use, dt_use = df[cols_use], dt[cols_use]

df_use['ys'] = df_use.year.astype(str) + df_use.state



not_in_use = ['state', 'year', 'ys']

X_train = df_use.drop([c for c in not_in_use if c in df_use.columns],axis = 1)

X_test = dt_use.drop([c for c in not_in_use if c in dt_use.columns],axis = 1)

y_train = target[target.index.isin(X_train.index)]
%%time

scores = []



gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=71)

for i, (train_ix, test_ix) in enumerate(tqdm(gss.split(X_train, y_train, groups=df_use.ys))):

    X_train_, y_train_ = X_train.iloc[train_ix,:], y_train.iloc[train_ix]

    X_val, y_val = X_train.iloc[test_ix,:], y_train.iloc[test_ix]

    

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

       importance_type='split', learning_rate=0.1, max_depth=-1,

       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

       n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

       random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

       subsample=1.0, subsample_for_bin=200000, subsample_freq=0,weight_column='alike_test')

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
sum(scores)/len(scores)
import eli5

eli5.show_weights(clf)
# 全データで再学習し、testに対して予測する

clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:,1] # predict_probaで確率を出力する
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')