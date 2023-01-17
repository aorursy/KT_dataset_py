# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import gc

from pathlib import Path

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('../'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# copy-paste

import numpy as np

import pandas as pd

import datetime

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import TruncatedSVD

import warnings

import math

warnings.filterwarnings('ignore')

np.random.seed(4590)
%%time

PATH = Path('../input/elo-merchant-category-recommendation')

df_train = pd.read_csv(PATH/'train.csv', parse_dates=['first_active_month']);

df_test = pd.read_csv(PATH/'test.csv', parse_dates=['first_active_month']);
%%time

df_hist_trans = pd.read_csv(PATH/'historical_transactions.csv', parse_dates=['purchase_date']);

df_new_merch_trans = pd.read_csv(PATH/'new_merchant_transactions.csv', parse_dates=['purchase_date']);
# https://www.kaggle.com/fabiendaniel/elo-world

def reduce_mem_usage(df, verbose=True):

    prefixes = ['int', 'float']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = str(df[col].dtype)

        if not col_type.startswith('int') and not col_type.startswith('float'):

#             print('col_type:', col_type, 'not compressed')

            continue

        c_min = df[col].min()

        c_max = df[col].max()

        if col_type.startswith('int'):

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                df[col] = df[col].astype(np.int8)

            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                df[col] = df[col].astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                df[col] = df[col].astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                df[col] = df[col].astype(np.int64)  

        elif col_type.startswith('float'):

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

            else:

                df[col] = df[col].astype(np.float64)    

    if verbose:

        end_mem = df.memory_usage().sum() / 1024**2

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699/

# Same logic

def fillna_mode(df, cols, pipeline=False):

    for c in cols:

        df[c].fillna(df[c].mode()[0], inplace=True)

    return df if pipeline else None





def get_nan_col_names(df: pd.DataFrame, df_name='<not named>'):

    total = df.shape[0]

    missing_cols = []

    for c in df.columns:

        quo = (total - pd.notna(df[c]).sum())/total

        if quo != 0:

            missing_cols.append(c)

    print(df_name, 'MISSING COLS:', missing_cols)

    return missing_cols
reduce_mem_usage(df_train);

reduce_mem_usage(df_test);

reduce_mem_usage(df_hist_trans);

reduce_mem_usage(df_new_merch_trans);
#https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

def add_month_diff(df, pipeline=False):

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30

    df['month_diff'] += df['month_lag']

    return df if pipeline else None





# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

def pre_process_trans(df, date_col, date_formated=False, add_month=True):

    fillna_mode(df, cols=get_nan_col_names(df))

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0}).astype(np.int8)

    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype(np.int8)    

    df['category_2'] = df['category_2'].astype(np.int8)



    if not date_formated:

        df[date_col] = pd.to_datetime(df[date_col])

    df['year'] = df[date_col].dt.year

    df['weekofyear'] = df[date_col].dt.weekofyear

    df['month'] = df[date_col].dt.month

    df['dayofweek'] = df[date_col].dt.dayofweek

    df['weekend'] = (df[date_col].dt.weekday >= 5).astype(np.int8)

    df['hour'] = df[date_col].dt.hour

    if add_month:

        add_month_diff(df)





# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699/

def get_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]





# Refactor of https://www.kaggle.com/chauhuynh/my-first-kernel-3-699/

def custom_group_by(df: pd.DataFrame, df_name: str, agg_by: dict):

    for col in ['category_2','category_3']:

        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')

        aggs[col+'_mean'] = ['mean']    

    new_columns = get_new_columns(df_name, agg_by)

#     print("new_columns: \n", *new_columns, sep='\t\n')

    df_group = df.groupby('card_id').agg(agg_by)

    df_group.columns = new_columns

    df_group.reset_index(drop=False, inplace=True)

    df_group[df_name + '_purchase_date_diff'] = (

        df_group[df_name + '_purchase_date_max']

            - df_group[df_name + '_purchase_date_min']

        ).dt.days

    df_group[df_name + '_purchase_date_average'] = (

        df_group[df_name + '_purchase_date_diff']

            / df_group[df_name + '_card_id_size']

        )

    df_group[df_name + '_purchase_date_uptonow'] = (

        datetime.datetime.today()

        - df_group[df_name + '_purchase_date_max']

        ).dt.days



    return df_group
aggs = {}



# Count number of unique values at each column.

for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

    aggs[col] = ('nunique',)





aggs['purchase_amount'] = ('sum','max','min','mean','var')

aggs['installments'] = ('sum','max','min','mean','var')



# purchase range

aggs['purchase_date'] = ('max','min')



aggs['month_lag'] = ('max','min','mean','var')

aggs['month_diff'] = ('mean',)



# How many transactions:

# - were on the weekend, and the percentage

# - has category_1 as 1 (binary feature), and the percentage

# - were authorized, and its percentage



aggs['weekend'] = ('sum', 'mean')

aggs['category_1'] = ('sum', 'mean')

aggs['authorized_flag'] = ('sum', 'mean')



# How many purchases each card did?

aggs['card_id'] = ('size', )
_old_train = df_train.copy()

_odl_test = df_test.copy()
%%time

gps = []

for name, df in [('hist', df_hist_trans), ('new_hist', df_new_merch_trans)]:    

    pre_process_trans(df, date_col='purchase_date')

    df_group = custom_group_by(df, name, aggs.copy())

    gps.append(df_group)

    df_train = df_train.merge(df_group, on='card_id', how='left')

    df_test = df_test.merge(df_group, on='card_id', how='left')

prob = [

    'hist_purchase_date_max',

    'hist_purchase_date_min',

    'new_hist_purchase_date_max',

    'new_hist_purchase_date_min',

]



def post_process_df(df):

    global prob

    

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['fam_dayofweek'] = df['first_active_month'].dt.dayofweek

    df['fam_weekofyear'] = df['first_active_month'].dt.weekofyear

    df['fam_month'] = df['first_active_month'].dt.month

    df['fam_elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days



    df[prob[:2]] = df[prob[:2]].astype(np.int64) * 1e-9



    

    df['transactions_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']

    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']



    for f in ['feature_1','feature_2','feature_3']:

        order_label = df_train.groupby([f])['outliers'].mean()

        df_train[f] = df_train[f].map(order_label)

        df_test[f] = df_test[f].map(order_label)

    
print(df_train.hist_purchase_date_max.notna().sum())

print(df_train.hist_purchase_date_min.notna().sum())

print(df_train.new_hist_purchase_date_max.notna().sum())

print(df_train.new_hist_purchase_date_min.notna().sum())
df_train['outliers'] = (df_train.target < -30).astype(np.int8)
post_process_df(df_train)

post_process_df(df_test)
df_test.drop(['new_hist_purchase_date_max', 'new_hist_purchase_date_min'], axis=1,inplace=True)

df_train.drop(['new_hist_purchase_date_max', 'new_hist_purchase_date_min'], axis=1,inplace=True)
def LGB_Train(df_train,df_test,target,num_leaves):

    param = {'num_leaves': num_leaves,

             'min_data_in_leaf': 30, 

             'objective':'regression',

             'max_depth': -1,

             'learning_rate': 0.01,

             "min_child_samples": 20,

             "boosting": "gbdt",

             "feature_fraction": 0.9,

             "bagging_freq": 1,

             "bagging_fraction": 0.9 ,

             "bagging_seed": 11,

             "metric": 'rmse',

             "lambda_l1": 0.1,

             "verbosity": -1,

             "nthread": 4,

             "random_state": 4590}

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)

    oof = np.zeros(len(df_train))

    predictions_train = np.zeros(len(df_train)) 

    predictions_test = np.zeros(len(df_test))

    feature_importance_df = pd.DataFrame()

    df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'outliers', 'target']]



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):

        print("fold {}".format(fold_))

        trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

        val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



        num_round = 10000

        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)

        oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)



        fold_importance_df = pd.DataFrame()

        fold_importance_df["Feature"] = df_train_columns

        fold_importance_df["importance"] = clf.feature_importance()

        fold_importance_df["fold"] = fold_ + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



        predictions_test += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

        predictions_train += clf.predict(df_train[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits



    return {'predictions_test':predictions_test,'predictions_train':predictions_train,'feature_importances':feature_importance_df}

    
#A dataframe containing only numerical values

df_num=df_train[[col for col in df_train.columns if not col in ['target','outliers']]].select_dtypes(include='number')



#NaN/Missing values will be replaced with zeroes

# df_num.fillna(0,inplace=True)

fillna_mode(df_num, df_num.columns)





decomp = TruncatedSVD(n_components=len(df_num.columns.values)-1,random_state=1)

decomp.fit(df_num)
plot=sns.lineplot(x=range(10),y=decomp.singular_values_[:10])

plot.set_title("Eigenvalues in descending order")

plot.set(ylabel="Eigenvalue",xlabel="Vector")

plt.show()

plot=sns.lineplot(x=range(3,len(decomp.singular_values_)),y=decomp.singular_values_[3:])

plot.set_title("Eigenvalues [from 3rd on] in descending order")

plot.set(ylabel="Eigenvalue",xlabel="Vector")
sns.heatmap([[x**2 for x in component] for component in decomp.components_[:10]])
for component in decomp.components_[:3]:

    weights=sorted(zip([x**2 for x in component],df_num.columns.values),reverse=True)[:3]

    sns.barplot(data=pd.DataFrame(weights,columns=['Squared Weight','Feature Name']),y='Feature Name',x='Squared Weight')

    plt.show()
for num_vectors in range(1,len(decomp.singular_values_)):

    acc_ratio=sum(decomp.singular_values_[:num_vectors])/sum(decomp.singular_values_)

    if (sum(decomp.singular_values_[:num_vectors])/sum(decomp.singular_values_))>=0.80:

            break;

print(str(num_vectors)+" ("+str(round(acc_ratio*10000)/100)+"%)")
sq_eigenval=dict(zip(df_num.columns.values,[0 for i in range(len(df_num.columns.values))]))

for component in decomp.components_[:num_vectors]:

    weights=zip([x**2 for x in component],df_num.columns.values)

    for line in weights:

        sq_eigenval[line[1]]+=line[0] if not np.isnan(line[0]) else 0



#Normalization

sq_eigenval=pd.DataFrame(sq_eigenval.items(),columns=['Feature','Sq. Weight']).sort_values(by='Sq. Weight')

sq_eigenval['Norm. Sq. W.']=(sq_eigenval['Sq. Weight']-sq_eigenval['Sq. Weight'].min())/(sq_eigenval['Sq. Weight'].max()-sq_eigenval['Sq. Weight'].min())



plt.figure(figsize=(14,25))

sns.barplot(

            y='Feature',

            x='Norm. Sq. W.',

            data=sq_eigenval

)

plt.title('Normalized Sq. Weights of Features (SVD)')

plt.tight_layout()

plt.savefig('svd_importances.png')
df_num=df_train[[col for col in df_train.columns if not col in ['target','outliers']]].select_dtypes(include='number')

# df_num.fillna(0,inplace=True)

fillna_mode(df_num, df_num.columns)



df_svd_train=pd.DataFrame([x[:num_vectors]for x in decomp.transform(df_num)],columns=['svd_'+str(i) for i in range(num_vectors)])

#df_train_svd=df_train.join(df_svd,lsuffix='_caller', rsuffix='_other')

df_svd_train['target']=df_train['target']

df_svd_train['outliers']=df_train['outliers']



df_num=pd.DataFrame(data=df_test,columns=df_num.columns.values)

# df_num.fillna(0,inplace=True)

fillna_mode(df_num, df_num.columns)





df_svd_test=pd.DataFrame([x[:num_vectors]for x in decomp.transform(df_num)],columns=['svd_'+str(i) for i in range(num_vectors)])

#df_test_svd=df_test.join(df_svd,lsuffix='_caller', rsuffix='_other')

print("Datasets generated.")
sns.scatterplot(data=df_svd_train,x='svd_1',y='svd_0',hue='target')

plt.show()

sns.scatterplot(data=df_svd_train.query('svd_1>0'),x='svd_1',y='svd_0',hue='target')
target=df_svd_train['target']

del df_svd_train['target']
%%time

results=LGB_Train(df_svd_train,df_svd_test,target,5)

predictions_train_svd=results['predictions_train']

predictions_test_svd=results['predictions_test']
df_train['svd_prediction']=predictions_train_svd

df_test['svd_prediction']=predictions_test_svd



df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'outliers', 'target']]

target = df_train['target']

del df_train['target']
%%time

results=LGB_Train(df_train,df_test,target,31)

feature_importance_df=results['feature_importances']

predictions_brute=results['predictions_test']
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
temporal_features=[c for c in df_train.columns if (("week" in c) or ("day" in c) or ("date" in c) or ("year" in c))]

df_train_temporal = df_train[temporal_features]

df_test_temporal = df_test[temporal_features]

df_train_temporal['outliers']=df_train['outliers']

df_train_atemporal = df_train[[c for c in df_train.columns if c not in temporal_features]]

df_test_atemporal = df_test[[c for c in df_test.columns if c not in temporal_features]]

df_train_atemporal['outliers']=df_train['outliers']
%%time

results=LGB_Train(df_train_atemporal,df_test_atemporal,target,31)

predictions_test_atemporal=results['predictions_test']

predictions_train_atemporal=results['predictions_train']
%%time

results=LGB_Train(df_train_temporal,df_test_temporal,target,31)

predictions_test_temporal=results['predictions_test']

predictions_train_temporal=results['predictions_train']
df_train_impera = df_train[['svd_prediction','outliers']]

df_train_impera['temporal_prediction']=predictions_train_temporal

df_train_impera['atemporal_prediction']=predictions_train_atemporal

df_test_impera=df_test[['svd_prediction']]

df_test_impera['temporal_prediction']=predictions_test_temporal

df_test_impera['atemporal_prediction']=predictions_test_atemporal
%%time

results=LGB_Train(df_train_impera,df_test_impera,target,31)

feature_importance_df=results['feature_importances']

predictions_impera=results['predictions_test']
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(7,3))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds) for Divide-et-Impera method')

plt.tight_layout()

plt.savefig('lgbm_dei_importances.png')
df_train['temporal_prediction']=predictions_train_temporal

df_train['atemporal_prediction']=predictions_train_atemporal

df_test['temporal_prediction']=predictions_test_temporal

df_test['atemporal_prediction']=predictions_test_atemporal

df_train=df_train.join(df_svd_train[[c for c in df_svd_train.columns if "svd" in c]], lsuffix='_caller', rsuffix='_other')

df_test=df_test.join(df_svd_test[[c for c in df_svd_train.columns if "svd" in c]], lsuffix='_caller', rsuffix='_other')
%%time

results=LGB_Train(df_train,df_test,target,31)

feature_importance_df=results['feature_importances']

predictions_frankie=results['predictions_test']
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds) for Frankenstein\'s method')

plt.tight_layout()

plt.savefig('lgbm_frankie_importances.png')
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_frankie

sub_df.to_csv("submission_frankie.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_impera

sub_df.to_csv("submission_impera.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_brute

sub_df.to_csv("submission_brutus.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_test_atemporal

sub_df.to_csv("submission_atemporal.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_test_svd

sub_df.to_csv("submission_naive_svd.csv", index=False)