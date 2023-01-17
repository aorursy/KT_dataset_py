import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from collections import defaultdict, Counter

from tqdm import tqdm

from catboost import CatBoostClassifier

from matplotlib import pyplot as plt

import time

import gc

from scipy.stats import entropy

from gensim.models import Word2Vec

from sklearn.metrics import *
test_path = "../input/2020-digix-advertisement-ctr-prediction/test_data_A.csv"

train_path = "../input/2020-digix-advertisement-ctr-prediction/train_data/train_data.csv"
# train dataset庞大（4千万），故以100万份为一批读取，并从每批中选取5%作为新的train dataset（n代表取100万中的第n份5%）

def load_data(train_data_path, test_data_path, n):

    chunkSize = 10 ** 6

    num_of_chunk = 0

    train = pd.DataFrame()



    for chunk in tqdm(pd.read_csv(train_data_path, iterator=True, sep='|', chunksize=chunkSize)):

        num_of_chunk += 1

        train = pd.concat([train, chunk.iloc[n*50000:(n+1)*50000,:]], axis=0)

        #print('Processing Chunk No. ' + str(num_of_chunk))  

    train.reset_index(drop=True, inplace=True)

    loop = True

    chunks = []

    test = pd.read_csv(test_data_path, iterator=True, sep='|')

    while loop:

        try:

            #print(index)

            chunk = test.get_chunk(chunkSize)

            chunks.append(chunk)

            #index += 1

        except StopIteration:

            loop = False

            print("testing data iteration stopped.")

    for i in tqdm(chunks):

        test = pd.concat(chunks, ignore_index=True)

    test['label'] = np.nan

    data = pd.concat([train, test], axis=0)

    return data
# 减小数据容量

def reduce_mem(df):

    starttime = time.time()

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if pd.isnull(c_min) or pd.isnull(c_max):

                continue

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

    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,

                                                                                                           100*(start_mem-end_mem)/start_mem,

                                                                                                           (time.time()-starttime)/60))

    return df
df = load_data(train_path, test_path, 0)
df.head()
age = Counter(age for age in df['age'])

items = sorted(age.items())

x_age_count = [i[:][0] for i in items]

y_age_count = [i[:][1] for i in items]

plt.figure(figsize = (5, 5))

sns.barplot(y = y_age_count, x = x_age_count)
city = Counter(city for city in df['city'])

city_items = city.most_common(20)

x_city_count = [i[:][0] for i in city_items]

y_city_count = [i[:][1] for i in city_items]

plt.figure(figsize = (10, 10))

sns.barplot(y = y_city_count, x = x_city_count)

plt.xticks(rotation=90)
city_rank = Counter(city_rank for city_rank in df['city_rank'])

city_rank_items = sorted(city_rank.items())

x_city_rank_count = [i[:][0] for i in city_rank_items]

y_city_rank_count = [i[:][1] for i in city_rank_items]

plt.figure(figsize = (5, 5))

sns.barplot(y = y_city_rank_count, x = x_city_rank_count)

#plt.xticks(rotation=90)
##########################cate feature#######################

cate_cols = ['slot_id','net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','dev_id','tags']

for f in tqdm(cate_cols):

    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))

    df[f + '_count'] = df[f].map(df[f].value_counts())

df = reduce_mem(df)
df.head()
##########################groupby feature#######################

def group_fea(df,key,target):

    tmp = df.groupby(key, as_index=False)[target].agg({

        key+target + '_nunique': 'nunique',

    }).reset_index()

    del tmp['index']

    print("**************************{}**************************".format(target))

    return tmp



feature_key = ['uid','age','career','net_type']

feature_target = ['task_id','adv_id','dev_id','slot_id','spread_app_id','indu_name']



for key in tqdm(feature_key):

    for target in feature_target:

        tmp = group_fea(df,key,target)

        df = df.merge(tmp,on=key,how='left')
df.head()
test_df = df[df["pt_d"]==8].copy().reset_index()

train_df = df[df["pt_d"]<8].reset_index()

del df

gc.collect()
#统计做了groupby特征的特征

group_list = []

for s in train_df.columns:

    if '_nunique' in s:

        group_list.append(s)

print(group_list)
##########################target_enc feature#######################

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

enc_list = group_list + ['net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','uid_count','dev_id','tags','slot_id']

for f in tqdm(enc_list):

    train_df[f + '_target_enc'] = 0

    test_df[f + '_target_enc'] = 0

    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):

        trn_x = train_df[[f, 'label']].iloc[trn_idx].reset_index(drop=True)

        val_x = train_df[[f]].iloc[val_idx].reset_index(drop=True)

        enc_df = trn_x.groupby(f, as_index=False)['label'].agg({f + '_target_enc': 'mean'})

        val_x = val_x.merge(enc_df, on=f, how='left')

        test_x = test_df[[f]].merge(enc_df, on=f, how='left')

        val_x[f + '_target_enc'] = val_x[f + '_target_enc'].fillna(train_df['label'].mean())

        test_x[f + '_target_enc'] = test_x[f + '_target_enc'].fillna(train_df['label'].mean())

        train_df.loc[val_idx, f + '_target_enc'] = val_x[f + '_target_enc'].values

        test_df[f + '_target_enc'] += test_x[f + '_target_enc'].values / skf.n_splits
#线下数据集的切分

X_train = train_df[train_df["pt_d"]<=6].copy()

y_train = X_train["label"].astype('int32')

X_valid = train_df[train_df["pt_d"]>6]

y_valid = X_valid["label"].astype('int32')
#筛选特征

drop_fea = ['pt_d','label','communication_onlinerate','index']

feature= [x for x in X_train.columns if x not in drop_fea]

print(len(feature))

print(feature)