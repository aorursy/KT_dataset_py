# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 基本

import datetime as dt

import random

import os

from os import path

import gc

import sys

import math

import pandas as pd

from pandas import DataFrame, Series

import numpy as np

import scipy as sp

import scipy.sparse as sps



import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib import ticker

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns



# 検定

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import roc_auc_score, r2_score



###encoding

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import QuantileTransformer   # 数値列のRankGauss処理用



# モデリング

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor

import xgboost as xgb



#import datarobot as dr







# 表示量

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 100)

###定数宣言

STEP = '003'



### kaggle・・・kaggle設定で上書き

MODE='kaggle' #kaggle or local



#画像ファイルのディレクトリ

PATH_TO_TRAIN = 'data/train.csv'

PATH_TO_PRED = 'data/test.csv'

PATH_TO_SUBMIT= 'data/sample_submission.csv'

PATH_TO_MY_SUBMIT='my_submission_%s.csv' % (STEP)

PATH_TO_CITY= 'data/city_info.csv'

PATH_TO_STATION= 'data/station_info.csv'





###テストデータの割合

PER_TEST = 0.2



#kaggle環境用設定

if MODE == 'kaggle':    

    PATH_TO_TRAIN= '/kaggle/input/exam-for-students20200527/train.csv'

    PATH_TO_PRED= '/kaggle/input/exam-for-students20200527/test.csv'

    PATH_TO_SUBMIT= '/kaggle/input/exam-for-students20200527/sample_submission.csv'

    PATH_TO_CITY= '/kaggle/input/exam-for-students20200527/city_info.csv'

    PATH_TO_STATION= '/kaggle/input/exam-for-students20200527/station_info.csv'

    

    #PATH_TO_MY_SUBMIT= '/kaggle/input/exam-for-students20200527/' + PATH_TO_MY_SUBMIT



### Columns

TARGET_COL = 'TradePrice'



DATA_SIZE='full'

SEED_AVERAGING=True

SEEDS = [71, 48, 7]
def outlier(df,cols):

    for col in cols:

        df1 = df[col].copy()

        # 平均と標準偏差

        avg = np.mean(df1)

        std = np.std(df1)



        # 外れ値の基準点

        out_min = avg - (std * 2)

        out_max = avg + (std * 2)



        # 範囲から外れている値を除く

        df1[df1 < out_min] = None

        df1[df1 > out_max] = None



        df[col] = df1

    return df

#

# データ読み込み

#

DF_CACHE=None

DF_Y_CACHE=None

COLUMNS=None

def load_data(size='small', reload=False, parse_dates = []):

    

    df_train = None

    

    global DF_CACHE

    global DF_Y_CACHE

    if DF_CACHE is None or reload:

        print('load data....')

        if size == 'small':

            df_train = pd.read_csv(PATH_TO_TRAIN, index_col=0, parse_dates=parse_dates, skiprows=lambda x: x%95!=0)

            df_pred = pd.read_csv(PATH_TO_PRED, index_col=0, parse_dates= parse_dates,skiprows=lambda x: x%95!=0)

            #submission = pd.read_csv(PATH_TO_SUBMIT, index_col=0, skiprows=lambda x: x%95!=0)

        elif size == 'mid':

            df_train = pd.read_csv(PATH_TO_TRAIN, index_col=0, parse_dates=parse_dates, skiprows=lambda x: x%25!=0)

            df_pred = pd.read_csv(PATH_TO_PRED, index_col=0, parse_dates= parse_dates,skiprows=lambda x: x%25!=0)

            #submission = pd.read_csv(PATH_TO_SUBMIT, index_col=0, skiprows=lambda x: x%25!=0)

        elif size == 'full':

            df_train = pd.read_csv(PATH_TO_TRAIN, index_col=0, parse_dates=parse_dates,)

            df_pred = pd.read_csv(PATH_TO_PRED, index_col=0, parse_dates= parse_dates)

            #submission = pd.read_csv(PATH_TO_SUBMIT, index_col=0)



        #DF_CACHE['train'] = df_train

        #DF_CACHE['pred'] = df_pred

        #DF_CACHE['submit'] = submission 

        

        ###外れ値

        df_train = outlier(df_train,[TARGET_COL])

        

        ### train + pred

        df_train['is_train'] = True

        df_train['is_pred'] = False

        df_pred['is_pred'] = True

        df_pred['is_train'] = False

        

        df_y = df_train[TARGET_COL]

        df_train = df_train.drop(TARGET_COL,axis=1)

        

        DF_CACHE= pd.concat([df_train, df_pred], axis=0)

        DF_Y_CACHE = df_y

        

        global COLUMNS

        COLUMNS = DF_CACHE.columns

        

        

    return DF_CACHE,DF_Y_CACHE
#

# 数値系特徴量を網羅的に特徴量追加する

#

def all_numeric_cols_to_feture(df):

    # 全数値列に対して網羅的に特徴量生成

    num_cols_for_auto_create = df.select_dtypes(include='number').columns.values.tolist()



    print(num_cols_for_auto_create)



    for i in range(0, len(num_cols_for_auto_create)-1):

        for j in (i+1, len(num_cols_for_auto_create)-1):

            c1 = num_cols_for_auto_create[i]

            c2 = num_cols_for_auto_create[j]

            df[c1 + 'X' + c2] = df[c1] * df[c2]



            # c2に0が含まれてないなら割り算カラムも追加

            if (df[c2] == 0).sum() == 0:

                df[c1 + '/' + c2] = df[c1] / df[c2]



    # 無限大をNanに変換（後で欠損値として補完される）

    df=df.replace([np.inf, -np.inf], np.nan)

    

    return df
### 欠損値があるカラムのフラグカラムを追加

def appned_missing_flg(df):

    print( df.isnull().sum(axis=1))

    for col in COLUMNS:

        if df[col].isnull().any():

            df[col + '_isnan'] = df[col].isnull().replace({True : 1, False : 0})

            print('append. col='+col + '_isnan')



    return df

    
### StandardScaler



from sklearn.preprocessing import StandardScaler

def standard_scaler(df_arg,cols):

    df = df_arg.copy()

    scaler = StandardScaler()



    for col in cols:

        scaler.fit(df.iloc[:, df.columns == col])

        df[col] = scaler.transform(df.iloc[:, df.columns == col])

        print('standard_scaler:',col)

        

    return df
### RankGauss

from sklearn.preprocessing import QuantileTransformer

def rank_gauss(df_arg,cols):

    df = df_arg.copy()



    scaler = QuantileTransformer(n_quantiles=100, random_state=71, output_distribution='normal')



    for col in cols:

        

        # 学習データに基づいてRankGaussによる変換を定義

        scaler.fit(df.iloc[:, df.columns == col])

        # 変換後データの列を作成

        df[col] = scaler.transform(df.iloc[:, df.columns == col])

        print('rank_gauss:',col)



    return df
### LabelEncoding

def label_encoding(df_arg,cols):

    df = df_arg.copy()



    le = LabelEncoder()

    for col in cols:

        print(col)

        le.fit(df[col])

        df[col] = le.transform(df[col])

        print('label_encoding:',col)

        

    return df
### OneHotEncoding

def one_hot_encoding(df, cols):



    encoder = OneHotEncoder()

    for col in cols:

        df[col] = encoder.fit_transform(df[col])

        print('one_hot_encoding:',col)

    

    return df
def map_encoding(df_arg, cols, dictionary):

    df = df_arg.copy()

    for col in cols:

        d = dictionary.pop(0)

        df[col] = df[col].map(d)

        print('map_encoding:',col,d)



    return df
from sklearn.model_selection import KFold

def target_encoding(df_arg, cols):

    

    ###ターゲットと分割

    df = df_arg.copy()

    

    kf = KFold(n_splits=5, shuffle=True, random_state=71)

    

    for i, (tr_idx,va_idx) in enumerate(kf.split(df)):

        tr_x ,va_x = df_x.iloc[tr_idx].copy(), df_x.iloc[va_idx].copy()

        tr_y, va_y = df_y.iloc[tr_idx], df_y.iloc[va_idx]



        for col in cols:

            data_tmp = pd.DataFrame({c:tr_x[col], TARGET_COL: tr_y})

            target_mean = data_tmp.groupby(col)[TARGET_COL].mean()



            va_x.log[:,col] = va_x[col].map(target_mean)



            tmp = np.repeat(np.nan, tr_x.shape[0])

            kf2 = KFold(n_splits = 5, shuffle=True, random_stae=71)

            for idx_1, idx_2 in kf2.split(tr_x):

                target_mean = data_tmp.iloc[idx_1].groupby(col)[TARGET_COL].mean()

                tmp[idx_2] = tr_x[col].iloc[idx_2].map(target_mean)

            

            tr_x.loc[:,col] = tmp

            

    return df

            
###欠損値を埋める。median

def fillna_median(df_arg,cols):

    df = df_arg.copy()

    # 中央値で埋める

    for col in cols:

        median = df[col].median()

        df[col] = df[col].fillna(median, inplace=True)

        print('fillna_median:',col)

    return df 
### TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_max_features = 2000

def tfidf_vevtorizer(df_arg, cols):

    df = df_arg.copy()

    

    # テキストをTFIDF

    tdidf = TfidfVectorizer(min_df=3, norm='l2', ngram_range=(1, 3), stop_words='english', max_features=TFIDF_max_features)



    for col in cols:

        # 小文字化

        df[col] = df[col].str.lower()



        tdidf.fit(df[col] .fillna('#'))

        df[col]  = tdidf.transform(df[col] .fillna('#'))

        print('Vocabulary size: {}'.format(len(tdidf.vocabulary_)))

        print('tfidf_vevtorizer:',col)

        

    return df
def modeling(df_train_x, df_train_y, df_pred_x,params={}, n_splits=5, rounds=30, seed=None):

    

    ## cv score (for stacking)

    stacking_scores = pd.DataFrame({'score': np.zeros(df_train_x.shape[0])})

    scores = []

    predictions = []

    

    ### kfoldか層化抽出か

    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

    #skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    

    for i, (train_ix, valid_ix) in enumerate(kf.split(df_train_x, df_train_y)):

        ### foldに分割

        train_x, train_y = df_train_x.iloc[train_ix], df_train_y.iloc[train_ix]

        valid_x, valid_y = df_train_x.iloc[valid_ix], df_train_y.iloc[valid_ix]

        

        ###

        model = LGBMRegressor(n_estimators=9999, random_state=seed, **params)

        

        model.fit(train_x, train_y, early_stopping_rounds=rounds, eval_metric='rmse', eval_set=[(valid_x, valid_y )], verbose=100)



        ###検定

        y_valid  = model.predict(valid_x)

        

        # test_ixはインデックスではなく行番号のリストなのでilocでアクセス

        stacking_scores.iloc[valid_ix, stacking_scores.columns.get_loc('score')]= y_valid

        

        ### RMSLE 

        score = math.sqrt(sum((valid_y - y_valid)**2) / (len(y_valid)))

        scores.append(score)

        

        ### 予測

        y_pred = model.predict(df_pred_x)

        predictions.append(y_pred)

    

    ### スコアはFOLDの平均

    mean = sum(scores) / n_splits

    print(f' scores={scores}, mean={mean}')

    

    stacking_scores = stacking_scores['score'].values

    pred = sum(predictions) / n_splits

    

    return stacking_scores, scores, pred
def type_columns(coltypes):

    

    ret_cols=[]

    for coltype in coltypes:

        cols = df.select_dtypes(include=coltype).columns.values.tolist()



        if 'is_train' in cols:

            cols.remove('is_train')

        if 'is_pred' in cols:

            cols.remove('is_pred')

        ret_cols.extend(cols)



    return ret_cols
#

#　データ読み込み

#

df,df_y = load_data(size=DATA_SIZE,reload=True)



display(df.describe())

display(df.columns)

display(df_y)



### RMSLE用

df_y = np.log1p( df_y )

display(df_y)



print('all columns:',len(df.columns))
#

#　特徴量エンジニアリング

#

## StandardScalerで標準化

#df = standard_scaler(df, ['loan_amnt', 'installment', 'mths_since_last_record'])



### 欠損値処理

df = appned_missing_flg(df)



df = all_numeric_cols_to_feture(df)



## rank gauss

df = rank_gauss(df,type_columns(['number']))

###　カテゴリ



#df = one_hot_encoding(df,['initial_list_status'])



#dictionaly = {np.nan:0, '< 1 year':0.5, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}

#df = map_encoding(df,['emp_length'],[dictionaly])



#df = tfidf_vevtorizer(df,['Remarks'])



### 欠損値

df = fillna_median(df,type_columns(['number']))



#df = df.drop(["Remarks"],axis=1)



for t in type_columns(['object']):

    df[t] = df[t].fillna('NaN')

    

### カテゴリ

df = label_encoding(df, type_columns(['object']))

df.columns

#

#　モデリング

#





### trainとpredの分離

df_train_x = df[df['is_train']==1].drop(columns=['is_train','is_pred'])

#df_train_y = df[df['is_train']==1][TARGET_COL]

df_train_y = df_y.copy()

df_pred_x = df[df['is_pred']==1].drop(columns=['is_train','is_pred'])





#df = target_encoding(df_train_x, df_train_y, BASE_COLUMNS['object'])

#df = target_encoding(df, BASE_COLUMNS['object'])

display(df_train_x)

display(df_train_y)

display(df_pred_x)


if SEED_AVERAGING:

    print('performing seed averaging')

    preds = []

    for seed in SEEDS:

        stacking_scores, scores, pred = modeling(df_train_x, df_train_y, rounds=100, df_pred_x=df_pred_x, seed=seed)

        preds.append(pred)

    pred = sum(preds) / len(SEEDS)

else:

    stacking_scores, scores, pred = modeling(df_train_x, df_train_y, rounds=100, df_pred_x=df_pred_x, seed=SEEDS[0])



### 学習データに範囲があれば区切る

#df_train_x = df_train_x[df_train_x['issue_d'] >= dt.datetime(2015,1,1)]

#

#　後処理

#

display(stacking_scores)

display(scores)

display(pred)
#

#　Submit

#



submit = pd.read_csv(PATH_TO_SUBMIT)

submit.TradePrice = pred



submit.TradePrice = np.expm1(submit.TradePrice )

display(submit.head())

submit.to_csv(PATH_TO_MY_SUBMIT,index=False)