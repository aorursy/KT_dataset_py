IS_KAGGLE_KERNEL=True
import gc
import time
import datetime
from collections import Counter
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import featuretools
from multiprocessing import Pool
import multiprocessing as multi
from functools import partial
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
if IS_KAGGLE_KERNEL:
    from kaggle.competitions import twosigmanews
pd.set_option('max_columns', 10)
pd.set_option('max_rows', 10)

def make_window_ranges(windows):
    window_ranges=[]
    for left, right in windows:
        window_ranges.append(range(left, right+1))
    return window_ranges
START_DT=datetime.datetime(2010, 1, 1, 0, 0, 0).date() #
TEST_RATE=0.2
OUTLIER_THRES=0.8 #１日のうちにOUTLIER_THRES以上変化するのは修正する
INDIV_THRES=0.8 #個別銘柄として学習するか、全体でマージした学習器にするか？
RETURN_THRES=50 #前後１日で２５％以上変化するのは学習しなくてよい。株式が分割された場合もあるが…
WINDOWS=[(1,5),(6,10),(11,20),(21,40),(41,60)]; #max,min,ave,stdを出すwindowサイズ
WINDOW_RANGES= make_window_ranges(WINDOWS)

TARGET='returnsOpenNextMktres10' #目的変数
if IS_KAGGLE_KERNEL:
    env = twosigmanews.make_env()
    market_df, news_train = env.get_training_data()
else:
    market_df=pd.read_pickle('market_train.zip')
#     news_train_df=pd.read_pickle('news_train.zip')
# market_df=market_df.dropna(subset=['returnsOpenNextMktres10'])
# market_test_df=market_df[market_df['returnsOpenNextMktres10'].isnull()]
# display(len(market_df))
# display(len(market_test_df))    
market_df=market_df.astype({'volume':'int', 'universe':'bool'})
market_df = market_df.loc[market_df['time'].dt.date >= START_DT]
market_df['dif_close_open']=np.abs((market_df['close']-market_df['open'])/market_df['open'])
market_df[market_df['dif_close_open']>OUTLIER_THRES]
#2016-07-06に多いことが確認できるが、その他の銘柄ではおかしくなかったため特定銘柄についてのバグだった模様
# おかしい値の前後日確認用コード
# market_df[(market_df['assetCode']=='ZNGA.O')&
#                 (market_df['time'].dt.date>=datetime.datetime(2016,7,5).date())&
#                 (market_df['time'].dt.date<=datetime.datetime(2016,7,7).date())]
market_df.ix[1127598, 'open']=market_df.ix[1127598, 'close'] # 9998.89→50
market_df.ix[1862465, 'open']=market_df.ix[1862465, 'close'] #0.09→0.736
market_df.ix[3845309, 'close']=(market_df.ix[3843511, 'close']+market_df.ix[3847108, 'close'])/2 #DISH.O
market_df.ix[3845015, 'close']=(market_df.ix[3843216, 'close']+market_df.ix[3846813, 'close'])/2 #BBBY.O
market_df.ix[3845467, 'close']=(market_df.ix[3843668, 'close']+market_df.ix[3847265, 'close'])/2 #FLEX.O
market_df.ix[3845835, 'close']=(market_df.ix[3844037, 'close']+market_df.ix[3847633, 'close'])/2 #MAT.O
market_df.ix[3846067, 'close']=(market_df.ix[3844269, 'close']+market_df.ix[3847865, 'close'])/2 #PCAR.O
market_df.ix[3846276, 'close']=(market_df.ix[3844479, 'close']+market_df.ix[3848074, 'close'])/2 #SHLD.O
market_df.ix[3846636, 'close']=(market_df.ix[3844838, 'close']+market_df.ix[3848433, 'close'])/2 #ZNGA.O
market_df.ix[3845309, 'open']=(market_df.ix[3843511, 'open']+market_df.ix[3847108, 'open'])/2 #DISH.O
market_df.ix[3845015, 'open']=(market_df.ix[3843216, 'open']+market_df.ix[3846813, 'open'])/2 #BBBY.O
market_df.ix[3845467, 'open']=(market_df.ix[3843668, 'open']+market_df.ix[3847265, 'open'])/2 #FLEX.O
market_df.ix[3845835, 'open']=(market_df.ix[3844037, 'open']+market_df.ix[3847633, 'open'])/2 #MAT.O
market_df.ix[3846067, 'open']=(market_df.ix[3844269, 'open']+market_df.ix[3847865, 'open'])/2 #PCAR.O
market_df.ix[3846276, 'open']=(market_df.ix[3844479, 'open']+market_df.ix[3848074, 'open'])/2 #SHLD.O
market_df.ix[3846636, 'open']=(market_df.ix[3844838, 'open']+market_df.ix[3848433, 'open'])/2 #ZNGA.O
def calc_features(df):
    start = time.time()
    g_df=df.groupby('assetCode')['close']
    # N日前から現在への変動率（dif{i})の計算
    for i in range(len(WINDOW_RANGES)):
        window_range=WINDOW_RANGES[i]
        columns=[f'dif{i}' for i in window_range]
        for j in window_range:
            df[f'dif{j}']=g_df.diff(periods=j)*100.0/g_df.shift(j)
        range_str=f'{window_range[0]}_{window_range[-1]}'
        dif_df=df.loc[:,columns]
        df[f'dif{range_str}_max']=dif_df.max(axis='columns')
        df[f'dif{range_str}_min']=dif_df.min(axis='columns')
        df[f'dif{range_str}_ave']=dif_df.mean(axis='columns')
        df[f'dif{range_str}_std']=dif_df.std(axis='columns')
        df[f'dif{range_str}_nan_count']=dif_df.isna().sum(axis='columns')
        if i!=0:
            # 直近（range(0,5)）の値動き重視する。それ以外のrangeを削除する。
            df=df.drop(columns, axis='columns')
    display(f'time calc_feature :{time.time() - start}')
    return df
market_df=calc_features(market_df)
def delete_overshoot(df):
    # 異常値（前後５日で５０％以上上昇 or ５０％以上下落）を含むレコードを抜く
    dif_df=df.loc[:,['dif1','dif2','dif3','dif4','dif5']].fillna(0)
    df=df[(dif_df.loc[:,'dif1']<RETURN_THRES)&(dif_df.loc[:,'dif1']>-RETURN_THRES)&
          (dif_df.loc[:,'dif2']<RETURN_THRES)&(dif_df.loc[:,'dif2']>-RETURN_THRES)&
          (dif_df.loc[:,'dif3']<RETURN_THRES)&(dif_df.loc[:,'dif3']>-RETURN_THRES)&
          (dif_df.loc[:,'dif4']<RETURN_THRES)&(dif_df.loc[:,'dif4']>-RETURN_THRES)&
          (dif_df.loc[:,'dif5']<RETURN_THRES)&(dif_df.loc[:,'dif5']>-RETURN_THRES)]
    return df
market_df=delete_overshoot(market_df)
drop_columns=['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
              'returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevMktres1', 'returnsOpenPrevMktres1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
              'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'universe', 'dif_close_open']

market_df=market_df.drop(drop_columns, axis='columns')
best_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 63,
#     'num_boost_round':1000,
    'n_estimators': 1000,
    'min_child_samples': 20,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.8,
}

market_train_df, market_valid_df = train_test_split(market_df, test_size=0.2, shuffle=False)
clfs={}
_1_5_feats=['dif1','dif2','dif3','dif4','dif5']+[column for column in market_df.columns if column.startswith('dif1_5_')]
_6_10_feats=_1_5_feats+[column for column in market_df.columns if column.startswith('dif6_10_')]
_11_20_feats=_6_10_feats+[column for column in market_df.columns if column.startswith('dif11_20_')]
_21_40_feats=_11_20_feats+[column for column in market_df.columns if column.startswith('dif21_40_')]
_41_60_feats=_21_40_feats+[column for column in market_df.columns if column.startswith('dif41_60_')]
for max_nan, period, columns in [(2,'1_5', _1_5_feats),
                                 (2,'6_10', _6_10_feats), 
                                 (4,'11_20', _11_20_feats),
                                 (8,'21_40', _21_40_feats),
                                 (8,'41_60', _41_60_feats)]:
    market_train_df=market_train_df[market_train_df[f'dif{period}_nan_count']<=max_nan] #forを回す中で、どんどん制限をかしていく
    lgb_train = lgb.Dataset(market_train_df.loc[:,columns], market_train_df[TARGET])
    lgb_eval = lgb.Dataset(market_valid_df.loc[:,columns], market_valid_df[TARGET], reference=lgb_train)
    clf = lgb.train(
        best_params,
        lgb_train,
        valid_sets=lgb_eval,
        early_stopping_rounds=5,
        verbose_eval=False)
    clfs[period]=clf
#     display(Series(clf.predict(market_train_df.loc[:,columns])).sort_values(ascending=False))
display(clfs)
def get_prediction_days():
    market_test_df=pd.read_pickle('market_test.zip')
    news_test_df=pd.read_pickle('news_test.zip')
    #timeをindexにしたほうがきれいに書ける
    market_test_df=market_test_df.set_index('time')
    news_test_df=news_test_df.set_index('time')
    test_start=datetime.datetime(2017,1,3)
    test_end=datetime.datetime(2018,12,30)
    day_range=range((test_end-test_start).days)
    days=[]
    for day_num in day_range:
        market_obs_df=market_test_df.loc[test_start+datetime.timedelta(days=day_num):test_start+datetime.timedelta(days=day_num+1)]
        news_obs_df=market_test_df.loc[test_start+datetime.timedelta(days=day_num):test_start+datetime.timedelta(days=day_num+1)]
        predictions_template_df=DataFrame(columns=['assetCode','confidenceValue'])
        predictions_template_df['assetCode']=market_obs_df['assetCode'].values
        predictions_template_df['confidenceValue']=0*len(predictions_template_df)
        days.append((market_obs_df,news_obs_df,predictions_template_df))
    #timeのindexをもとに戻す
    market_test_df=market_test_df.reset_index()
    news_test_df=news_test_df.reset_index()
    return days
# def write_submission(model, env):
if IS_KAGGLE_KERNEL:
    days = env.get_prediction_days()
else:
    days=get_prediction_days()
# def __clf_predict(_, **kwargs):
#     max_nan, period, columns, confidence=_
#     market_day_df=kwargs['market_day_df']; clfs=kwargs['clfs']
#     display(kwargs)
#     market_day_df=market_day_df[market_day_df[f'dif{period}_nan_count']<=max_nan] #forを回す中で、どんどん制限をかしていく
#     if len(market_day_df)>0:
#         prd=(clfs[period].predict(market_day_df.loc[:,columns])*10).clip(-1.0,1.0)*confidence
#         return Series(prd, index=market_day_df['assetCode'])
#     else:
#         return 0*len(prd_df.index)
    
# def clf_predict(max_workers, iter_list, **kwargs):    
#     p = Pool(max_workers)
# #     if seeds is None:
# #     seeds=range(len(iter_list))
#     result=p.map(partial(__clf_predict, **kwargs), iter_list)
#     p.close()
#     return result
# pd.set_option('max_columns',None)
# pd.set_option('max_rows',5)
# market_test_df=DataFrame()
# day_id=0
# for (market_day_df, news_df, prd_tmp_df) in days:
#     prd_tmp_df=prd_tmp_df.set_index('assetCode')
#     market_day_df['id'] = day_id
#     market_test_df = pd.concat([market_test_df,market_day_df], ignore_index=True, sort=False)
#     market_test_df = calc_features(market_test_df)
#     market_day_df=market_test_df[market_test_df['id']==day_id]
#     prd_df=pd.DataFrame(index=prd_tmp_df.index)
#     iter_list=[(2,'1_5', _1_5_feats, 0.1),
#      (2,'6_10', _6_10_feats, 0.3), 
#      (4,'11_20', _11_20_feats, 0.5),
#      (8,'21_40', _21_40_feats, 0.7),
#      (8,'41_60', _41_60_feats, 1.0)]
#     result=clf_predict(max_workers=5, iter_list=iter_list, market_day_df=market_day_df, clfs=clfs)
#     display(result)
#     break
# #     for max_nan, period, columns, confidence in :
# #         market_day_df=market_day_df[market_day_df[f'dif{period}_nan_count']<=max_nan] #forを回す中で、どんどん制限をかしていく
# #         if len(market_day_df)>0:
# #             prd=(clfs[period].predict(market_day_df.loc[:,columns])*10).clip(-1.0,1.0)*confidence
# #             prd_df[period]=Series(prd, index=market_day_df['assetCode'])
# #         else:
# #             prd_df[period]=0*len(prd_df.index)
#     prd_tmp_df['confidenceValue']=prd_df.mean(axis='columns')
#     display(prd_tmp_df)
#     if IS_KAGGLE_KERNEL:
#         env.predict(prd_tmp_df.reset_index())
#     gc.collect()
#     day_id+=1
# if IS_KAGGLE_KERNEL:
#     env.write_submission_file()
pd.set_option('max_columns',None)
pd.set_option('max_rows',5)
market_test_df=DataFrame()
day_id=0
for (market_day_df, news_df, prd_tmp_df) in days:
    if len(market_day_df.index)==0:        
        env.predict(prd_tmp_df)
        continue
    prd_tmp_df=prd_tmp_df.set_index('assetCode')
    market_day_df['id'] = day_id
    oldest_day=market_day_df.ix[0,'time']-datetime.timedelta(days=100)
    #特徴量算出に必要な期間のみをmarket_test_dfに積み重ねる。（oldest_day以前のデータはいらない）
    market_test_df = pd.concat([market_test_df,market_day_df], ignore_index=True, sort=False)
    market_test_df=market_test_df[market_test_df['id']>day_id-65]
    market_test_df = calc_features(market_test_df)
    market_day_df=market_test_df[market_test_df['id']==day_id]
    prd_df=pd.DataFrame(index=prd_tmp_df.index)
    for max_nan, period, columns, confidence in [(2,'1_5', _1_5_feats, 0.1),
                                                 (2,'6_10', _6_10_feats, 0.3), 
                                                 (4,'11_20', _11_20_feats, 0.5),
                                                 (8,'21_40', _21_40_feats, 0.7),
                                                 (8,'41_60', _41_60_feats, 1.0)]:
        market_day_df=market_day_df[market_day_df[f'dif{period}_nan_count']<=max_nan] #forを回す中で、どんどん制限をかしていく
        if len(market_day_df)>0:
            prd=(clfs[period].predict(market_day_df.loc[:,columns])*10).clip(-1.0,1.0)*confidence
            prd_df[period]=Series(prd, index=market_day_df['assetCode'])
        else:
            prd_df[period]=0*len(prd_df.index)
    prd_tmp_df['confidenceValue']=prd_df.fillna(0).mean(axis='columns')
    if IS_KAGGLE_KERNEL:
        env.predict(prd_tmp_df.reset_index())
    gc.collect()
    day_id+=1
if IS_KAGGLE_KERNEL:
    env.write_submission_file()