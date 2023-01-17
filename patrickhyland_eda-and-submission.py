# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
env=twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
market_train_df=market_train_df.sort_values('time')
market_train_df['time']=market_train_df['time'].dt.date
market_train_df['time']=pd.to_datetime(market_train_df['time'])
#keep just for experimenting
market_train_df=market_train_df.tail(1000000)

# news_train_df=news_train_df.sort_values('time')
# news_train_df['time']=news_train_df['time'].dt.date
# news_train_df=news_train_df.tail(2600000)
del news_train_df
# news_train_df['sent_score']=news_train_df[['sentimentPositive','sentimentNegative','relevance','marketCommentary']]\
# .apply(lambda x: x.relevance*(x.sentimentPositive-x.sentimentNegative) if x.marketCommentary == False else 0,axis=1)
# news_train_df=news_train_df[['time','urgency','assetCodes','firstMentionSentence','noveltyCount7D','volumeCounts7D','sent_score']]
# import re
# news_train_df['assetCodes']=news_train_df[['assetCodes']].apply(lambda x: re.sub('[{\\\'}]','',x.assetCodes).split(','),axis=1)
# news_train_df.set_index('time')\
#                                .apply(lambda x: x.apply(pd.Series).stack())\
#                                .reset_index(level=1, drop=True)\
#                                .reset_index()
# columns = list(df.columns.difference(['athlete_id']))
# print(columns)
# result = [df]
# #print(result)
# for window in range(2, 4):
#     rolled = df.groupby('athlete_id', group_keys=False).rolling(
#         center=False, window=window, min_periods=1)

#     new_df = rolled.sum().drop('athlete_id', axis=1)
#     new_df.columns = ['sum_col{}_winsize{}'.format(col, window) for col in columns]
#     result.append(new_df)

#     new_df = rolled.min().drop('athlete_id', axis=1)
#     new_df.columns = ['min_col{}_winsize{}'.format(col, window) for col in columns]
#     result.append(new_df)

#     new_df = rolled.max().drop('athlete_id', axis=1)
#     new_df.columns = ['max_col{}_winsize{}'.format(col, window) for col in columns]
#     result.append(new_df)

# df = pd.concat(result, axis=1)
# print(new_df)
# type(result)
#backup
def getFeatures(df,pred=False):
    df.reset_index(drop=True,inplace=True)
    if pred:
        df_new=df[['time','assetCode','volume','returnsOpenPrevMktres1']].groupby('assetCode').apply(lambda x: x.assign(avg_10d=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).mean(),
                                                                        avg_3d=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).mean(),
                                                                       avg_30d=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).mean(),
                                                                       q25_10=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).quantile(.25),
                                                                       q25_3=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).quantile(.25),
                                                                       q25_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).quantile(.25),
                                                                       q75_10=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).quantile(.75),
                                                                       q75_3=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).quantile(.75),
                                                                       q75_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).quantile(.75),
                                                                       vol_10=x.volume.rolling(window=10,min_periods=9).mean(),
                                                                       vol_30=x.volume.rolling(window=30,min_periods=29).mean(),
                                                                       vol_std=x.volume.rolling(window=30,min_periods=29).std(),
                                                                       std_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).std()))

# get max date for each asset (prediction date needed), concat new features, join to get same day data
        df_new = df_new.groupby(['assetCode'])[['time']].max().reset_index(drop=False)\
        .merge(df_new,'left',on=['assetCode','time'])\
        .drop(['returnsOpenPrevMktres1','volume'],axis=1)\
        .merge(df,'left',on=['time','assetCode'])
                  
    else:
        df_new=df.groupby('assetCode').apply(lambda x: x.assign(avg_10d=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).mean(),
                                                                        avg_3d=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).mean(),
                                                                       avg_30d=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).mean(),
                                                                       q25_10=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).quantile(.25),
                                                                       q25_3=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).quantile(.25),
                                                                       q25_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).quantile(.25),
                                                                       q75_10=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).quantile(.75),
                                                                       q75_3=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).quantile(.75),
                                                                       q75_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).quantile(.75),
                                                                       vol_10=x.volume.rolling(window=10,min_periods=9).mean(),
                                                                       vol_30=x.volume.rolling(window=30,min_periods=29).mean(),
                                                                       vol_std=x.volume.rolling(window=30,min_periods=29).std(),
                                                                       std_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).std()))
    return df_new
# def getFeatures(df,pred=False):
#     df.reset_index(drop=True,inplace=True)
#     df=df.groupby('assetCode').apply(lambda x: x.assign(avg_10d=pd.rolling_mean(x.returnsOpenPrevMktres1,window=10,min_periods=9),
#                                                                         avg_3d=rolling_mean(x.returnsOpenPrevMktres1,window=3,min_periods=2),
#                                                                        avg_30d=rolling_mean(x.returnsOpenPrevMktres1,window=30,min_periods=29),
#                                                                        q25_10=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).quantile(.25),
#                                                                        q25_3=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).quantile(.25),
#                                                                        q25_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).quantile(.25),
#                                                                        q75_10=x.returnsOpenPrevMktres1.rolling(window=10,min_periods=9).quantile(.75),
#                                                                        q75_3=x.returnsOpenPrevMktres1.rolling(window=3,min_periods=2).quantile(.75),
#                                                                        q75_30=x.returnsOpenPrevMktres1.rolling(window=30,min_periods=29).quantile(.75),
#                                                                        vol_10=rolling_mean(x.volume,window=10,min_periods=9),
#                                                                        vol_30=rolling_mean(x.volume,window=30,min_periods=29),
#                                                                        vol_std=rolling_std(window=30,min_periods=29).std(),
#                                                                        std_30=rolling_std(x.returnsOpenPrevMktres1,window=30,min_periods=29)))
#     return df
%%time
fe_market=getFeatures(market_train_df)

# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
fe_market['returnsOpenNextMktres10']=fe_market['returnsOpenNextMktres10'].apply(lambda x: 1 if x >0 else 0)
fe_market_new=fe_market.drop(['assetCode','assetName','open','close','universe'],axis=1)
import lightgbm
lgbm=lightgbm.LGBMClassifier(n_estimators=300,silent=9,reg_lambda=.5,learning_rate=0.05)
#train_test_split(X, y,test_size=.3,random_state=122,stratify=y)
# import matplotlib.pyplot as plt
# from sklearn import metrics
# def plot_roc(y_true,y_score):
#     print(metrics.roc_auc_score(y_true,y_score))
#     a_curve=metrics.roc_curve(y_score=y_score,y_true=y_true)
#     plt.clf()
#     plt.figure(figsize=(10,10))
#     plt.plot(a_curve[0], a_curve[1], label='ROC curve')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.style.use('ggplot')

#     plt.legend(loc="lower left")
#     plt.show()
# def plot_pr (y_true,y_score,thresh_prec=.99):
#     print(metrics.average_precision_score(y_true,y_score))
#     lines=metrics.precision_recall_curve(probas_pred=y_score,y_true=y_true)
#     #rec=max(lines[1][lines[0]>thresh_prec])
#     plt.clf()
#     plt.figure(figsize=(10,10))
#     plt.plot(lines[1], lines[0], label='Precision-Recall curve')
#     #plt.axvline(x=rec,label='recall={:.3f}'.format(rec),color='blue')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.style.use('ggplot')
#     plt.legend(loc="lower left")
#     #plt.scatter(rec,thresh_prec,color='blue')
#     #plt.text(rec+.03,thresh_prec+.03,'Recall={:.3f}'.format(rec),color='blue',fontsize=12)
#     plt.show()
%%time
lgbm.fit(fe_market_new.drop(['time','returnsOpenNextMktres10'],axis=1),fe_market['returnsOpenNextMktres10'])
must_match_cols=list(fe_market_new.drop(['time','returnsOpenNextMktres10'],axis=1).columns)
# clf1=XGBClassifier(n_jobs=4,n_estimators=600,max_depth=5,eta=0.1,tree_method='hist')
# %%time
# clf1.fit(fe_market_new.drop('returnsOpenNextMktres10',axis=1),fe_market['returnsOpenNextMktres10'])
# preds=clf1.predict(fe_market_new.drop('returnsOpenNextMktres10',axis=1))
# preds=lgbm.predict_proba(fe_market_new.drop(['time','returnsOpenNextMktres10'],axis=1))[:,1]
# sorted(zip(fe_market_new.drop(['time','returnsOpenNextMktres10'],axis=1).columns,lgbm.feature_importances_), key=lambda tup: tup[1], reverse=True)
# plot_pr(np.array(fe_market['returnsOpenNextMktres10']),preds,.9)
# def summary(classifier, X, y):
#     print("Summary:\n%s\n" % (
#         metrics.classification_report(
#             y,
#             classifier.predict(X))))
# summary(lgbm,fe_market_new.drop('returnsOpenNextMktres10',axis=1),np.array(fe_market['returnsOpenNextMktres10']))
# from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score
# print(average_precision_score(fe_market['returnsOpenNextMktres10'], preds))
# print(roc_auc_score(fe_market['returnsOpenNextMktres10'], preds))
# print(accuracy_score(fe_market['returnsOpenNextMktres10'], preds))
from datetime import timedelta
days = env.get_prediction_days()
#(market_obs_df, news_obs_df, predictions_template_df) = next(days)
# old_assets=fe_market['assetCode'][fe_market['time']==test_date].unique()
# new_assets=predictions_template_df['assetCode'].unique()
# len([i for i in new_assets if i not in old_assets])
def make_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * lgbm.predict_proba(predictions_df)[:,1] - 1.0
# hist_date=pd.to_datetime(market_obs_df['time'].dt.date - timedelta(days=35))[0]
# print(hist_date)
# fe_market['time']=pd.to_datetime(fe_market['time'])
# test_date= pd.to_datetime(market_obs_df['time'].dt.date - timedelta(days=35))[0]
# print(pd.to_datetime(market_obs_df['time'].dt.date)[0])
# print(test_date)
# fe_market.loc[fe_market['time']>=test_date]
# test_df = fe_market[fe_market['time']==test_date].merge(predictions_template_df.drop('confidenceValue',axis=1),'inner',on='assetCode')
# drop_cols=['assetCode','assetName','open','close','time','returnsOpenNextMktres10']
# market_obs_df['time']=market_obs_df['time'].dt.date
# market_obs_df['time']=pd.to_datetime(market_obs_df['time'])
# hist_date=pd.to_datetime(market_obs_df['time'].dt.date - timedelta(days=35))[0]
# pred_date=pd.to_datetime(market_obs_df['time'].dt.date)[0]
# market_train_df=market_train_df[market_train_df['time']>=hist_date]
#     ## fe_market is now the most recent 35 days of stock data
    
#     #append new data with old
# market_train_df=market_train_df.append(market_obs_df)
# fe_market=getFeatures(market_train_df)
#     ## make sure we only predict on latest date and relevant asset
# fe_market=fe_market[fe_market['time']==pred_date].merge(predictions_template_df.drop('confidenceValue',axis=1),'inner',on='assetCode')
# print(list(fe_market.columns))
# make_predictions(fe_market.drop(drop_cols,axis=1))
# list(fe_market.drop(drop_cols,axis=1).columns)
# make_predictions(fe_market[must_match_cols])
#env.predict(predictions_template_df)
# [i for i in market_obs_df.columns if i not in market_train_df.columns]
# import numpy as np
# def make_random_predictions(predictions_df):
#     predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
# for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
#     make_random_predictions(predictions_template_df)
#     env.predict(predictions_template_df)
# print('Done!')

# env.write_submission_file()
market_train_df=market_train_df.drop(['universe','assetName','open','close','returnsOpenNextMktres10'],axis=1)

# #iteration takes 27 seconds with this method
# import numpy as np
# import time

# drop_cols=['assetCode','assetName','open','close','time','returnsOpenNextMktres10']

# #
# for i,(market_obs_df, news_obs_df, predictions_template_df) in enumerate(days):

#     start=time.time()
#     print("\n round",i,"\n\n")
#     #print("dtypes \n")
#     #print(market_obs_df.dtypes,market_train_df.dtypes)
#     # convert time to correct format
#     market_obs_df['time']=market_obs_df['time'].dt.date
#     market_obs_df['time']=pd.to_datetime(market_obs_df['time'])
#     hist_date=pd.to_datetime(market_obs_df['time'].dt.date - timedelta(days=34))[0]
#     pred_date=pd.to_datetime(market_obs_df['time'].dt.date)[0]
#     print(pred_date)
#     market_train_df=market_train_df[market_train_df['time']>=hist_date]
#     ## fe_market is now the most recent 35 days of stock data
    
#     #append new data with old
#     market_train_df=market_train_df.append(market_obs_df)
#     #print("market_train_df cols part 2: ",list(market_train_df.columns))
#     print(market_train_df.head())
#     fe_market=getFeatures(market_train_df)
    
#     print ("got features")
#     ## make sure we only predict on latest date and relevant asset
#     fe_market=fe_market[fe_market['time']==pred_date].merge(predictions_template_df.drop('confidenceValue',axis=1),'inner',on='assetCode')
#     print(list(fe_market.columns))
#     make_predictions(fe_market[must_match_cols])
#     env.predict(predictions_template_df)
#     stop = time.time()
#     print("iteration in seconds: \n",stop-start)
# print('Done!')
# New way
# about 15.5 seconds per iteration
#prediction time takes only .04 seconds
import numpy as np
import time
drop_cols=['assetCode','assetName','open','close','time','returnsOpenNextMktres10']
market_cols=['time', 'assetCode', 'volume',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']

#
for i,(market_obs_df, news_obs_df, predictions_template_df) in enumerate(days):

    start=time.time()
    print("\n round",i,"\n\n")
    # convert time to correct format
    #print("dtypes \n")
    #print(market_obs_df.dtypes,market_train_df.dtypes)
    market_obs_df['time']=market_obs_df['time'].dt.date
    market_obs_df['time']=pd.to_datetime(market_obs_df['time'])
    pred_date=pd.to_datetime(market_obs_df['time'].dt.date)[0]
    hist_date=pred_date - timedelta(days=34)
    print(pred_date)
    market_train_df=market_train_df[market_train_df['time']>=hist_date]
    ## fe_market is now the most recent 34 days of stock data
    
    #append new data with old
    market_train_df=market_train_df.append(market_obs_df[market_cols])
    #print("market_train_df cols part 2: ",list(market_train_df.columns))
    print(market_train_df.head())
    mid=time.time()
    print("pre-fe time in sec: \n",mid-start)
    fe_market=getFeatures(market_train_df,pred=True)
    
    print ("got features")
    ## make sure we only predict on latest date and relevant asset
    fe_market=fe_market[fe_market['assetCode'].isin(predictions_template_df.assetCode)]
    print(list(fe_market.columns))
    make_predictions(fe_market[must_match_cols])
    env.predict(predictions_template_df)
    stop = time.time()
    print("iteration in seconds: \n",stop-start)
print('Done!')

env.write_submission_file()
import os
print([filename for filename in os.listdir('.') if '.csv' in filename])
