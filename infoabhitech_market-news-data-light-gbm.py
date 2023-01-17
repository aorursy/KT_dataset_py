# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
a=market_train_df['assetCode']
aa=set(a)
aaa = '|'.join(aa)
del a
del aa
ndf1 = news_train_df[news_train_df.assetCodes.str.contains(aaa)]
ndf1.head()
ac = ndf1['assetCodes']
del ndf1
del aaa
d=ac.to_frame()

d0=d[0:3300000]
d1=d[3300000:6600000]
d2=d[6600000:len(d)]

def fun_df(df):
    s = df.assetCodes.str.split(',',expand=True)
    s1=s.stack()
    s11 = s1.str.strip().reset_index(level=1, drop=True)
    df_a=s11.to_frame()
    df_a.rename(columns={0:'assetCode'},inplace=True)
    df_a.assetCode.replace({'{':''},regex=True,inplace=True)
    df_a.assetCode.replace({'}':''},regex=True,inplace=True)
    df_a.assetCode.replace({'\'':''},regex=True,inplace=True)
    ndf2=news_train_df.join(df_a).reset_index()
    ndf2.drop(columns={'assetCodes'},inplace=True)
    ndf3 = ndf2[ndf2.assetCode.notnull()]
    return ndf3
d01=fun_df(d0)
del d0
d02=fun_df(d1)
del d1
d03=fun_df(d2)
del d2

del news_train_df
df_fin1=d01.append(d02)
del d01
del d02

df_fin=df_fin1.append(d03)
del d03
del df_fin1
a=market_train_df['assetCode']
aa=set(a)
df_fin1 = df_fin[df_fin['assetCode'].isin(aa)]
del df_fin
del a
del aa
b=df_fin1['assetCode']
bb=set(b)
market_train_fin1 = market_train_df[market_train_df['assetCode'].isin(bb)]
del market_train_df
del b
del bb
df_fin1['time'] = df_fin1['time'].dt.strftime('%d-%m-%Y')
df_fin1['time'] = pd.to_datetime(df_fin1['time'],format="%d-%m-%Y")
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
df_fin1['marketCommentary'] = number.fit_transform(df_fin1['marketCommentary'].astype('str'))
news_train_fin1 = df_fin1[['time', 'assetCode', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 'marketCommentary', 'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']]
del df_fin1
news_train_fin1=news_train_fin1[news_train_fin1['time']>'2007-01-31']
temp=news_train_fin1.groupby(['assetCode','time'] ,axis=0).mean()
del news_train_fin1
news_train_fin=temp.reset_index()
del temp
news_train_fin['urgency'] = news_train_fin['urgency'].astype('int')
news_train_fin['takeSequence'] = news_train_fin['takeSequence'].astype('int')
news_train_fin['bodySize'] = news_train_fin['bodySize'].astype('int')
news_train_fin['companyCount'] = news_train_fin['companyCount'].astype('int')
news_train_fin['marketCommentary'] = news_train_fin['marketCommentary'].astype('int')
news_train_fin['sentenceCount'] = news_train_fin['sentenceCount'].astype('int')
news_train_fin['wordCount'] = news_train_fin['wordCount'].astype('int')
news_train_fin['firstMentionSentence'] = news_train_fin['firstMentionSentence'].astype('int')
news_train_fin['sentimentWordCount'] = news_train_fin['sentimentWordCount'].astype('int')
news_train_fin['noveltyCount12H'] = news_train_fin['noveltyCount12H'].astype('int')
news_train_fin['noveltyCount24H'] = news_train_fin['noveltyCount24H'].astype('int')
news_train_fin['noveltyCount3D'] = news_train_fin['noveltyCount3D'].astype('int')
news_train_fin['noveltyCount5D'] = news_train_fin['noveltyCount5D'].astype('int')
news_train_fin['noveltyCount7D'] = news_train_fin['noveltyCount7D'].astype('int')
news_train_fin['volumeCounts12H'] = news_train_fin['volumeCounts12H'].astype('int')
news_train_fin['volumeCounts24H'] = news_train_fin['volumeCounts24H'].astype('int')
news_train_fin['volumeCounts3D'] = news_train_fin['volumeCounts3D'].astype('int')
news_train_fin['volumeCounts5D'] = news_train_fin['volumeCounts5D'].astype('int')
news_train_fin['volumeCounts7D'] = news_train_fin['volumeCounts7D'].astype('int')
import warnings
warnings.filterwarnings('ignore')
market_train_fin1['time'] = market_train_fin1['time'].dt.strftime('%d-%m-%Y')
market_train_fin1['time'] = pd.to_datetime(market_train_fin1['time'],format="%d-%m-%Y")
market_train_fin1['returnsClosePrevMktres1']=market_train_fin1['returnsClosePrevMktres1'].fillna(market_train_fin1['returnsClosePrevRaw1'])
market_train_fin1['returnsOpenPrevMktres1']=market_train_fin1['returnsOpenPrevMktres1'].fillna(market_train_fin1['returnsOpenPrevRaw1'])
market_train_fin1['returnsClosePrevMktres10']=market_train_fin1['returnsClosePrevMktres10'].fillna(market_train_fin1['returnsClosePrevRaw10'])
market_train_fin1['returnsOpenPrevMktres10']=market_train_fin1['returnsOpenPrevMktres10'].fillna(market_train_fin1['returnsOpenPrevRaw10'])
market_train_fin1['return'] = (market_train_fin1['close'] - market_train_fin1['open']) / market_train_fin1['open']
market_train_fin = market_train_fin1[['assetCode', 'time', 'volume', 'return', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']]
del market_train_fin1
from sklearn.preprocessing import Normalizer
scaler =Normalizer()
market_train_fin['return'] = scaler.fit_transform(market_train_fin[['return']])
market_train_fin['returnsClosePrevRaw1'] = scaler.fit_transform(market_train_fin[['returnsClosePrevRaw1']])
market_train_fin['returnsOpenPrevRaw1'] = scaler.fit_transform(market_train_fin[['returnsOpenPrevRaw1']])
market_train_fin['returnsClosePrevMktres1'] = scaler.fit_transform(market_train_fin[['returnsClosePrevMktres1']])
market_train_fin['returnsOpenPrevMktres1'] = scaler.fit_transform(market_train_fin[['returnsOpenPrevMktres1']])
market_train_fin['returnsClosePrevRaw10'] = scaler.fit_transform(market_train_fin[['returnsClosePrevRaw10']])
market_train_fin['returnsOpenPrevRaw10'] = scaler.fit_transform(market_train_fin[['returnsOpenPrevRaw10']])
market_train_fin['returnsClosePrevMktres10'] = scaler.fit_transform(market_train_fin[['returnsClosePrevMktres10']])
market_train_fin['returnsOpenPrevMktres10'] = scaler.fit_transform(market_train_fin[['returnsOpenPrevMktres10']])
market_train_fin['returnsOpenNextMktres10'] = scaler.fit_transform(market_train_fin[['returnsOpenNextMktres10']])
dffin=pd.merge(market_train_fin, news_train_fin, on=['assetCode','time'])
del market_train_fin
del news_train_fin
dffin['day'], dffin['month'] = dffin['time'].dt.dayofweek, dffin['time'].dt.month
from sklearn.preprocessing import LabelEncoder
num = LabelEncoder()
dffin['assetCode'] = num.fit_transform(dffin['assetCode'].astype('str'))
dffin.index = dffin['time']
y = dffin['returnsOpenNextMktres10']
dffin = dffin.drop(['returnsOpenNextMktres10'], axis=1)
dffin = dffin.drop(['time'], axis=1)
dffin = dffin.drop(['assetCode'], axis=1)
import lightgbm as lgb
model = lgb.sklearn.LGBMRegressor(objective = 'regression_l1',
    learning_rate = 0.05,
    max_depth = 3,
    num_leaves = 100,
    min_data_in_leaf = 150,
    bagging_fraction = 0.75,
    bagging_freq = 2,
    feature_fraction = 0.5,
    lambda_l1 = 0,
    lambda_l2 = 1,
    seed = 50,
    metrics = 'l1',                              
    n_estimators=5000)
nrow = int(dffin.shape[0] * 0.8)
X_train, y_train = dffin.iloc[:nrow], y.iloc[:nrow]
X_valid, y_valid = dffin.iloc[nrow:], y.iloc[nrow:]
m = model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)] ,categorical_feature =['urgency','day','month','marketCommentary','firstMentionSentence'] ,verbose='false') 
preds = model.predict(X_valid)
y_preds=pd.Series(preds)
y_preds.index = y_valid.index
lgb.plot_importance(m,max_num_features=10)
import sklearn
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=False, random_state=None)
for train_index, valid_index in kfold.split(dffin, y):
    dffin_train, dffin_valid = dffin.iloc[train_index], dffin.iloc[valid_index] 
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)] ,categorical_feature =['urgency','day','month','marketCommentary','firstMentionSentence'] ,verbose='false')
    preds = model.predict(X_valid)
    y_preds=pd.Series(preds)
    y_preds.index = y_valid.index
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(y_valid)
    plt.plot(y_preds)
