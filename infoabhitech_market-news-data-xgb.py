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
import matplotlib.pyplot as plt

import seaborn as sns
# Using Kaggle provided function to load 2 sets of training data - Market & News



from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()

print('Done!')

(market_train_df, news_train_df) = env.get_training_data()
print("Market training data -->", len(market_train_df))
print(market_train_df.dtypes)
print("                     NULL value count")

print(market_train_df.isnull( ).sum( ))
market_train_df.head()
print("News training data -->", len(news_train_df))
news_train_df.head()
print("             NULL value count")

print(news_train_df.isnull( ).sum( ))
# As target variable lies in market data , unique asset codes is filter out 

# Delete objects which no longer be required due to memory constraints



ac=market_train_df['assetCode']

acu=set(ac)

acuj = '|'.join(acu)

del ac

del acu
# Use unique asset code from market data to filter news data



ndf1 = news_train_df[news_train_df.assetCodes.str.contains(acuj)]

print ("News training data reduced to -->", len(ndf1))
ac = ndf1['assetCodes']

del ndf1

del acuj
# News data asset code is in the format {'GOOG.O', 'GOOG.OQ', 'GOOGa.DE'},a function is created to normalise it to separate asset code series form



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
# Due to memory limits news data segregated to 3 parts , apply the function and then join all of them



d=ac.to_frame()



d0=d[0:3300000]

d1=d[3300000:6600000]

d2=d[6600000:len(d)]



del ac
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
# Now asset code in market and news data are in same format , again filter out market asset code from news data



ac=market_train_df['assetCode']

acu=set(ac)

del ac
df_fin1 = df_fin[df_fin['assetCode'].isin(acu)]

del df_fin

del acu
ac=df_fin1['assetCode']

acu=set(ac)
market_train_fin1 = market_train_df[market_train_df['assetCode'].isin(acu)]

del market_train_df

del ac

del acu
# Change time format to remove hours , minutes , UTC time zone



df_fin1['time'] = df_fin1['time'].dt.strftime('%d-%m-%Y')

df_fin1['time'] = pd.to_datetime(df_fin1['time'],format="%d-%m-%Y")
# Encode True / False to binary 1 / 0



from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

df_fin1['marketCommentary'] = number.fit_transform(df_fin1['marketCommentary'].astype('str'))
# Mean of various similar features



df_fin1['noveltyCount'] = df_fin1[['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D']].mean(axis=1)
df_fin1['volumeCounts'] = df_fin1[['volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']].mean(axis=1)
# Selecting important features from news data based on domain understanding and given requirement

news_train_fin1 = df_fin1[['time', 'assetCode', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 'marketCommentary', 'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 'sentimentWordCount', 'noveltyCount', 'volumeCounts']] 

del df_fin1
# Jan 2007 has news data but no corresponding market data , filter out news data

# News data has several entries for a single day , group by is used to take average of all features to make one one entry per day



news_train_fin1=news_train_fin1[news_train_fin1['time']>'2007-01-31']

temp=news_train_fin1.groupby(['assetCode','time'] ,axis=0).mean()

del news_train_fin1

news_train_fin=temp.reset_index()

del temp
# After taking mean , values are in float , converting them to int so as to restore feature importance which were earlier in int format



news_train_fin['urgency'] = news_train_fin['urgency'].astype('int')

news_train_fin['takeSequence'] = news_train_fin['takeSequence'].astype('int')

news_train_fin['bodySize'] = news_train_fin['bodySize'].astype('int')

news_train_fin['companyCount'] = news_train_fin['companyCount'].astype('int')

news_train_fin['marketCommentary'] = news_train_fin['marketCommentary'].astype('int')

news_train_fin['sentenceCount'] = news_train_fin['sentenceCount'].astype('int')

news_train_fin['wordCount'] = news_train_fin['wordCount'].astype('int')

news_train_fin['firstMentionSentence'] = news_train_fin['firstMentionSentence'].astype('int')

news_train_fin['sentimentWordCount'] = news_train_fin['sentimentWordCount'].astype('int')

news_train_fin['noveltyCount'] = news_train_fin['noveltyCount'].astype('int')

news_train_fin['volumeCounts'] = news_train_fin['volumeCounts'].astype('int')
# Change time format to remove hours , minutes , UTC time zone



market_train_fin1['time'] = market_train_fin1['time'].dt.strftime('%d-%m-%Y')

market_train_fin1['time'] = pd.to_datetime(market_train_fin1['time'],format="%d-%m-%Y")
# Treat null values in market data by filling it with similar column values 



market_train_fin1['returnsClosePrevMktres1']=market_train_fin1['returnsClosePrevMktres1'].fillna(market_train_fin1['returnsClosePrevRaw1'])

market_train_fin1['returnsOpenPrevMktres1']=market_train_fin1['returnsOpenPrevMktres1'].fillna(market_train_fin1['returnsOpenPrevRaw1'])

market_train_fin1['returnsClosePrevMktres10']=market_train_fin1['returnsClosePrevMktres10'].fillna(market_train_fin1['returnsClosePrevRaw10'])

market_train_fin1['returnsOpenPrevMktres10']=market_train_fin1['returnsOpenPrevMktres10'].fillna(market_train_fin1['returnsOpenPrevRaw10'])
# Create a new feature in market data from asset opening and closing price



market_train_fin1['return'] = (market_train_fin1['close'] - market_train_fin1['open']) / market_train_fin1['open']
# Selecting important features from market data based on domain understanding and given requirement



market_train_fin = market_train_fin1[['assetCode', 'time', 'volume', 'return', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']]

del market_train_fin1
# Normalise market data column values them to range [-1,10] as variance for positive values is more



market_train_fin['return'] = np.where(market_train_fin['return'] > 1,10.0,market_train_fin['return'])

market_train_fin['return'] = np.where(market_train_fin['return'] < -1,-1.0,market_train_fin['return'])



market_train_fin['returnsClosePrevRaw1'] = np.where(market_train_fin['returnsClosePrevRaw1'] > 1,1.0,market_train_fin['returnsClosePrevRaw1'])

market_train_fin['returnsClosePrevRaw1'] = np.where(market_train_fin['returnsClosePrevRaw1'] < -0.5,-1.0,market_train_fin['returnsClosePrevRaw1'])



market_train_fin['returnsOpenPrevRaw1'] = np.where(market_train_fin['returnsOpenPrevRaw1'] > 1,1.0,market_train_fin['returnsOpenPrevRaw1'])

market_train_fin['returnsOpenPrevRaw1'] = np.where(market_train_fin['returnsOpenPrevRaw1'] < -0.5,-1.0,market_train_fin['returnsOpenPrevRaw1'])



market_train_fin['returnsClosePrevMktres1'] = np.where(market_train_fin['returnsClosePrevMktres1'] > 1,1.0,market_train_fin['returnsClosePrevMktres1'])

market_train_fin['returnsClosePrevMktres1'] = np.where(market_train_fin['returnsClosePrevMktres1'] < -0.5,-1.0,market_train_fin['returnsClosePrevMktres1'])



market_train_fin['returnsOpenPrevMktres1'] = np.where(market_train_fin['returnsOpenPrevMktres1'] > 1,1.0,market_train_fin['returnsOpenPrevMktres1'])

market_train_fin['returnsOpenPrevMktres1'] = np.where(market_train_fin['returnsOpenPrevMktres1'] < -0.5,-1.0,market_train_fin['returnsOpenPrevMktres1'])



market_train_fin['returnsClosePrevRaw10'] = np.where(market_train_fin['returnsClosePrevRaw10'] > 1,1.0,market_train_fin['returnsClosePrevRaw10'])

market_train_fin['returnsClosePrevRaw10'] = np.where(market_train_fin['returnsClosePrevRaw10'] < -0.5,-1.0,market_train_fin['returnsClosePrevRaw10'])



market_train_fin['returnsOpenPrevRaw10'] = np.where(market_train_fin['returnsOpenPrevRaw10'] > 1,1.0,market_train_fin['returnsOpenPrevRaw10'])

market_train_fin['returnsOpenPrevRaw10'] = np.where(market_train_fin['returnsOpenPrevRaw10'] < -0.5,-1.0,market_train_fin['returnsOpenPrevRaw10'])



market_train_fin['returnsClosePrevRaw10'] = np.where(market_train_fin['returnsClosePrevRaw10'] > 1,1.0,market_train_fin['returnsClosePrevRaw10'])

market_train_fin['returnsClosePrevRaw10'] = np.where(market_train_fin['returnsClosePrevRaw10'] < -0.5,-1.0,market_train_fin['returnsClosePrevRaw10'])



market_train_fin['returnsOpenPrevRaw10'] = np.where(market_train_fin['returnsOpenPrevRaw10'] > 1,1.0,market_train_fin['returnsOpenPrevRaw10'])

market_train_fin['returnsOpenPrevRaw10'] = np.where(market_train_fin['returnsOpenPrevRaw10'] < -0.5,-1.0,market_train_fin['returnsOpenPrevRaw10'])



market_train_fin['returnsOpenNextMktres10'] = np.where(market_train_fin['returnsOpenNextMktres10'] > 1,1.0,market_train_fin['returnsOpenNextMktres10'])

market_train_fin['returnsOpenNextMktres10'] = np.where(market_train_fin['returnsOpenNextMktres10'] < -0.5,-1.0,market_train_fin['returnsOpenNextMktres10'])
# Merge market and news data based on asset code and date



dffin=pd.merge(market_train_fin, news_train_fin, on=['assetCode','time'])
del market_train_fin

del news_train_fin
dffin.index = dffin['time']
# Take out target variable



y = dffin['returnsOpenNextMktres10']
# Drop unwanted columns



dffin = dffin.drop(['returnsOpenNextMktres10'], axis=1)

dffin = dffin.drop(['time'], axis=1)

dffin = dffin.drop(['assetCode'], axis=1)
import xgboost as xgb

from xgboost import XGBRegressor
# Create XGBoost Model with parameters tuned



xgb_model = xgb.XGBRegressor(objective = 'reg:linear',

    booster = 'gbtree',

    learning_rate = 0.3,

    max_depth = 6,

    reg_alpha=0, 

    reg_lambda=1,

    nthread=2048,

    seed = 100,                             

    n_estimators=10000,

    subsample = 0.5,

    early_stopping_rounds=10)
# Split train test data



nrow = int(dffin.shape[0] * 0.8)

X_train, y_train = dffin.iloc[:nrow], y.iloc[:nrow]

X_valid, y_valid = dffin.iloc[nrow:], y.iloc[nrow:]
# Train model 



mxgb = xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose='true')
# Predict values and make it a series



predsxgb = xgb_model.predict(X_valid)

y_predsxgb=pd.Series(predsxgb)

y_predsxgb.index = y_valid.index
# Plot feature importance



xgb.plot_importance(mxgb,max_num_features=10)
# Plot validation and prediction values in confidence interval [-1,1]



plt.figure(figsize=(10,5))

plt.plot(y_valid,label='Validation')

plt.plot(y_predsxgb,label='Prediction')
# Histogram of validation values



y_valid.plot.hist(label='Validation')
# Histogram of predicted values



y_predsxgb.plot.hist(label='Prediction')
# Kdeplot of validation values



sns.kdeplot(y_valid,label='Validation')
# Kdeplot of prediction values



sns.kdeplot(y_predsxgb,label='Prediction')
y_valid.plot.line(label='Validation')
y_predsxgb.plot.line(label='Prediction')