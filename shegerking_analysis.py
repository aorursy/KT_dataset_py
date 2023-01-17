# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization,Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
    curr_asset = market_train_df[market_train_df['assetName'] == asset]
    data.append(go.Scatter(
        x = curr_asset['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = curr_asset['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 10 random assets",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))

init_notebook_mode(connected=True)
iplot(dict(data=data, layout=layout), filename='basic-line')
#numer of assetNames
market_train_df["assetName"].unique().size
market_train_df[market_train_df['returnsClosePrevRaw1'] > 1].head()
def sampleAssetData(assetCode, date, numDays):
    d = datetime.strptime(date,'%Y-%m-%d')
    start = d - timedelta(days=numDays)
    end = d + timedelta(days=numDays)
    return market_train_df[(market_train_df['assetCode'] == assetCode)
                             & (market_train_df['time'] >= start.strftime('%Y-%m-%d'))
                             & (market_train_df['time'] <= end.strftime('%Y-%m-%d'))].copy()
sampleAssetData('EBR.N', '2016-10-13', 5)
#We can see the 
#check market data
market_train_df[market_train_df['assetCode'] == 'EBR.N'].head()


#lets only use the stock market values after 2009
market_train_df = market_train_df[market_train_df.time > '2009'].reset_index()
len(market_train_df)
df = market_train_df[market_train_df['assetCode'] == 'EBR.N']
market_train_df.iloc[df.index].head()
pd.options.mode.chained_assignment = None
class preprocess_market:
    def __init__(self, df, is_test = False):
        self.data = df.copy()
        self.scaler = MinMaxScaler()
        self.is_test = is_test
        if not is_test:
            self.y = self.data['returnsOpenNextMktres10']
    encode = []
        
    catagorical = ['assetCode']
    
#     numerical = ['volume', 'close', 'open',
#        'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
#        'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
#        'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
#        'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
#        'returnsOpenNextMktres10', 'universe']
        
    def transform(self):
        if self.is_test:
            Mktres = ['returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                     'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
        else:
            Mktres = ['returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
             'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
              'returnsOpenNextMktres10']
        self.data[Mktres] = self.data[Mktres].clip(-1, 1)
        self.data['assetCode'] = self.data['assetCode'].apply(lambda x : x.split('.')[0])
       #drop the traget column 
        self.data['volume'] = self.data['volume'] / self.data['volume'].mean()
       #add a feauture
        self.data['close/open'] = self.data['close'] / self.data['open']
#        self.data['ema_diff'] = self.data[['open']].ewm(span = 20, adjust = False).mean(
#        ) - self.data[['open']].ewm(span = 50, adjust = False).mean()
       #drop other unnecessary columns
        self.data.drop(['close_to_open', 'mean_close'], axis = 1, inplace = True)
    
        #
    def get_y(self, idx = None):
        if type(idx) == type(None):
           return self.y
        else:
           return self.y.iloc[idx.index]
        
    
    def adjust_time(self):
        self.data['month'] = self.data['time'].dt.month
        self.data['dayofweek'] = self.data['time'].dt.dayofweek
        self.data['time'] = self.data['time'].dt.date #index
        
        #drop the time columns
    
    #get back the dataframe
    def get_data(self):
        return self.data

    
    def get_scaler(self):
        return scaler
    
    
    #replace data with abnormal change in close to open ratio
    #taken from https://www.kaggle.com/artgor/eda-feature-engineering-and-everything
    def remove_variance(self):
        #percentage of change in close to open price
        self.data['close_to_open'] =  np.abs(self.data['close'] / self.data['open'])
        #add the mean of the opening and closing price as a feature
        self.data['mean_open'] = self.data.groupby('assetName')['open'].transform('mean')
        self.data['mean_close'] = self.data.groupby('assetName')['close'].transform('mean')
        
        
        for i, row in self.data.loc[self.data['close_to_open'] >= 2].iterrows():
            if np.abs(row['mean_open'] - row['open']) > np.abs(row['mean_close'] - row['close']):
                self.data.iloc[i,6] = row['mean_open']
            else:
                self.data.iloc[i,5] = row['mean_close']
        
        for i, row in self.data.loc[self.data['close_to_open'] <= 0.5].iterrows():
            if np.abs(row['mean_open'] - row['open']) > np.abs(row['mean_close'] - row['close']):
                self.data.iloc[i,6] = row['mean_open']
            else:
                self.data.iloc[i,5] = row['mean_close']
                
    def fit(self):
        self.adjust_time()
        self.remove_variance()
        self.transform()
#         self.normalize()
        
    
    
market = preprocess_market(market_train_df)
market.fit()

market_train = market.get_data()
len(market_train)
# free up some space
del market_train_df
market_train.head()
#lots of report by reuters likely to drop this feature
plt.figure(figsize = (10, 5))
(news_train_df['provider'].value_counts()/100)[:10].plot('bar')
#sentiment_class

plt.figure(figsize = (10, 5))
news_train_df['sentimentClass'].value_counts().plot('bar')
#urgency_class
#one or three
news_train_df['urgency'].value_counts().plot('bar')
#most articles generally have high relevance
sns.distplot(news_train_df['relevance'])
news_cols_numeric = ['urgency', 'takeSequence', 'wordCount', 'sentenceCount', 'companyCount',
                         'marketCommentary', 'relevance', 'sentimentNegative', 'sentimentNeutral',
                         'sentimentPositive', 'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
                         'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
                         'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']
flg, ax = plt.subplots(figsize = (10, 10))
corr = news_train_df[news_cols_numeric].corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
class process_news:
    def __init__(self, news_df):
        self.data = news_df.copy()

       
    def transform(self):
                
        dublet =  ['urgency', 'marketCommentary','relevance', 'sentimentClass',
       'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
       'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
       'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
       'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
       'volumeCounts7D']
        #replace null values with zero
     
        self.data['firstCreated'] = self.data['firstCreated'].dt.date
        self.data['assetCodes'] = self.data['assetCodes'].apply(lambda x: list(eval(x))[0]).apply(lambda x : x.split('.')[0])
        self.data[dublet] = self.data.groupby(['firstCreated', 'assetCodes'])[dublet].transform('mean')
        return self.data
news = process_news(news_train_df)
news_data = news.transform()
# # news_train_df['firstCreated'].dt.date
del news_train_df
del process_news
news_data.head()
def combine_sub(market_data, news_data, is_test = False):
    feature_cols = ['assetCode', 'volume', 'open', 
        'returnsOpenPrevRaw1', 'returnsOpenPrevMktres1', 'returnsOpenPrevRaw10',
        'returnsOpenPrevMktres10', 
        'month', 'close/open', 'urgency',
       'marketCommentary',
        'relevance', 'sentimentClass',
       'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
       'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
       'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
       'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
       'volumeCounts7D']
    
    if not is_test:
        feature_cols.append('returnsOpenNextMktres10')
    if not is_test:
        market = pd.merge(market_data, news_data, how='left', left_on=['time', 'assetCode'], 
                                right_on=['firstCreated', 'assetCodes'])
    else:
        market = pd.merge(market_data, news_data, how='left', left_on=['assetCode'], 
                                right_on=['assetCodes'])
    #enumerate asset code
    lbl = {k: v for v, k in enumerate(market['assetCode'].unique())}
    market['assetCode'] = market['assetCode'].map(lbl)
        
    market = market[feature_cols]
    market = market.drop_duplicates()
    
    target = None
    if not is_test:
        market.dropna(0, inplace = True)
        target = market['returnsOpenNextMktres10']
        market.drop(['returnsOpenNextMktres10'], axis = 1, inplace = True)
       
    else:
        market.fillna(0, inplace = True)
                      
    return market, target
market, label = combine_sub(market_train, news_data)
#write to csv file
# market.to_csv('train_modified', sep=',')
# label.to_csv('train_label', sep = ',')
label = label.apply(lambda x : 1 if x > 0 else -1)
#build graph

def build_graph(num_feat):
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(26, input_dim=num_feat, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1, kernel_initializer='uniform', activation='tanh'))

    model.compile(loss = 'mean_squared_error',
                 optimizer = optimizer, 
                 metrics = ['mse'])
    return model
num_feat = market.shape[1]
filepath="weights1.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model = build_graph(num_feat)

history = model.fit(market.values, label.values, validation_split=0.20, epochs=40, batch_size=256, callbacks=callbacks_list, verbose=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
class process_news_test:
    def __init__(self, news_df):
        self.data = news_df.copy()

       
    def transform(self):
                
        dublet =  ['urgency', 'marketCommentary','relevance', 'sentimentClass',
       'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
       'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
       'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
       'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
       'volumeCounts7D']
        #replace null values with zero
     
        self.data['firstCreated'] = self.data['firstCreated'].dt.date
        self.data['assetCodes'] = self.data['assetCodes'].apply(lambda x: list(eval(x))[0]).apply(lambda x : x.split('.')[0])
        self.data[dublet] = self.data.groupby(['assetCodes'])[dublet].transform('mean')
        return self.data
days = env.get_prediction_days()
model.load_weights("weights1.best.hdf5")
for (market_obs_df, news_obs_df, predictions_template_df) in days:
        
    market = preprocess_market(market_obs_df, is_test = True)
    market.fit()
    market_test = market.get_data()
    
    news = process_news_test(news_obs_df)
    news_test = news.transform()
    
    market_news, label = combine_sub(market_test, news_test, is_test = True)
    
    predictions_template_df.confidenceValue = model.predict(market_news.values)
    env.predict(predictions_template_df)
print('Done!')
env.write_submission_file()
