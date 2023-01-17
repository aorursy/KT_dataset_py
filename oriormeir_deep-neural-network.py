import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kaggle.competitions import twosigmanews
import datetime
import time

env = twosigmanews.make_env()
def prepare_market_data(market_df):    
    market_df.drop(['assetName', 'volume'], axis=1, inplace=True)
    
    return market_df
def prepare_news_data(news_train_df):
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'urgency','wordCount','sentimentWordCount']
    news_train_df.drop(droplist, axis=1, inplace=True)

    # create a mapping between 'assetCode' to 'news_index'
    assets = []
    indices = []
    for i, values in news_train_df['assetCodes'].iteritems():
        assetCodes = eval(values)
        assets.extend(assetCodes)
        indices.extend([i]*len(assetCodes))
    mapping_df = pd.DataFrame({'news_index': indices, 'assetCode': assets})
    del assets, indices
    
    # join 'news_train_df' and 'mapping_df' (effectivly duplicating news entries)
    news_train_df['news_index'] = news_train_df.index.copy()
    expanded_news_df = mapping_df.merge(news_train_df, how='left', on='news_index')
    del mapping_df, news_train_df
    
    expanded_news_df.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return expanded_news_df.groupby(['time', 'assetCode']).mean().reset_index()
def prepare_data(market_df, news_df, start=None):
    market_df['time'] = market_df['time'].dt.date
    news_df['time'] = news_df['time'].dt.date
    if start is not None:
        market_df = market_df[market_df['time'] >= start].reset_index(drop=True)
        news_df = news_df[news_df['time'] >= start].reset_index(drop=True)

    market_df = prepare_market_data(market_df)
    news_df = prepare_news_data(news_df)

    # join news_df to market_df using ['assetCode', 'time']
    return market_df.merge(news_df, how='left', on=['assetCode', 'time']).fillna(0)
(market_df, news_df) = env.get_training_data()

# TODO: remove this
# market_df = market_df.tail(100_000)
# news_df = news_df.tail(300_000)

print('preparing data...')
start = datetime.date(2009,1,1)
merged_df = prepare_data(market_df, news_df, start)

train_columns = [x for x in merged_df.columns if x not in ['assetCode', 'time', 'returnsOpenNextMktres10', 'universe']]
X = merged_df[train_columns].values
y = (merged_df.returnsOpenNextMktres10 >= 0).astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)
print('Ready!')
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Input
from keras.optimizers import Adam

model = Sequential()

model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))

model.add(Dense(1))

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
#model.summary()
model.fit(x=X_train, y=y_train, epochs=5,shuffle=True,validation_data=(X_test, y_test))
