import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kaggle.competitions import twosigmanews
import datetime
import time

env = twosigmanews.make_env()
def prepare_market_data(market_df):
    market_df['ratio'] = market_df['close'] / market_df['open']
    market_df['average'] = (market_df['close'] + market_df['open'])/2
    market_df['pricevolume'] = market_df['volume'] * market_df['close']
    market_df.drop(['assetName', 'volume'], axis=1, inplace=True)

    return market_df
def prepare_news_data(news_df):
    news_df['position'] = news_df['firstMentionSentence'] / news_df['sentenceCount']
    news_df['coverage'] = news_df['sentimentWordCount'] / news_df['wordCount']

    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'urgency','wordCount','sentimentWordCount']
    news_df.drop(droplist, axis=1, inplace=True)

    # create a mapping between 'assetCode' to 'news_index'
    assets = []
    indices = []
    for i, values in news_df['assetCodes'].iteritems():
        assetCodes = eval(values)
        assets.extend(assetCodes)
        indices.extend([i]*len(assetCodes))
    mapping_df = pd.DataFrame({'news_index': indices, 'assetCode': assets})
    del assets, indices
    
    # join 'news_train_df' and 'mapping_df' (effectivly duplicating news entries)
    news_df['news_index'] = news_df.index.copy()
    expanded_news_df = mapping_df.merge(news_df, how='left', on='news_index')
    del mapping_df, news_df
    
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
# market_df = market_df.tail(10_000)
# news_df = news_df.tail(30_000)

print('preparing data...')
start = datetime.date(2009,1,1)
merged_df = prepare_data(market_df, news_df, start)
print('Ready!')
train_columns = [x for x in merged_df.columns if x not in ['assetCode', 'time', 'returnsOpenNextMktres10', 'universe']]
X_train = merged_df[train_columns].values
y_train = (merged_df.returnsOpenNextMktres10 >= 0).astype(int).values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)
from xgboost import XGBClassifier
import time

print('Training XGBoost')
t = time.time()

xgb_market = XGBClassifier(n_jobs=4, n_estimators=200, max_depth=8, eta=0.05)
xgb_market.fit(X_train, y_train)
print(f'Done, time = {time.time() - t}s')
print("generating predictions...")
days = env.get_prediction_days()

for market_df, news_df, pred_template_df in days:
    test_df = prepare_data(market_df, news_df, start)
    X_test = test_df[train_columns].values
    preds = xgb_market.predict_proba(X_test)[:,1] * 2 - 1
    preds_df = pd.DataFrame({'assetCode':test_df['assetCode'],'confidenceValue':preds})
    env.predict(preds_df)
env.write_submission_file()
