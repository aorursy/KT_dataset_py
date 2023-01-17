# SOURCE: https://www.kaggle.com/rabaman/0-64-in-100-lines
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

# # TODO: remove this
# market_df = market_df.tail(1_000_000)
# news_df = news_df.tail(3_000_000)

print('preparing data...')
start = datetime.date(2009,1,1)
merged_df = prepare_data(market_df, news_df, start)
print('Ready!')
train_columns = [x for x in merged_df.columns if x not in ['assetCode', 'time', 'returnsOpenNextMktres10']]
X = merged_df[train_columns].values
y = (merged_df.returnsOpenNextMktres10 >= 0).astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)
import lightgbm as lgb
import random
from sklearn.metrics import mean_squared_error

t = time.time()
print ('Tune hyperparameters for lightgbm')

train_set, val_set = lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)

# best_params = None
# best_loss = 100
# for _ in range(25):
#     # generate params
#     params = {"objective" : "binary",
#           "metric" : "binary_logloss",
#           "num_leaves" : random.choice([10, 25, 60, 75, 100]),
#           "max_depth": -1,
#           "learning_rate" : random.choice([0.1, 0.01, 0.08, 0.05, 0.001, 0.003]),
#           "bagging_fraction" : random.choice([0.7, 0.8, 0.9, 0.95]),  # subsample
#           "feature_fraction" : random.choice([0.7, 0.8, 0.9, 0.95]),  # colsample_bytree
#           "bagging_freq" : 5,        # subsample_freq
#           "bagging_seed" : 2018,
#           "verbosity" : -1 }
    
#     lgbm_model = lgb.train(params, train_set, 2000, valid_sets=[train_set, val_set], early_stopping_rounds=100, verbose_eval=False)
#     loss = mean_squared_error(lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration), y_test.astype(float))

#     if loss < best_loss:
#         best_params = params
#         best_loss = loss

best_params = {
    'objective': 'binary', 
    'metric': 'binary_logloss', 
    'num_leaves': 75, 
    'max_depth': -1, 
    'learning_rate': 0.05, 
    'bagging_fraction': 0.9, 
    'feature_fraction': 0.9, 
    'bagging_freq': 5, 
    'bagging_seed': 2018, 
    'verbosity': -1
}

print(f'Train again with the best params: {best_params}s')
lgbm_model = lgb.train(best_params, train_set, 2000, valid_sets=[train_set, val_set], early_stopping_rounds=100, verbose_eval=2000)

print(f'Done, time = {time.time() - t}s')
import matplotlib.pyplot as plt
import seaborn as sns

feat_importance = pd.DataFrame()
feat_importance["feature"] = train_columns
feat_importance["gain"] = lgbm_model.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(8,10))
ax = sns.barplot(y="feature", x="gain", data=feat_importance)
print("generating predictions...")

for market_df, news_df, pred_template_df in env.get_prediction_days():
    test_df = prepare_data(market_df, news_df, start)
    test_columns = [x for x in test_df.columns if x not in ['assetCode', 'time', 'returnsOpenNextMktres10']]
    X_test = test_df[test_columns].values
    preds = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration) * 2 - 1
    preds_df = pd.DataFrame({'assetCode':test_df['assetCode'],'confidenceValue':preds})
    env.predict(preds_df)
env.write_submission_file()
