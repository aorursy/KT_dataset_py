import pickle
import pandas as pd
from pandas import DataFrame, Series
from kaggle.competitions import twosigmanews
import gc
env = twosigmanews.make_env()
market_train_df, news_train_df = env.get_training_data()
market_train_df.head()
news_train_df.head()
market_train_df.to_pickle('market_train.zip')
news_train_df.to_pickle('news_train.zip')
del market_train_df
del news_train_df
test_market_df=DataFrame()
test_news_df=DataFrame()
for (market_df, news_df, prd_tmp) in env.get_prediction_days():
    test_market_df=pd.concat([test_market_df, market_df])
    test_news_df=pd.concat([test_news_df, news_df])
    prd_tmp.confidenceValue = 0.0
    env.predict(prd_tmp)
test_market_df.to_pickle('market_test.zip')
test_news_df.to_pickle('news_test.zip')