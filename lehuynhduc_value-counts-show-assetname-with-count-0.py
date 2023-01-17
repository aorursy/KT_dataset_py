import numpy as np
import lightgbm as lgb
import pandas as pd
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df.copy(), news_train_df.copy()
news_train.assetName.value_counts()
