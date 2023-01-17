import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import gc
from kaggle.competitions import twosigmanews

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Get 2Sigma environment
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df.shape

news_train_df.shape
market_train_df.head(5)
news_train_df.head()
gc.collect()
merge_df = pd.merge(market_train_df, news_train_df, how='inner',left_on=['assetName','time'], right_on = ['assetName','time'])
merge_df.shape
merge_df.head(10)
merge_df.isna().sum()
merge_df.dropna()
merge_df.isna().sum()
