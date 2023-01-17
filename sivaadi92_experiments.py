# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Ctrl + Shift + P
import os
print(os.listdir("../input"))

toy = True
# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

# train_my_model(market_train_df, news_train_df)
print(market_train_df.shape, news_train_df.shape)
print(market_train_df.columns)
print(news_train_df.columns)
# We will reduce the number of samples for memory reasons
toy = True

if toy:
    market_train_df = market_train_df.tail(100_000)
    news_train_df = news_train_df.tail(300_000)
else:
    market_train_df = market_train_df.tail(3_000_000)
    news_train_df = news_train_df.tail(6_000_000)

print(market_train_df.shape)
print(news_train_df.shape)


import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.linear_model import LinearRegression


market_train_df.columns

X_train = market_train_df.iloc[:, 6:-2]
Y_train = market_train_df.iloc[:,-2]

# market_train_df.head(5)
market_train_df[market_train_df.assetCode.str.contains('SPY')]
# market_train_df[market_train_df.asse.str.startswith('f')]
# market_train_df['assetCode'].unique()
# market_train_df.loc[market_train_df['assetCode'] == 'MSFT']
#Experiment
columns = {'returnsClosePrevMktres1', 'returnsOpenPrevMktres1'}
market_train_df.columns.isin(columns)
# temp = market_train_df.loc[columns]
temp.head(5)

lm = LinearRegression()
X_train = X_train.ffill()
# X_train.isnull().any()
lm.fit(X_train, Y_train)
# Y_train.isnull().any()
def make_predictions(predictions_template_df, market_obs_df, news_obs_df):
    sample = market_obs_df.iloc[: , 6:]
    sample = sample.ffill()
#     y_pred = pd.DataFrame(0.9, len(sample))
    y_pred = np.array([0.9 for _ in range(len(sample))])
    random_noise = np.array([random.uniform(0, 0.01) for _ in range(len(sample))])
    y_pred = y_pred + random_noise
    predictions_template_df.confidenceValue = y_pred

days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
# sample = market_obs_df.iloc[: , 6:]
# sample = sample.ffill()
# y_pred = np.array([0.9 for _ in range(len(sample))])
# random_noise = np.array([random.uniform(0, 0.1) for _ in range(len(sample))])
# # random_noise
# y_pred = y_pred + random_noise
# predictions_template_df.confidenceValue = y_pred.clip(-1, 1)
# predictions_template_df
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(predictions_template_df, market_obs_df, news_obs_df)
    env.predict(predictions_template_df)

print('Done!')

env.write_submission_file()
print("Second run")