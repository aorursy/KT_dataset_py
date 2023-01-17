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
toy = False
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
market_train_df, _ = env.get_training_data()

market_train_df.shape
market_train_df.tail()
if toy:
    market_train_df = market_train_df.tail(1_000_000)
else:
    market_train_df = market_train_df.tail(2_000_000)
market_train_df.columns
metrics = ["mean", "std", "sum", "min", "max"]
cols = ['volume', 'close', 'open', 
        'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10']

aggs = {col: metrics for col in cols}
assets_df = market_train_df.groupby("assetCode").agg(aggs)
assets_df.columns = ["{}_{}".format(col, metric) for col, metric in assets_df.columns.values]
assets_df.head()
market_train_df = pd.merge(market_train_df, assets_df, on="assetCode", how="left")
market_train_df.head()
feature_cols = assets_df.columns.values.tolist() + [feat for feat in cols if feat != 'returnsOpenNextMktres10']
fcols = [f for f in feature_cols if f in market_train_df.columns]


X = market_train_df[fcols].values
y = market_train_df['returnsOpenNextMktres10'].values
universe = market_train_df['universe'].values

del market_train_df
import gc; gc.collect()
import time; time.sleep(20)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y, train_universe, val_universe = \
    train_test_split(X, y, universe, test_size=0.1, shuffle=True)
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(verbose=2, n_estimators=100)
model.fit(np.nan_to_num(train_X), np.nan_to_num(train_y))
from sklearn.metrics import mean_squared_error
mean_squared_error(
    y_true=np.nan_to_num(val_y[val_universe==1]), 
    y_pred=model.predict(np.nan_to_num(val_X[val_universe==1, :]))
)
import matplotlib.pyplot as plt
_, _, _ = plt.hist(np.nan_to_num(val_y[val_universe==1]), bins=100)
_, _, _ = plt.hist(model.predict(np.nan_to_num(val_X[val_universe==1, :])), bins=100)
dict(zip(fcols, model.feature_importances_.tolist()))
days = env.get_prediction_days()

for market_obs_df, _, predictions_template_df in days:
    market_obs_df = pd.merge(market_obs_df, assets_df, on="assetCode", how="left")
    X = np.nan_to_num(market_obs_df[fcols].values)
    predictions_template_df.confidenceValue = np.clip(model.predict(X), -1, 1)
    env.predict(predictions_template_df)
print('Done!')
env.write_submission_file()