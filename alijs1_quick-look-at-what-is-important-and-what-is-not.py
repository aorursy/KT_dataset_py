import numpy as np 

import pandas as pd 

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")

print("Train shape: {}".format(df.shape))
# Values from top public kernel https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189

low_y_cut = -0.086092

high_y_cut = 0.093496



print("Preparing data for model...")

df = df.sample(frac=0.1)

df.fillna(df.mean(axis=0), inplace=True)

y_is_within_cut = ((df['y'] > low_y_cut) & (df['y'] < high_y_cut))



train_X = df.loc[y_is_within_cut, df.columns[2:-1]]

train_y = df.loc[y_is_within_cut, 'y'].values.reshape(-1, 1)

print("Data for model: X={}, y={}".format(train_X.shape, train_y.shape))
import xgboost as xgb

model = xgb.XGBRegressor()

print("Fitting...")

model.fit(train_X, train_y)

print("Fitting done")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 30))

xgb.plot_importance(model, ax=ax)

print("Features importance done")