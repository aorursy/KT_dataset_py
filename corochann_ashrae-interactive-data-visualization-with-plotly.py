import gc

import os

from pathlib import Path

import random

import sys



import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff
%%time

root = Path('/kaggle/input/ashrae-feather-format-for-fast-loading')





train_df = pd.read_feather(root/'train.feather')

test_df = pd.read_feather(root/'test.feather')

weather_train_df = pd.read_feather(root/'weather_train.feather')

weather_test_df = pd.read_feather(root/'weather_test.feather')

building_meta_df = pd.read_feather(root/'building_metadata.feather')

#sample_submission = pd.read_feather(root/'sample_submission.feather')
print('train_df', train_df.shape)

print('test_df', test_df.shape)
debug = False

if debug:

    train_df = train_df.iloc[:10000]
meter_type_list = ["Electricity", "ChilledWater", "Steam", "HotWater"]

cnt_srs = train_df["meter"].value_counts()

cnt_srs_df = cnt_srs.to_frame()



# cnt_srs_df['meter_type'] = cnt_srs_df.index

cnt_srs_df['meter_type'] = meter_type_list

cnt_srs_df
fig = px.bar(cnt_srs_df, x='meter_type', y='meter', title='Number of rows for each meter type')

# fig.update_layout(showlegend=True)

fig.show()
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
sampled_train_df = train_df.sample(10000)
# It takes long time for whole train_df, so use...



fig = px.histogram(sampled_train_df, x="meter_reading_log1p", color="meter", cumulative=False, opacity=0.4)

fig.show()
fig = ff.create_distplot([sampled_train_df[sampled_train_df['meter'] == i]['meter_reading_log1p'] for i in range(4)],

                         meter_type_list, bin_size=0.2, histnorm='probability')

fig.show()
# target_building_id = 184

target_building_id = 1298

temp_df = train_df[train_df["building_id"]==target_building_id].reset_index(drop=True)



fig = px.line(temp_df, x='timestamp', y='meter_reading_log1p', color='meter')

fig.show()
weather_train_df.describe()
temp_weather_train_df = weather_train_df[(weather_train_df['site_id'] == 0) | (weather_train_df['site_id'] == 1) | (weather_train_df['site_id'] == 2)]
fig = px.line(temp_weather_train_df, x='timestamp', y='air_temperature', color='site_id')

fig.show()
fig = px.line(temp_weather_train_df, x='timestamp', y='cloud_coverage', color='site_id')

fig.show()
fig = px.line(temp_weather_train_df, x='timestamp', y='dew_temperature', color='site_id')

fig.show()
fig = px.line(temp_weather_train_df, x='timestamp', y='precip_depth_1_hr', color='site_id')

fig.show()
fig = px.line(temp_weather_train_df, x='timestamp', y='sea_level_pressure', color='site_id')

fig.show()
fig = px.line(temp_weather_train_df, x='timestamp', y='wind_direction', color='site_id')

fig.show()
weather_train_df.head()