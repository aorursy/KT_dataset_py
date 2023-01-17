import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import feather

import warnings

warnings.filterwarnings('ignore')

print(os.listdir('../input'))
# read files

ucf = pd.read_feather('../input/ucf-building-meter-reading/site0.ft')

test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv', parse_dates=['timestamp'], dtype={'row_id':'int32', 'building_id':'int16', 'meter':'int8',})

sub = pd.read_csv('../input/ashrae-half-and-half/submission.csv', dtype={'row_id':'int32', 'meter_reading':'float32'})
merged = test_df.merge(ucf, left_on=['building_id', 'meter', 'timestamp'], 

              right_on=['building_id', 'meter', 'timestamp'], how='left')
tmp = merged[~merged['meter_reading'].isna()][['row_id', 'meter_reading']]

tmp2 = sub[merged['meter_reading'].isna()]

final = pd.concat([tmp, tmp2], axis=0).reset_index(drop=True).sort_values(by='row_id')

final['row_id'] = final['row_id'].astype('int32')

final['meter_reading'] = final['meter_reading'].astype('float32')

final.to_csv('submission.csv', chunksize=25000, index=False)

final.head()
print("final mean:", final['meter_reading'].mean())

print("final std:", final['meter_reading'].std())

print("final min:", final['meter_reading'].min())

print("final max:", final['meter_reading'].max())
# evaluation functions

def rmse(ytrue, ypred):

    return np.sqrt(np.mean(np.square(ypred - ytrue), axis=0))

def rmsle(ytrue, ypred):

    return np.sqrt(np.mean(np.square(np.log1p(ypred) - np.log1p(ytrue)), axis=0))
ytrue = merged[~merged['meter_reading'].isna()].sort_values(by='row_id')['meter_reading']

pred = sub[~merged['meter_reading'].isna()].sort_values(by='row_id')['meter_reading']
print(f'RMSLE of buildings 0-104: {rmsle(ytrue, pred):.3f}')
df_true = merged[~merged['meter_reading'].isna()].sort_values(by='row_id')
df_pred = test_df.merge(sub, right_on='row_id', left_on='row_id', how='inner')

df_pred = df_pred[~merged['meter_reading'].isna()].sort_values(by='row_id')
# plot all predicted meter 1 by building_id from 2017-2018



meter = 1

buildings = set(range(105)).intersection(set(df_pred[df_pred['meter']==meter]['building_id'].unique()))



for i, building in enumerate(sorted(buildings)):

    fig, ax = plt.subplots(figsize=(15,1))

    plt.title(f"Building {building} Meter {meter}")

    # plot meter_reading

    idx = (df_pred['building_id'] == building) & (df_pred['meter'] == meter) 

    dates = matplotlib.dates.date2num(df_pred.loc[idx, 'timestamp'])

    plt.plot_date(dates, df_pred.loc[idx, 'meter_reading'], '-', label='meter_reading')

    plt.show()
# plot all ground truth (GT) meter 1 by building_id from 2017-2018



meter = 1

buildings = set(range(105)).intersection(set(df_true[df_true['meter']==meter]['building_id'].unique()))



for i, building in enumerate(sorted(buildings)):

    fig, ax = plt.subplots(figsize=(15,1))

    plt.title(f"Building {building} Meter {meter}")

    # plot meter_reading

    idx = (df_true['building_id'] == building) & (df_true['meter'] == meter) 

    dates = matplotlib.dates.date2num(df_true.loc[idx, 'timestamp'])

    plt.plot_date(dates, df_true.loc[idx, 'meter_reading'], '-', label='meter_reading')

    plt.show()