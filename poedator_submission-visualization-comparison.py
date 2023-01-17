KAGGLE_MODE = True  # drives file loading
import numpy as np

import pandas as pd 



from tqdm import tqdm, tqdm_notebook

import gc

import zipfile

import os

import datetime



import matplotlib.pyplot as plt

%matplotlib inline  
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
%%time

# load files for test

if KAGGLE_MODE:

    building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

#     weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

#     weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

    train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

    test =  pd.read_csv("../input/ashrae-energy-prediction/test.csv")

else:

    zf = zipfile.ZipFile('./ashrae-energy-prediction.zip') 

    building_df = pd.read_csv(zf.open('building_metadata.csv'))

    train = pd.read_csv(zf.open('train.csv'))

#     weather_train = pd.read_csv(zf.open('weather_train.csv'))

#     weather_test = pd.read_csv(zf.open('weather_test.csv'))

    test =  pd.read_csv(zf.open('test.csv'))
def process_df(df):

    # adding timestamp, building_id and meter to submission data

    if not 'timestamp' in df.columns:

        df = pd.merge (df, test, on='row_id')

        df = df.drop(columns=['row_id'])



    # transforming timestamp

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["date"] = df["timestamp"].dt.date

    df["hour"] = df["timestamp"].dt.hour.astype(np.uint8)

    

    # aggregating data EDIT THIS PART TO GENERATE MORE STATISTICS

    df_daily = df.groupby(['building_id', 'date', 'meter']).agg({'meter_reading':['std','mean','count']}).reset_index()

    df_daily[('meter_reading','mean')] = np.log1p(df_daily[('meter_reading','mean')])

    df_daily[('meter_reading','std')] = np.log1p(df_daily[('meter_reading','std')])

    return df_daily
# loading and processing train data

dfs = [process_df(train)]

del train

gc.collect()

dfs[0].shape
# ls ../input/ -l
# Manually edit path, filenames and names. Names should correspond to filenames.

sub_path = '../input/'

sub_filenames = ["ashrae-baseline-lgbm/submission.csv","ashrae-half-and-half/submission.csv" ]

sub_names = ["baseline","half-and-half" ]
for sub_filename in sub_filenames:

    print(f'adding submission {sub_path + sub_filename}: ', end='')

    print('loading...', end='')

    sub = pd.read_csv(sub_path + sub_filename)

    print(', processing...')

    dfs.append(process_df(sub))

    del sub

    gc.collect()

print('done!')
del test

gc.collect()
# function to generate chart data and build charts

def chart_submissions (building_id, meter=0):

    titles = ['2016 train data', *sub_names]

    tmp_df = [df[(df.building_id == building_id) & (df.meter == meter)] \

                [['date', 'meter_reading']].set_index('date') for df in dfs]

    if tmp_df[0].shape[0]:



        fig, axes = plt.subplots(nrows=len(tmp_df), figsize=(18, 2+len(tmp_df)*2),)

        fig.suptitle(f'Building {building_id}, meter {meter}', fontsize=18, y = 0.94)

        max_y = np.concatenate([df[('meter_reading','mean')].values for df in tmp_df]).max() *  1.05



        for i in range (3):

#             tmp_df[i][('meter_reading', 'std')].plot(ax=axes[i], label='log_std')

            tmp_df[i][('meter_reading', 'mean')].plot(ax=axes[i], label='log mean')

#             tmp_df[i][('meter_reading', 'count')].plot(ax=axes[i], label='count')

            axes[i].axvline(x=datetime.date(2018, 1, 1),  color='k', linestyle='--')



            axes[i].legend()

            axes[i].set_title(titles[i], fontsize=16, y = 0.8)

            axes[i].set_ylim(0,max_y)

        building_df[building_df.building_id==building_id]

    else:

        print (f"Building_id={building_id}, Meter={meter} combination not present..." )
chart_submissions (building_id=0, meter=0)

# some interesting ones: 107, 869, 1001, 0, 888