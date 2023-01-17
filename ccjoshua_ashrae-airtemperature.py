# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import gc

from  pathlib import Path

from datetime import datetime

import matplotlib.pyplot as plt



%matplotlib inline
path = Path('/kaggle/input/ashrae-energy-prediction/')

building_file = path / 'building_metadata.csv'

weather_file = path / 'weather_train.csv'

train_file = path / 'train.csv'

test_file = path / 'test.csv'

weathertest_file = path / 'weather_test.csv'

#sample_file = path / 'sample_submission.csv'
%%time

mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

wt = pd.read_csv(weather_file, parse_dates=['timestamp'], date_parser=mydateparser )

testwt = pd.read_csv(weathertest_file, parse_dates=['timestamp'], date_parser=mydateparser)
allwt = pd.concat([wt, testwt], axis=0)
allwt
from fastai.tabular.transform import add_datepart

add_datepart(allwt, field_name='timestamp', prefix="ts", drop=False, time=True)
allwt.tsYear
allwt["nYear"] = allwt.tsYear  - 2016

allwt["nMonth"] = allwt.nYear * 12 + allwt.tsMonth
wts = allwt.groupby(['site_id', 'nMonth'])['air_temperature'].agg(['min', 'max', 'mean']).reset_index()
f, axes = plt.subplots(4, 4,figsize=(25,30))

plt.subplots_adjust(hspace=0.5)

for i in range(0,16):

    wts[ (wts.site_id == i) & (wts.nMonth.isin(range(1,13)))].plot('nMonth', ['min', 'mean', 'max'], 

                                       title=f'site:{i}', 

                                       figure = f, ax=axes[ (i// 4), (i % 4) ], 

                                       grid=True)
f, axes = plt.subplots(4, 4,figsize=(25,30) , sharey = True)

plt.subplots_adjust(hspace=0.5)

for i in range(0,16):

    wts[ (wts.site_id == i) & (wts.nMonth.isin(range(1,13)))].plot('nMonth', ['min', 'mean', 'max'], title=f'site:{i}', figure = f, ax=axes[ (i//4), (i % 4) ])
f, axes = plt.subplots(4, 4,figsize=(25,30))

plt.subplots_adjust(hspace=0.5)

for i in range(0,16):

    wts[ (wts.site_id == i) & (wts.nMonth.isin(range(1,37)))].plot('nMonth', ['min', 'mean', 'max'], 

                                       title=f'site:{i}', 

                                       figure = f, ax=axes[ (i // 4), (i % 4) ], 

                                       grid=True)
f, axes = plt.subplots(4, 4,figsize=(25,30), sharey=True)

plt.subplots_adjust(hspace=0.5)

for i in range(0,16):

    wts[ (wts.site_id == i) & (wts.nMonth.isin(range(1,37)))].plot('nMonth', ['min', 'mean', 'max'], 

                                                           title=f'site:{i}', 

                                                           figure = f, ax=axes[ (i//4), (i % 4) ], )
heatdf = wts [ (wts.site_id == 0) & (wts.nMonth.isin(range(1,13))) ][['mean']].reset_index()

heatdf.index = heatdf.index + 1

heatdf = heatdf.T

heatdf.drop('index', axis=0, inplace=True)

for i in range(1,16):

    temp = wts [ (wts.site_id == i) & (wts.nMonth.isin(range(1,13))) ][['mean']].reset_index()

    temp.index = temp.index + 1

    temp = temp.T

    heatdf = pd.concat( [heatdf, temp.drop('index', axis=0)], axis=0 )
heatdf
fig = plt.figure( figsize=(25,15))

ax = fig.add_subplot(111)

ax.set_xlabel('Month in 2016')

ax.set_title(label='HeatMap for Average Temperature in 2016 for 16 station ')



ax.set_xticks(np.arange(13))

ax.set_yticks(np.arange(16))

ax.set_xticklabels( [ i for i in range(1,13)] )

ax.set_yticklabels( [ f'site_{i}' for i in range(0,16) ])



# Loop over data dimensions and create text annotations.

for i in range(12):

    for j in range(16):

        text = ax.text(i, j, 

                       heatdf.iloc[j, i].round(2),

                       ha="center", va="center", color="black")



im = ax.imshow( heatdf , interpolation='nearest' , cmap='coolwarm')

plt.show()