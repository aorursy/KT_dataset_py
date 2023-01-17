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

from pathlib import Path

import seaborn as sns

import matplotlib.pyplot as plt
Input = Path('/kaggle/input/ashrae-feather-format-for-fast-loading')
lists = ['test.feather','train.feather','weather_train.feather','weather_test.feather','building_metadata.feather']

test = pd.read_feather(Input/lists[0])

train = pd.read_feather(Input/lists[1])

wtrain = pd.read_feather(Input/lists[2])

wtest = pd.read_feather(Input/lists[3])

bmeta = pd.read_feather(Input/lists[4])
cm = sns.light_palette("red", as_cmap=True)
table = wtrain.groupby(['site_id','timestamp']

               ,as_index=False)['air_temperature'].mean()

table = table.loc[(table.site_id==0)|(table.site_id==8)|(table.site_id==7)|(table.site_id==11)]

pd.pivot_table(table,index=['timestamp'],columns=['site_id']).corr(method='pearson').style.background_gradient(cmap=cm)
fig,ax =plt.subplots(1,2)

fig.set_size_inches(15, 3)

table1 = wtrain.loc[(wtrain.site_id==0)|(wtrain.site_id==8)].set_index('site_id')

pd.pivot_table(table1,index=['timestamp'],columns=['site_id']).air_temperature.plot(ax=ax[0])

table2 = wtrain.loc[(wtrain.site_id==7)|(wtrain.site_id==11)].set_index('site_id')

pd.pivot_table(table2,index=['timestamp'],columns=['site_id']).air_temperature.plot(ax=ax[1])
table = wtrain.groupby(['site_id','timestamp']

               ,as_index=False)['dew_temperature'].max()

table = table.loc[(table.site_id==0)|(table.site_id==8)|(table.site_id==7)|(table.site_id==11)]

pd.pivot_table(table,index=['timestamp'],columns=['site_id']).corr(method='pearson').style.background_gradient(cmap=cm)
fig,ax =plt.subplots(1,2)

fig.set_size_inches(15, 3)

table1 = wtrain.loc[(wtrain.site_id==0)|(wtrain.site_id==8)].set_index('site_id')

pd.pivot_table(table1,index=['timestamp'],columns=['site_id']).dew_temperature.plot(ax=ax[0])

table2 = wtrain.loc[(wtrain.site_id==7)|(wtrain.site_id==11)].set_index('site_id')

pd.pivot_table(table2,index=['timestamp'],columns=['site_id']).dew_temperature.plot(ax=ax[1])
table = wtrain.groupby(['site_id','timestamp']

               ,as_index=False)['cloud_coverage'].min()

table = table.loc[(table.site_id==0)|(table.site_id==8)|(table.site_id==7)|(table.site_id==11)]

pd.pivot_table(table,index=['timestamp'],columns=['site_id']).corr(method='pearson').style.background_gradient(cmap=cm)
fig,ax =plt.subplots(1,2)

fig.set_size_inches(15, 3)

table = wtrain.loc[(wtrain.site_id==0)|(wtrain.site_id==8)].set_index('site_id')

pd.pivot_table(table,index=['timestamp'],columns=['site_id']).cloud_coverage.plot(ax=ax[0])
table = wtrain.groupby(['site_id','timestamp']

               ,as_index=False)['sea_level_pressure'].mean()

table = table.loc[(table.site_id==0)|(table.site_id==8)|(table.site_id==7)|(table.site_id==11)]

pd.pivot_table(table,index=['timestamp'],columns=['site_id']).corr(method='pearson').style.background_gradient(cmap=cm)
fig,ax =plt.subplots(1,2)

fig.set_size_inches(15, 3)

table1 = wtrain.loc[(wtrain.site_id==0)|(wtrain.site_id==8)].set_index('site_id')

pd.pivot_table(table1,index=['timestamp'],columns=['site_id']).sea_level_pressure.plot(ax=ax[0])

table2 = wtrain.loc[(wtrain.site_id==7)|(wtrain.site_id==11)].set_index('site_id')

pd.pivot_table(table2,index=['timestamp'],columns=['site_id']).sea_level_pressure.plot(ax=ax[1])
table = wtrain.groupby(['site_id','timestamp']

               ,as_index=False)['wind_direction'].mean()

table = table.loc[(table.site_id==0)|(table.site_id==8)|(table.site_id==7)|(table.site_id==11)]

pd.pivot_table(table,index=['timestamp'],columns=['site_id']).corr(method='pearson').style.background_gradient(cmap=cm)
fig,ax =plt.subplots(1,2)

fig.set_size_inches(15, 3)

table1 = wtrain.loc[(wtrain.site_id==0)|(wtrain.site_id==8)].set_index('site_id')

pd.pivot_table(table1,index=['timestamp'],columns=['site_id']).wind_direction.plot(ax=ax[0])

table2 = wtrain.loc[(wtrain.site_id==7)|(wtrain.site_id==11)].set_index('site_id')

pd.pivot_table(table2,index=['timestamp'],columns=['site_id']).wind_direction.plot(ax=ax[1])
table = wtrain.groupby(['site_id','timestamp']

               ,as_index=False)['wind_speed'].mean()

table = table.loc[(table.site_id==0)|(table.site_id==8)|(table.site_id==7)|(table.site_id==11)]

pd.pivot_table(table,index=['timestamp'],columns=['site_id']).corr(method='pearson').style.background_gradient(cmap=cm)
fig,ax =plt.subplots(1,2)

fig.set_size_inches(15, 3)

table1 = wtrain.loc[(wtrain.site_id==0)|(wtrain.site_id==8)].set_index('site_id')

pd.pivot_table(table1,index=['timestamp'],columns=['site_id']).wind_speed.plot(ax=ax[0])

table2 = wtrain.loc[(wtrain.site_id==7)|(wtrain.site_id==11)].set_index('site_id')

pd.pivot_table(table2,index=['timestamp'],columns=['site_id']).wind_speed.plot(ax=ax[1])
print('Missing Values(Site 0)')

print(wtrain.loc[wtrain.site_id==0].isnull().sum())

print('########################')

print('Missing Values(Site 8)')

print(wtrain.loc[wtrain.site_id==8].isnull().sum())
print('Missing Values(Site 7)')

print(wtrain.loc[wtrain.site_id==7].isnull().sum())

print('########################')

print('Missing Values(Site 11)')

print(wtrain.loc[wtrain.site_id==11].isnull().sum())