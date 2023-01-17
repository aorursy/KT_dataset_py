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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math



train = pd.read_csv('../input/train.csv', parse_dates = ['Dates'])



train["Hour"] = train.Dates.dt.hour



categories = train.PdDistrict.unique()



all_cats = [train[train.PdDistrict== cat].Hour.value_counts().reindex(range(24)) for cat in categories]



f, axarr = plt.subplots(4,3, figsize=(6,40))

f.subplots_adjust(right=2.2, hspace=0.5)





for x in range(10):

    axarr[math.floor(x/3), x % 3].plot(all_cats[x])

    axarr[math.floor(x/3), x % 3].set_title(categories[x])



plt.savefig('crimes_by_hour.png', orientation='landscape')
import pandas as pd

import zipfile

import matplotlib.pyplot as pl



z = zipfile.ZipFile('../input/train.csv.zip')

print(z.namelist())



train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])



train['Year'] = train['Dates'].map(lambda x: x.year)

train['Week'] = train['Dates'].map(lambda x: x.week)

train['Hour'] = train['Dates'].map(lambda x: x.hour)



print(train.head())
train['event']=1

weekly_events = train[['Hour','PdDistrict','event']].groupby(['PdDistrict','Hour']).count().reset_index()

weekly_events_years = weekly_events.pivot(index='Hour', columns='PdDistrict', values='event').fillna(method='ffill')
ax = weekly_events_years.interpolate().plot(title='number of cases every 2 weeks', figsize=(10,6))

pl.savefig('events_every_two_weeks.png')
train = pd.read_csv("../input/train.csv", parse_dates = ["PdDistrict"])

test = pd.read_csv("../input/test.csv", parse_dates = ["PdDistrict"])
