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