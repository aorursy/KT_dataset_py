# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/weather.csv', index_col=0)
df['date'] = pd.to_datetime(df['date'], format="%m/%d/%Y")
df['time'] = pd.to_datetime(df.time,format='%H:%M').dt.time

# replace '-----' to nan
df['T'].replace('-----', np.nan, inplace=True)
df['Tmax'].replace('-----', np.nan, inplace=True)
df['Tmin'].replace('-----', np.nan, inplace=True)

df['T'] = df['T'].astype(float)
df['Tmax'] = df['Tmax'].astype(float)
df['Tmin'] = df['Tmin'].astype(float)

df.tail(100)
df.drop_duplicates(subset =['date','time'], 
                     keep = False, inplace = True) 
df.info()
# replace incorrect values
df['T'].replace(50.2, np.nan, inplace=True)
df['T'].replace(-90.4, np.nan, inplace=True)
df['T'].replace(-88.0, np.nan, inplace=True)
df['T'].replace(-48.8, np.nan, inplace=True)

df.drop_duplicates(subset =['date','time'], 
                     keep = False, inplace = True) 
df.info()
# replace incorrect values
df['T'].replace(50.2, np.nan, inplace=True)
df['T'].replace(-90.4, np.nan, inplace=True)
df['T'].replace(-88.0, np.nan, inplace=True)
df['T'].replace(-48.8, np.nan, inplace=True)
# analysis
df.loc[df['T'] < -39]
df.info()
# only with Tmin
df = df[['date', 'Tmin']]
#df['Tmin'] = df['Tmin'].notna
df = df.dropna()
df.head(30)
filtered = df[df.Tmin <= -30]
import datetime
filtered['inc4days'] = ((filtered.date - filtered.shift(1).date == datetime.timedelta(days=1)) & (filtered.date - filtered.shift(2).date == datetime.timedelta(days=2)) & (filtered.date - filtered.shift(3).date == datetime.timedelta(days=3)))

filtered = filtered[filtered.inc4days == True]
filtered['shift'] = filtered['date'] - filtered['date'].shift(1)
filtered1 = filtered[filtered['shift'] != datetime.timedelta(days=1)]
filtered1