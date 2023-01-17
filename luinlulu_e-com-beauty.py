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
df = pd.concat([pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Oct.csv"),
                pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Nov.csv"),
                pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Dec.csv"),
                pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2020-Jan.csv"),
                pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2020-Feb.csv")
               ])
# check data types
df.info()
# get smaller dataset randomly
#df = df.sample(n = 5000, random_state=42)
#df.to_csv("df_sample.csv")
df = pd.read_csv("df_sample.csv")
df.head()
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
# convert feature types
df.astype({'product_id': 'object', 'category_id': 'object', 'user_id': 'object'}).dtypes
df = df.reset_index()
# change event_time into datetime format
df['event_time'] = pd.to_datetime(df['event_time'])
df.info()
# check for null values
df.isnull().sum()
# drop category_code column
df = df.drop(['category_code'], axis=1)
# add month, year, hour to df
df['month'] = pd.DatetimeIndex(df['event_time']).month
df['year'] = pd.DatetimeIndex(df['event_time']).year
df['hour'] = pd.DatetimeIndex(df['event_time']).hour
# add period of the day
prods = pd.DataFrame({'hour':range(1, 25)})

b = [0,4,8,12,16,20,24]
l = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
prods['session'] = pd.cut(prods['hour'], bins=b, labels=l, include_lowest=True)

df['session'] = prods['session']
# add day of week
df['day'] =df['event_time'].dt.dayofweek
df['day'] = df['day'].replace({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
df.groupby(['event_type']).count()
df.groupby(['user_session']).count().sort_values(by='event_time', ascending=False)

id = df.groupby(['user_session']).count().sort_values(by='event_time', ascending=False).reset_index()['user_session'][2]
df[df['user_session'] == id]
df.groupby(['user_id']).count().sort_values(by='event_type', ascending=False)
us_id = df.groupby(['user_id']).count().sort_values(by='event_type', ascending=False).reset_index()['user_id'][1]

df[df['user_id'] == us_id]

df.groupby(['event_type']).count()

df[df['event_type'] == 'purchase']
