# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
df.head()
df.columns
len(df)
df.nunique()
df.count()-max(df.count())
df.head()
len(df[df['manufacturer'].isnull() & df['condition'].isnull() & df['cylinders'].isnull() & df['drive'].isnull()])
df_model = df['manufacturer', 'model', 'cylinders']
df.columns

sns.lineplot(x="year", y="mean", data=df_year_odometer.reset_index())

sns.boxplot(x=df_year_odometer["odometer"])
df_year_odometer = df[['year', 'odometer']].dropna()
len(df_year_odometer)
df_year_odometer = df_year_odometer.groupby(df_year_odometer['year'])['odometer'].agg(['sum', 'mean', 'max'])
q_low = df_year_odometer["odometer"].quantile(0.01)
q_hi  = df_year_odometer["odometer"].quantile(0.99)

df_year_odometer = df_year_odometer[(df_year_odometer["odometer"] < q_hi) & (df_year_odometer["odometer"] > q_low)]
len(df_year_odometer)
df_year_odometer
df_year_odometer_price = df[['year', 'odometer', 'price']].dropna()
df_year_odometer_price = df_year_odometer_price[df_year_odometer_price.year>1980]
df_year_odometer_pricen = df_year_odometer_price[df_year_odometer_price.price!=99]
q_lowo = df_year_odometer_price["odometer"].quantile(0.01)
q_hio  = df_year_odometer_price["odometer"].quantile(0.99)
df_year_odometer_price = df_year_odometer_price[(df_year_odometer_price["odometer"] < q_hio) & (df_year_odometer_price["odometer"] > q_lowo)]
q_lowp = df_year_odometer_price["price"].quantile(0.01)
q_hip  = df_year_odometer_price["price"].quantile(0.99)
df_year_odometer_price = df_year_odometer_price[(df_year_odometer_price["price"] < q_hip) & (df_year_odometer_price["price"] > q_lowp)]
sample = df_year_odometer_price.sample(n = 1000)
sns.lmplot(x='odometer', y='price', data=sample)
sns.lmplot(x='year', y='price', data=sample)
sns.boxplot(x=df_year_odometer_price["price"])
len(df_year_odometer_price)
num_year = df.groupby('year').size().reset_index()
num_year = num_year[num_year.year > 1980]
sns.lineplot(x="year", y=0, data=num_year)
df_year_odometer_price[df_year_odometer_price.price!=99]
