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
import pandas as pd

df1 = pd.read_csv ("/kaggle/input/june15.csv")

df2 = pd.read_csv ("/kaggle/input/july15.csv")

df3 = pd.read_csv ("/kaggle/input/aug15.csv")

df4 = pd.read_csv ("/kaggle/input/sep15.csv")

df5 = pd.read_csv ("/kaggle/input/oct15.csv")

# df1=pd.read_csv("june15.csv")

# /kaggle/input/june15.csv

# df = pd.read_csv (r'Path where the CSV file is stored\File name.csv')

# print (df)
df1['timestamp'] = pd.to_datetime(df1['timestamp'])  # Makes sure your timestamp is in datetime format

df1=df1.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df1.reset_index(level=0, inplace=True)
import plotly.express as px



fig = px.line(df1, x='timestamp', y='mb')

fig.show()
df2['timestamp'] = pd.to_datetime(df2['timestamp'])  # Makes sure your timestamp is in datetime format

df2=df2.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df2.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df2, x='timestamp', y='mb')

fig.show()
df3['timestamp'] = pd.to_datetime(df3['timestamp'])  # Makes sure your timestamp is in datetime format

df3=df3.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df3.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df3, x='timestamp', y='mb')

fig.show()
df4['timestamp'] = pd.to_datetime(df4['timestamp'])  # Makes sure your timestamp is in datetime format

df4=df4.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df4.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df4, x='timestamp', y='mb')

fig.show()
df5['timestamp'] = pd.to_datetime(df5['timestamp'])  # Makes sure your timestamp is in datetime format

df5=df5.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df5.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df5, x='timestamp', y='mb')

fig.show()
df=[df1,df2,df3,df4,df5]

df=pd.concat(df,join='outer',axis=0,ignore_index=False)
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Makes sure your timestamp is in datetime format

df=df.groupby(pd.Grouper(key='timestamp', freq='60Min')).sum()
df.reset_index(level=0, inplace=True)

import plotly.express as px



fig = px.line(df, x='timestamp', y='mb')

fig.show()