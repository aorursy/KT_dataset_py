# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import plotly.graph_objects as go

import plotly.offline as py



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadslocationcsv/location.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'location.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
ax = df.groupby('New_Camp_Name')['Total_Pop'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(12,8),

                                                                                   title='Mean Total Population by New Camp')

plt.xlabel('Count')

plt.ylabel('New_Camp_Name')

plt.show()
ax = df.groupby('New_Camp_SSID')['Total_HH'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(12,8), color='r',

                                                                                   title='Mean Total Households by New Camp SSID')

plt.xlabel('Count')

plt.ylabel('New_Camp_SSID')

plt.show()
ax = df.groupby('New_Camp_Name')['Total_HH', 'Total_Pop'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Total Population & Households in New Camp')

plt.xlabel('New Camp Name')

plt.ylabel('Log')

plt.show()
ax = df.groupby('New_Camp_SSID')['Total_Pop', 'Total_HH'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='Total Population & Households in New Camp SSID', logx=True, linewidth=3)

plt.xlabel('Log')

plt.ylabel('Total Households')

plt.show()
fig = px.bar(df, x= "New_Camp_Name", y= "Total_Pop", color_discrete_sequence=['crimson'], title='Total Population in New Camp')

fig.show()