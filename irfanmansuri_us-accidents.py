! pip install dexplot
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
# Importing some of the Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot
import dexplot as dxp

df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_June20.csv')
df.head()
# Shape of the data

df.shape
# Checking the null value

df.isnull().sum()
# Getting the info about the data

df.info(verbose=True)
df['Start_Lat'].nunique()
# Counting the unique numbers of Source
df['Source'].unique()
# Calculating the Contribution of each unique sources in percentage

labels = df['Source'].value_counts()[:].index
values = df['Source'].value_counts()[:].values

colors = df['Source']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
# Counting the unique numbers of state

df['State'].unique()
labels = df['State'].value_counts()[:10].index
values = df['State'].value_counts()[:10].values

colors = df['State']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent", insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
# Counting the unique time zones

df['Timezone'].unique()
labels = df['Timezone'].value_counts()[:].index
values = df['Timezone'].value_counts()[:].values

colors = df['Timezone']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent", insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
df['Weather_Timestamp'].unique()
labels = df['Weather_Timestamp'].value_counts()[:10].index
values = df['Weather_Timestamp'].value_counts()[:10].values

colors = df['Weather_Timestamp']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent", insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
df['Precipitation(in)'].unique()
