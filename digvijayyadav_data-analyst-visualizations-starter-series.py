!pip install dexplot

!pip install chart_studio

!pip install pandas-profiling
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import dexplot as dxp

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import seaborn as sns

from pandas_profiling import ProfileReport



import plotly.express as px



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')

df.shape
df.head(5)
df.describe()
df.kurtosis()
report = ProfileReport(df)

report
df.isnull().sum()
df.replace(-1, np.nan, inplace = True)

df.replace('-1', np.nan, inplace = True)

df.replace(-1.0, np.nan, inplace = True)
plt.figure(figsize=(10,5))

colormap = plt.cm.plasma

sns.heatmap(df.corr(), annot=True, cmap=colormap)
missing_df = df.isnull().sum()

missing_df
number_of_rec = df.shape[0]

print(number_of_rec)



prop_msval = pd.DataFrame(missing_df, columns = ['Missing after update'])

prop_msval = prop_msval/(number_of_rec/100)

display(prop_msval)
prop_msval.plot(kind='barh', figsize=(15,10))

plt.xlabel('Missing data frequency')
prop_msval2 = prop_msval[prop_msval['Missing after update']>10.0]

display(prop_msval2)
to_drop = prop_msval[prop_msval['Missing after update']>20.0]

to_drop = to_drop.transpose()

display(to_drop)
df = df.drop(to_drop,axis=1)

df = df.drop('Unnamed: 0',axis=1)

df.head(5)
df.info()
dxp.count('Rating',df)
labels = df['Company Name'].value_counts()[:15].index

values = df['Company Name'].value_counts()[:15].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = df['Salary Estimate'].value_counts()[:15].index

values = df['Salary Estimate'].value_counts()[:15].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = df['Industry'].value_counts()[:15].index

values = df['Industry'].value_counts()[:15].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = df['Location'].value_counts()[:15].index

values = df['Location'].value_counts()[:15].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = df['Sector'].value_counts()[:15].index

values = df['Sector'].value_counts()[:15].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = df['Job Title'].value_counts()[:15].index

values = df['Job Title'].value_counts()[:15].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = df['Revenue'].value_counts()[:20].index

values = df['Revenue'].value_counts()[:20].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
sns.countplot(y=df['Salary Estimate'], color='orange')

sns.countplot(y=df['Sector'], color='green')
dxp.count('Type of ownership', df, orientation='h')