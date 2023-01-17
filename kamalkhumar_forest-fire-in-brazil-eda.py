# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = 'latin_1', index_col = 'date')

df.head()
print(df.dtypes)

print(df.shape)
df.info()
df.isna().sum()
len(df[df.duplicated()] )
df = df.drop_duplicates()
df.state.value_counts()
df.month.value_counts()



df['month'] = df['month'].map({'Agosto':'August', 'Outubro':'October', 'Novembro':'November', 'Setembro':'September','Junho':'June', 'Julho':'July', 'Janeiro':'January', 'Fevereiro':'February','Abril':'April','Março':'March','Maio':'May','Dezembro':'December'})
df['number'] = df['number'].astype(int)
fig = go.Figure(data=go.Scatter(x = df.groupby(['year'])['number'].sum().index, y = df.groupby(['year'])['number'].sum().values))

fig.update_layout(title='Forest fire in Brazil over the years',

                   xaxis_title='Year',

                   yaxis_title='Number of forest fires')

fig.show()
fig = go.Figure(data=go.Scatter(x = df.groupby(['month'])['number'].sum().index, y = df.groupby(['month'])['number'].sum().values))

fig.update_layout(title='Forest fire distribution with respect to months (1998 - 2017)',

                   xaxis_title='Month',

                   yaxis_title='Number of forest fires')

fig.show()
sns.set(rc={'figure.figsize':(15,10)})

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(df['state'].value_counts().values, labels = df['state'].value_counts().index)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
import plotly.express as px

fig = px.pie(values=df['state'].value_counts().values, names=df['state'].value_counts().index,title='Cities involving forest fires')

fig.show()
import plotly.express as px

fig = px.sunburst(df, path=['year', 'month'], values='number')

fig.show()
fig = px.scatter(df, x="year", y="month",size='number')

fig.show()