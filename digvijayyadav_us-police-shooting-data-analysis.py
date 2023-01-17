# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_validate

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier 

from sklearn import metrics



import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport

import seaborn as sns

import plotly.graph_objects as go

import plotly.figure_factory as ff

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/us-police-shootings/shootings.csv')

print('Rows {} columns {} in data'.format(df.shape[0], df.shape[1]))
df.head()
df.isnull().sum()
df.isna().sum()
df.info()
df.describe()
ProfileReport(df)
plt.figure(figsize=(10,5))

sns.heatmap(df.corr(), annot=True, cmap=plt.cm.plasma)
labels = ['Males', 'Females']

values = df['gender'].value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
sns.distplot(df['age'],kde=True,color='b')
males = df[df['gender']=='M']['age'].values

females = df[df['gender']=='F']['age'].values



sns.distplot(males, hist=False, label='Males')

sns.distplot(females, hist=False, label='Females')

plt.xlabel('Age')

plt.ylabel('Distribution Frequency')
data=df['race'].value_counts().reset_index().rename(columns={'index':'race','race':'count'})



fig = go.Figure(go.Bar(

    x=data['race'],y=data['count'],

    marker={'color': data['count'], 

    'colorscale': 'Viridis'},  

))

fig.update_layout(title_text='frequency of different race',xaxis_title="race",yaxis_title="count",height=500,width=500)

fig.show()
labels = df['manner_of_death'].value_counts().index

values = df['manner_of_death'].value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
labels = df['armed'].value_counts().index

values = df['armed'].value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, textinfo='value', values=values)])

fig.show()
black = df[df['race']=='black']['state'].value_counts().to_frame().reset_index().rename(columns={'index':'state','state':'count'})
fig = go.Figure(go.Choropleth(

    locations=black['state'],

    z=black['count'].astype(float),

    locationmode='USA-states',

    colorscale='Reds',

    autocolorscale=False,

    text=black['state'], # hover text

    marker_line_color='white', # line markers between states

    colorbar_title="Millions USD",showscale = False,

))

fig.update_layout(

    title_text='US Police shooting cases of black people',

    title_x=0.5,

    geo = dict(

        scope='usa',

        projection=go.layout.geo.Projection(type = 'albers usa'),

        showlakes=True, # lakes

        lakecolor='rgb(255, 255, 255)'))

fig.update_layout(

    template="plotly_dark")

fig.show()

df['year']=pd.to_datetime(df['date']).dt.year
year_shoot=df['year'].value_counts().to_frame().reset_index().rename(columns={'index':'year','year':'count'}).sort_values(by="year")

fig = go.Figure(data=go.Scatter(

    x= year_shoot['year'],

    y= year_shoot['count'],

    mode='lines+markers',

    marker_color="red"

))

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='Deaths - All Years',xaxis_title='Years',

                 yaxis_title='Total number of kills', title_x=0.5)



fig.show()