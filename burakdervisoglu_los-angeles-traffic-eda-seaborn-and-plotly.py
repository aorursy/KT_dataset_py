# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/traffic-collision-data-from-2010-to-present.csv')
pd.set_option('display.max_columns',None) # for show all columns

df.head()
df.info()
df.isnull().sum()
df.shape # how much have value and columns
df['Date Reported'] = pd.to_datetime(df['Date Reported']).dt.year # to take the year

df['Date Occurred'] = pd.to_datetime(df['Date Occurred']).dt.year

df.head()
plt.subplots(figsize = (15,6))

sns.countplot(df['Date Occurred'])

plt.show()
plt.subplots(figsize = (15,6))

sns.countplot(df['Area Name'].sort_values(ascending = False))

plt.xticks(rotation = 90)

plt.show()
df['Premise Description'].value_counts().head(25)
female = []

male = []

for i in df['Victim Sex']:

    if i == "M":

        male.append(i)

    elif i == "F":

        female.append(i)

gender = female + male

gender = pd.DataFrame(gender)

df['Gender'] = gender



g = sns.FacetGrid(df,row= 'Date Occurred',col = 'Gender',margin_titles = True)

g.map(plt.hist,'Victim Age')

plt.show()
df.groupby(['Victim Sex'])['Victim Age'].sum().sort_values(ascending = False)[:2].plot(kind = 'bar')

plt.show()
plt.subplots(figsize = (15,6))

sns.countplot(df['Victim Descent'])

plt.show()
plt.subplots(figsize = (9,15))

df.groupby(['Area Name'])['Census Tracts'].sum().sort_values().plot(kind = 'barh')

plt.show()
df['Census Tracts'] = df['Census Tracts'].astype(float)

trace = []

for name,group in df.groupby(['Area Name']):

    trace.append(go.Box(

            x = group['Census Tracts'].values,

            name = name))

layout = go.Layout(

            title = 'AAa',

            width = 800,

            height = 2000)

fig = go.Figure(data= trace,layout = layout)

py.iplot(fig)
df.groupby(['Area Name'])['Area ID'].unique()
df2019 = df[df['Date Occurred'] == 2019]

df2018 = df[df['Date Occurred'] == 2018]

df2017 = df[df['Date Occurred'] == 2017]



trace1 = go.Bar(

        x = df2019['Victim Sex'].values,

        y = df2019['Victim Sex'].index,

        name = '2019 Victim Sex',

        )

trace2 = go.Bar(

        x = df2018['Victim Sex'].values,

        y = df2018['Victim Sex'].index,

        name = '2018 Victim Sex',

        )

trace3 = go.Bar(

        x = df2017['Victim Sex'].values,

        y = df2017['Victim Sex'].index,

        name = '2017 Victim Sex',

        )

layout = go.Layout(barmode = 'group')



data = [trace1,trace2,trace3]

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)
trace1 = go.Box(

       

        y = df2019['Census Tracts'].head(100),

        name = '2019 Census Tracts',

        )

trace2 = go.Box(

        

        y = df2018['Census Tracts'].head(100),        

        name = '2018 Census Tracts')

trace3 = go.Box(

        y = df2017['Census Tracts'],

        name = '2017 Census Tracts')

data = [trace1,trace2,trace3]

py.iplot(data)
print("Below 18 years = ",len(df[df['Victim Age'] <= 18]),"Person")

print("Above 18 years = ",len(df[df['Victim Age'] >= 18]),"Person")
print("Max Age of Victim = ",df['Victim Age'].max(),"Years Old")

print("Min Age of Victim = ",df['Victim Age'].min(),"Years Old")