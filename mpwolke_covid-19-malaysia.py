#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRh2h1wC9mclhgCUeUpS1iyxXBF-JmYwstGQ1EMdFkKAUWfMI7k&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/covid19-malaysia-by-region/Cases_ByState.csv")

df.head()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.countplot(x="JH",data=df,palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
df['KD'].hist()

plt.show()
fig = px.bar(df[['Date', 'KD']].sort_values('KD', ascending=False), 

             y="KD", x="Date", color='Date', 

             log_y=True, template='ggplot2', title='Covid-19 Malaysia')

fig.show()
fig = px.bar(df, 

             x='KD', y='Date', color_discrete_sequence=['#D63230'],

             title='Covid-19 Malaysia', text='Source')

fig.show()
# filling missing values with NA

df[['KD', 'Source']] = df[['KD', 'Source']].fillna('NA')
fig = px.bar(df,

             y='Date',

             x='KD',

             orientation='h',

             color='Source',

             title='Covid-19 Malaysia',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.scatter(df, x="Date", y="KD", color="Source", marginal_y="rug", marginal_x="histogram")

fig
fig = px.line(df, x="Date", y="KD", color_discrete_sequence=['green'], 

              title="Covid-19 Malaysia")

fig.show()
fig = px.bar(df[['Date','KD']].sort_values('KD', ascending=False), 

                        y = "KD", x= "Date", color='KD', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Covid-19 Malaysia")



fig.show()
fig = px.pie(df,

             values="KD",

             names="Date",

             template="presentation",

             labels = {'Date' : 'JH', 'SE' : 'KL'},

             color_discrete_sequence=['#4169E1', '#DC143C', '#006400'],

             width=800,

             height=450,

             hole=0.6)

fig.update_traces(rotation=180, pull=0.05, textinfo="percent+label")

py.offline.iplot(fig)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRJM9F6lpTjBq4RAU4znWXIF1nJyKl8OAGNtDzElY0KESY8kWP1&usqp=CAU',width=400,height=400)