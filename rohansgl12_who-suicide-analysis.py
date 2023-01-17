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


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

import plotly.figure_factory as ff

import plotly.express as px
data  = pd.read_csv('/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv')



data.head()
data.info()
# renaming columns



data.rename({'sex':'gender' , 'suicides_no':'suicides'} , inplace = True , axis = 1)
# lets check missing values



check = data.isnull().sum()/data.shape[0]

print(check*100)
# let's check what all category we have for age group



data['age'].unique()
# lets check the countries with more than 5000 suicides in age groupb 15-24 years 



check = data[(data['age'] == '15-24 years') & (data['suicides'] >=5000)][['country',  'suicides' , 'gender' , 'year']].sort_values('suicides' , ascending = False)

check.style.background_gradient(cmap = 'PuBu')

# Top 10 Countries wrt Suicides



data[['country' , 'suicides']].groupby(['country']).agg('sum').sort_values('suicides' , ascending = False).head(5).style.background_gradient('PuBu')
# lets check in which year we highest no. of suicides



data[['suicides',

      'year']].groupby(['year']).agg('sum').sort_values(by = 'suicides',

                                                        ascending = False).head(10).style.background_gradient(cmap = 'PuBu')

# filling missing values



data['suicides'].fillna(0, inplace = True)

# data['population'].mean()

data['population'].fillna(data['population'].mean(), inplace = True)



# checking if there is any null value left

data.isnull().sum().sum()



# converting these attributes into integer format

data['suicides'] = data['suicides'].astype(int)

data['population'] = data['population'].astype(int)
data.isnull().sum()
df = px.data.gapminder()

df.head()
x = pd.merge(data, df, on = 'country')

x = x[['country', 'suicides','gender','age','year_x','population','iso_alpha']]

x.head()
fig = px.choropleth(

    x,

    locations='iso_alpha',

    color = 'suicides',

    hover_name='country',

    animation_frame= 'year_x',

    height=500,

    width=500



         



)





fig.show()
check = data[['country' , 'suicides']].groupby(['country']).agg('sum').reset_index().sort_values('suicides' , ascending = False).head(10)

fig = px.bar(

    check,

    x = 'country',

    y = 'suicides',

    color = 'suicides',

    height = 500,

    width  = 500,

    title = 'Top 10 countries by number of Suicides'





)



fig.show()



check = x[['year_x' , 'suicides']].groupby(['year_x']).agg('sum').reset_index().sort_values('year_x' , ascending = True)

fig = px.line(

    check,

    x = 'year_x',

    y = 'suicides',

    height = 500,

    width = 900,

    title = 'Suicide Trend'



)



fig.show()