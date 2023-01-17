# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/covid19-symptoms-dataset/covid19-symptoms-dataset.xlsx')

df.head()
df = df.rename(columns={'Infected with Covid19':'infected', 'High Fever': 'fever', 'Dry Cough': 'cough'})
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['cough', 'fever'], as_index=False).infected.sum()



fig = px.bar(plot_data, x='cough', y='infected', color='fever')

fig.show()
import plotly.express as px



plot_data = df.groupby(['cough'], as_index=False).infected.sum()



fig = px.line(plot_data, x='cough', y='infected')

fig.show()
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['cough', 'fever'], as_index=False).infected.sum()



fig = px.line(plot_data, x='cough', y='infected', color='fever')

fig.show()
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['cough', 'Difficulty in breathing'], as_index=False).infected.sum()



fig = px.line_polar(plot_data, theta='cough', r='infected', color='Difficulty in breathing')

fig.show()
cnt_srs = df['cough'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='Dry Cough',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="cough")
cnt_srs = df['fever'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='High Fever',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="fever")