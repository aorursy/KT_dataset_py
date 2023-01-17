#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRqXVcEfwVPbdW94hnHkUKAwqwWuaCyul0XgQ53SRzc-ik5KgOT&usqp=CAU',width=400,height=400)
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
df = pd.read_csv("../input/coronaviruscovid19-canada/cases.csv")

df.head().style.background_gradient(cmap='prism')
plot_data = df.groupby(['provincial_case_id'], as_index=False).age.sum()



fig = px.line(plot_data, x='provincial_case_id', y='age')

fig.show()
plot_data = df.groupby(['provincial_case_id'], as_index=False).province.sum()



fig = px.line(plot_data, x='provincial_case_id', y='province')

fig.show()
fig = px.bar(df, x= "case_id", y= "province")

fig.show()
fig = px.bar(df, x= "province", y= "case_id")

fig.show()
cnt_srs = df['age'].value_counts().head()

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

    title='COVID-19 by Age',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="age")
cnt_srs = df['province'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='COVID-19 by Province',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="province")
cnt_srs = df['sex'].value_counts().head()

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

    title='COVID-19  by Sex',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="sex")
# Grouping it by age and province

plot_data = df.groupby(['age', 'provincial_case_id'], as_index=False).province.sum()



fig = px.bar(plot_data, x='age', y='province', color='provincial_case_id')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSJNM8Ceu6E7fvqTN2Di15XNt_aZHQg_lD_VuFJdJFRxaIEORie&usqp=CAU',width=400,height=400)