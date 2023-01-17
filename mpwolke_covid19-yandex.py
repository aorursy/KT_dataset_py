#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSYzKySnH1xjqwGsVf5JDvRDgK5Y-nnibFVSgwJw6kZYrT5Ify3&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/covid19yandexdataset/ID.csv')

df.head()
px.histogram(df, x='Index', color='Country')
px.histogram(df, x='Date', color='Index')
fig = px.histogram(df, x='Index', color='City')

fig.update_layout(showlegend=False)

fig.show()
fig = px.bar(df,

             y='Date',

             x='Index',

             orientation='h',

             color='Country',

             title='Covid-19 Yandex',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.parallel_categories(df, color="Index", color_continuous_scale=px.colors.sequential.Viridis)

fig.show()
fig = px.bar(df, x= "Date", y= "Index", color_discrete_sequence=['crimson'],)

fig.show()
# Scatter Matrix Plot

fig = px.scatter_matrix(df)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)