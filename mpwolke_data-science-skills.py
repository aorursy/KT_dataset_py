#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQDDE_kehEDUr-RWfuJypPzkJ-Pu9dvgmYJ9RCsjbOiwJmAfNlC',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data-behind-data-scientists/ds_skills_dfe.csv")
df.head()
df.dtypes
sns.countplot(df["programming_language_required"])

plt.xticks(rotation=90)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQjT5rkFwPcqQu1ro4HiXh5HSNCny_zwYDXStOMJ_BTBmJh1DAZ',width=400,height=400)
cnt_srs = df['cloud_software_required'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Cloud Software Required',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="cloud_software_required")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRq-EtJsc9wp2Ig5JncsUUf21MN-3mvzmVba4shFgz4uWLLJfQV',width=400,height=400)
cnt_srs = df['database_software_required'].value_counts().head()

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

    title='Database Software Required',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="database_software_required")
cnt_srs = df['statistic_software_required'].value_counts().head()

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

    title='Statistic Software Required',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="statistic_software_required")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQiymc_NGjdjSUQCjx2G7hS1QAhzm4hmlOizUkdo3YEaRaKbiAv',width=400,height=400)
cnt_srs = df['programming_language_required'].value_counts().head()

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

    title='Programming Language Required',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="programming_language_required")
fig = px.pie( values=df.groupby(['programming_language_required']).size().values,names=df.groupby(['programming_language_required']).size().index)

fig.update_layout(

    title = "Programming Language Required",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSp4eFgqh9fan2oobiJZadwu-93qVauK7LtKyESlkGpil_i8x1n',width=400,height=400)