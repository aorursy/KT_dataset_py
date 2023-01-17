#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ5v1AkT5BgOsGR2OHLDZG8gTvrVXLoi-uXM9FpSGlFXnTr3RgG&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import json



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/trec-covid-information-retrieval/CORD-19/CORD-19/metadata.csv')
title = df.copy()

title = title.dropna(subset=['title'])
title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)

title['title'] = title['title'].str.lower()
title['keyword_cytokine'] = title['title'].str.find('cytokine')
title.head()
included_cytokine = title.loc[title['keyword_cytokine'] != -1]

included_cytokine
import json

file_path = '/kaggle/input/trec-covid-information-retrieval/CORD-19/CORD-19/document_parses/pdf_json/4fa871503ddbbaaead7a34fce89298a36648f662.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
storm = pd.read_csv('../input/cusersmarildownloadscytokinecsv/cytokine.csv', sep=';')

storm
storm.corr()

plt.figure(figsize=(10,4))

sns.heatmap(storm.corr(),annot=True,cmap='Greens')

plt.show()
fig = px.parallel_categories(storm, color="1447617_at", color_continuous_scale=px.colors.sequential.Viridis)

fig.show()
storm['1447617_at'].hist()

plt.show()
fig = px.scatter_matrix(storm)

fig.show()
# 3D Scatter Plot

fig = px.scatter_3d(storm, x='1416873_a_at', y='1447617_at', z='Samples')

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=storm['Samples'][0:10],

    y=storm['1416873_a_at'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Cytokine distribution',

    xaxis_title="Samples",

    yaxis_title="1416873_a_at",

)

fig.show()
cnt_srs = df['source_x'].value_counts().head()

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

    title='Covid-19 & Cytokine Storm',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="source_x")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQrkNf4sgVaJFamrq6wktz4ewSfiNBlXu3fYm2zWSMGzJHu24Bt&usqp=CAU',width=400,height=400)