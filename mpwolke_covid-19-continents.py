#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS4Q0V6IdFjdgRjifawS3mfNE1HFfaaCG3GdGjVqOI_kBqnemkk',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

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
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/covid19pluspopulations/wiki_pop.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'wiki_pop.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head().style.background_gradient(cmap='gist_rainbow')
df1 = pd.read_csv('../input/covid19pluspopulations/country-and-continent-codes-list.csv', encoding='ISO-8859-2')

df1.head().style.background_gradient(cmap='gist_rainbow')
plot_data = df.groupby(['% of world population'], as_index=False).Country_cln.sum()



fig = px.line(plot_data, x='% of world population', y='Country_cln')

fig.show()
cnt_srs = df['% of world population'].value_counts().head()

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

    title='Percent of World Population',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="% of world population")
from wordcloud import WordCloud

def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df["Country_cln"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Countries')
cnt_srs = df1['Continent_Name'].value_counts().head()

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

    title='Continents',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Continent_Name")
from wordcloud import WordCloud

def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df1["Continent_Name"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='blue', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Continents')