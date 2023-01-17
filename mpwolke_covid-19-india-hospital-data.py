#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

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
df = pd.read_csv("../input/covid19-india-hospital-data/Covid19_India_Data.csv")

df.head()
fig_px = px.scatter_mapbox(df, lat="Latitude", lon="Longitude",

                           hover_name="State/UT",

                           zoom=11, height=300)

fig_px.update_layout(mapbox_style="open-street-map",

                     margin={"r":0,"t":0,"l":0,"b":0})



fig_px.show()
fig_px.update_traces(marker={"size": [10 for x in df]})
fig = px.bar(df,

             y='District Hospitals',

             x='Total Public Health Facility',

             orientation='h',

             color='Population',

             title='India Hospital Data',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



plot_correlation_map(df) 
df.hist()
sns.countplot(df["Government Laboratory"])

plt.xticks(rotation=90)

plt.show()
fig = px.bar(df[['Unnamed: 0', 'Public Beds']].sort_values('Public Beds', ascending=False), 

             y="Public Beds", x="Unnamed: 0", color='Unnamed: 0', 

             log_y=True, template='ggplot2', title='India Hospital Data')

fig.show()
fig = px.line(df, x="Urban Hospitals", y="Public Beds", color_discrete_sequence=['green'], 

              title="India Hospital Data")

fig.show()
fig = px.bar(df, x= "State/UT", y= "Public Beds", color_discrete_sequence=['crimson'],)

fig.show()
fig = px.bar(df, x= "Sl.No.", y= "District Hospitals", color_discrete_sequence=['sienna'],)

fig.show()
fig = px.bar(df, x= "State/UT", y= "Primary Health Center")

fig.show()
cnt_srs = df['Urban Hosp Beds'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'viridis',

        reversescale = True

    ),

)



layout = dict(

    title='India Hospital Data',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Urban Hosp Beds")
fig = px.bar(df,

             y='Unnamed: 0',

             x='State/UT',

             orientation='h',

             color='Public Beds',

             title='India Hospital Data',

             opacity=0.8,

             color_discrete_sequence=px.colors.sequential.Pinkyl,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.pie(df, values=df['Unnamed: 0'], names=df['State/UT'],

             title='India Hospital Data',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        colormap='Set3',

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(df['State/UT'])
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRY7vSIpFodVLaiCMYl9AZPfPBA5NyMpzGuGWiVzMuFpfsKsfDK&usqp=CAU',width=400,height=400)