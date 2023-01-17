#'''Importing Data Manipulation Modules'''
import numpy as np                 # Linear Algebra
import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)

#'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
import geopandas as gpd            # Python Geospatial Data Library
plt.style.use('fivethirtyeight')
%matplotlib inline

#'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.offline as py
init_notebook_mode(connected=True)


#'''NLP - WordCloud'''
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os
%pylab inline

from PIL import Image
df = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
df.head()
print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())
df.isnull().sum()
labels = df['type'].value_counts().index
values = df['type'].value_counts()
colors = ['pink', 'lightblue']


trace = go.Pie(labels=labels, values=values,title = 'Distribution of types',marker = dict(colors = colors), pull=[0.05, 0],textinfo='percent+label' )

iplot([trace])
top_10 = df['country'].value_counts()[:10]
labels = top_10.index
values = top_10

trace = go.Pie(labels=labels, values=values, title = 'Distribution of Countries',textinfo='percent+label')

iplot([trace])
top_15 = df['listed_in'].value_counts()[:15]
labels = top_15.index
values = top_15

trace = go.Pie(labels=labels, values=values, title = 'Distribution of Countries',textinfo='percent')

iplot([trace])
xmovie = df[df['type'] == 'Movie']
xtv = df[df['type'] == 'TV Show']

trace1 = go.Histogram(
    x=xmovie['release_year'],
    opacity=0.75,
    name = "Movie",
    marker=dict(color='rgb(165,0,38)'))
trace2 = go.Histogram(
    x=xtv['release_year'],
    opacity=0.75,
    name = "TV Show",
    marker=dict(color='rgb(215,48,39)'))

data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Number of shows released per year',
                   xaxis=dict(title='Years'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.rcParams['figure.figsize'] = (18,8)
sns.countplot(x = 'rating', data = df, order = df['rating'].value_counts()[:15].index, hue = 'type',palette="Set1")

plt.rcParams['figure.figsize'] = (18,8)
sns.countplot(y = 'release_year', data = df, order = df['release_year'].value_counts()[:15].index, hue = 'type', palette="Set2")
plt.title('Year wise analysis', fontsize = '20')
plt.rcParams['figure.figsize'] = (18,8)
sns.countplot(y = 'country', data = df, order = df['country'].value_counts()[:10].index, hue = 'type', palette="Set3")
plt.title('Top 10 Countries wise analysis', fontsize = '20')
plt.rcParams['figure.figsize'] = (18,8)
sns.countplot(y = 'duration', data = df, order = df['duration'].value_counts()[:10].index)
plt.title('Duration wise analysis',fontsize = 20)
#netflix_fr=df[df['country']=='China']
cleaned=df.dropna()
import plotly.express as px
fig = px.treemap(cleaned, path=['country','director'],
                  color='director', hover_data=['director','title'],color_continuous_scale='Purples')
fig.show()
wave_mask= np.array(Image.open("../input/laptop-clapperboard/laptop-pc-portable-in-black-and-white-vector-24598028.jpg"))
stopwords = set(STOPWORDS)
stopwords.update(["II", "III"])
plt.subplots(figsize=(15,15))

wordcloud = WordCloud(mask=wave_mask,background_color="lavenderblush",colormap="hsv" ,contour_width=2, contour_color="black",
                      width=950,stopwords=stopwords,
                          height=950

                         ).generate(" ".join(df.description))

plt.imshow(wordcloud ,interpolation='bilinear')
plt.axis('off')
plt.savefig('graph.png')
plt.title('Wordcloud for Description')
plt.show()
wave_mask= np.array(Image.open("../input/laptop-clapperboard/clapper-board-refixed.jpg"))
stopwords = set(STOPWORDS)
stopwords.update(["II", "III"])
plt.subplots(figsize=(15,15))
wordcloud = WordCloud(mask=wave_mask,background_color="lavenderblush",colormap="hsv" ,contour_width=2, contour_color="black",
                      width=950,stopwords=stopwords,
                          height=950
                         ).generate(" ".join(df.title))

plt.imshow(wordcloud ,interpolation='bilinear')
plt.axis('off')
plt.savefig('graph.png')
plt.title('Wordcloud for Title')
plt.show()
trace1 = go.Scatter(
                    x = df.country,
                    y = df['listed_in'][:100],
                    mode = "markers",
                    name = "North America",
                    marker = dict(color = 'rgba(28, 149, 249, 0.8)',size=8),
                    text= df.title)


data = [trace1]
layout = dict(title = 'Countries - List - Title',
              xaxis= dict(title= 'Countries',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white"),
              yaxis= dict(title= 'List',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white",),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)' )
fig = dict(data = data, layout = layout)
iplot(fig)
trace = go.Scatter3d(
    x = df.director,
    y = df.title,
    z = df.release_year,
    name = 'Marvel',
    mode = 'markers',
    marker = dict(
         size = 10,
         color = df.release_year,
         colorscale = "Rainbow",
         line=dict(color='rgb(140, 140, 170)')

    )
)

df = [trace]

layout = go.Layout(
    title = 'Cholestrol vs Heart Rate vs Age',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    ),
    scene = dict(
            xaxis = dict(title  = 'Director'),
            yaxis = dict(title  = 'Title'),
            zaxis = dict(title  = 'Release_Year')
        )
    
)
fig = go.Figure(data = df, layout=layout)
py.iplot(fig)