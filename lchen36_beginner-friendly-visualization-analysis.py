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
df = pd.read_csv('../input/us-police-shootings/shootings.csv')
df.head()
print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())
df['date'] = pd.to_datetime(df['date'])
def get_year(x):
    return x.year
df['year'] = df['date'].apply(get_year)
trace = go.Histogram(x=df.year,marker=dict(color='rgb(223,145,163)',line=dict(color='black', width=2)),opacity=0.75)
layout = go.Layout(
    title='Distribution of Years',
    xaxis=dict(
        title='Years'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor="rgb(243, 243, 243)")
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

xattack = df[df['threat_level'] == 'attack']
xother = df[df['threat_level'] == 'other']
xundetermined = df[df['threat_level'] == 'undetermined']
def get_month(x):
    return x.month
df['month'] = df['date'].apply(get_month)
trace1 = go.Histogram(
    x=xattack.month,
    opacity=0.75,
    name = "attack",
    marker=dict(color='rgb(153,201,69)'))
trace2 = go.Histogram(
    x=xother.month,
    opacity=0.75,
    name = "other",
    marker=dict(color='rgb(51,133,255)'))
trace3 = go.Histogram(
    x=xundetermined.month,
    opacity=0.75,
    name = "undetermined",
    marker=dict(color='rgb(244,109,67)'))


data = [trace1, trace2, trace3]
layout = go.Layout(barmode='stack',
                   title='Distribution of Month',
                   xaxis=dict(title='Month'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

def count_rows(rows):
    return len(rows)
plt.rcParams['figure.figsize'] = (12,6)
plt.title('Month vs Year', fontsize = '20')
a = df.groupby('year month'.split()).apply(count_rows).unstack()
sns.heatmap(a)

labels = sorted(df.manner_of_death.unique())
values = df.manner_of_death.value_counts().sort_index()
colors = ['crimson','aqua']
trace = go.Pie(labels=labels, values=values, pull=[0.05, 0],textinfo='percent+label', marker = dict(colors = colors))
iplot([trace])

plt.rcParams['figure.figsize'] = (18,8)
plt.style.use('ggplot')
plt.title('Distribution of Age', fontsize = '20', color = 'blue')
plt.axvline(df['age'].mean(),linestyle='dashed',color='red')
sns.distplot(df['age'])
labels = sorted(df.gender.unique())
values = df.gender.value_counts().sort_index()
colors = ['aqua','black']
trace = go.Pie(labels=labels, values=values, pull=[0.05, 0],textinfo='percent+label', marker = dict(colors = colors))
iplot([trace])
plt.rcParams['figure.figsize'] = (18,8)
plt.style.use('ggplot')
plt.title('Race vs Age', fontsize = 20, color = 'Blue')
sns.boxenplot(x = 'race', y = 'age', data = df)
xattack = df[df['threat_level'] == 'attack']
xother = df[df['threat_level'] == 'other']
xundetermined = df[df['threat_level'] == 'undetermined']

trace1 = go.Histogram(
    x=xattack.race,
    opacity=0.75,
    name = "attack",
    marker=dict(color='rgb(102,194,165)'))
trace2 = go.Histogram(
    x=xother.race,
    opacity=0.75,
    name = "other",
    marker=dict(color='rgb(30,110,161)'))
trace3 = go.Histogram(
    x=xundetermined.race,
    opacity=0.75,
    name = "undetermined",
    marker=dict(color='rgb(215,48,39)'))

data = [trace1, trace2, trace3]
layout = go.Layout(barmode='stack',
                   title='Race vs Threat Level',
                   xaxis=dict(title='Race'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.rcParams['figure.figsize'] = (18,8)
plt.style.use('dark_background')
plt.title('Top 10 Crime States', fontweight = 40, fontsize = 30, color = 'white')
sns.countplot(y = 'state', data = df, order = df['state'].value_counts()[:10].index)
labels = sorted(df.flee.unique())
values = df.flee.value_counts().sort_index()
colors = ['crimson','aqua','gold']
trace = go.Pie(labels=labels, values=values, pull=[0.05, 0],textinfo='percent+label', marker = dict(colors = colors))
iplot([trace])
wave_mask= np.array(Image.open("../input/usa-map/il_1140xN.1875110874_gr8n.jpg"))
stopwords = set(STOPWORDS)
stopwords.update(["II", "III"])
plt.subplots(figsize=(15,15))
wordcloud = WordCloud(mask=wave_mask,background_color="lavenderblush",colormap="hsv" ,contour_width=2, contour_color="black",
                      width=950,stopwords=stopwords,
                          height=950
                         ).generate(" ".join(df.armed))

plt.imshow(wordcloud ,interpolation='bilinear')
plt.axis('off')
plt.savefig('graph.png')
plt.title('Wordcloud for armed forces')
plt.show()
trace1 = go.Scatter(
                    x = df['city'][:100],
                    y = df['race'],
                    mode = "markers",
                    name = "123",
                    marker = dict(color = 'rgba(231,41,138,0.8)',size=8),
                    text= df.name)


data = [trace1]
layout = dict(title = 'City - Race',
              xaxis= dict(title= 'City',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white"),
              yaxis= dict(title= 'Arms Category',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white",),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)' )
fig = dict(data = data, layout = layout)
iplot(fig)