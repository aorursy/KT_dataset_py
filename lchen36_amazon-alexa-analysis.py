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

from PIL import Image

import os
%pylab inline

df = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter = '\t', quoting = 3)
df.head()
df['date'] = pd.to_datetime(df['date'])
df.head()
def get_month(x):
    return x.month
def get_weekday(x):
    return x.weekday()
df['month'] = df['date'].apply(get_month)
df['weekday'] = df['date'].apply(get_weekday)
df.head()
labels = sorted(df.rating.unique())
values = df.rating.value_counts().sort_index()
colors = ['pink', 'lightblue', 'aqua', 'gold', 'crimson']

trace = go.Pie(labels=labels, values=values,title='Distribution of rating',marker = dict(colors = colors))


iplot([trace])

trace = go.Histogram(x=df.variation,marker=dict(color="crimson",line=dict(color='black', width=2)),opacity=0.75)
layout = go.Layout(
    title='Numbers of different variation',
    xaxis=dict(
        title='Variations'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor="rgb(243, 243, 243)")
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

wave_mask= np.array(Image.open('../input/batman/batman.png'))
stopwords = set(STOPWORDS)
stopwords.update(["II", "III"])
plt.subplots(figsize=(15,15))
wordcloud = WordCloud(mask=wave_mask,background_color="lavenderblush",colormap="hsv" ,contour_width=2, contour_color="black",
                      width=950,stopwords=stopwords,
                          height=950
                         ).generate(" ".join(df.verified_reviews))

plt.imshow(wordcloud ,interpolation='bilinear')
plt.axis('off')
plt.savefig('graph.png')

plt.show()
xfeedback1 = df[df['feedback'] == 1]
xfeedback2 = df[df['feedback'] == 0]
trace1 = go.Histogram(
    x=xfeedback1['variation'],
    opacity=0.75,
    name = "feedback",
    marker=dict(color='rgb(165,0,38)'))
trace2 = go.Histogram(
    x=xfeedback2['variation'],
    opacity=0.75,
    name = "no_feedback",
    marker=dict(color='rgb(215,48,39)'))

data = [trace1, trace2]

layout = go.Layout(barmode='stack',
                   title='Counts of different Variation',
                   xaxis=dict(title='Variations'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.rcParams['figure.figsize'] = (18,8)
hist(df.weekday, bins = 7, range = (-.5,6.5),rwidth=.8)
plt.xticks(range(7),'Mon Tue Wed Thu Fri Sat Sun'.split());

plt.ylabel('Counts')
plt.title('Date of review')
plt.show()
sns.countplot(x = 'month', data = df)
sns.violinplot(df.variation,df.rating).plot(kind="bar",figsize=(20,6),fontsize = 15)
plt.title('Variations sort by ratings', fontsize = 25)
plt.xlabel('Variations',fontsize = 25)
plt.ylabel('Ratings',fontsize = 25)
plt.xticks(rotation = 90)

#plt.savefig('heartDiseaseAndAges.png')
plt.show()