import numpy as np
import pandas as pd
import os
import string
import re
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import squarify
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

td = pd.read_csv('../input/demo - Sheet1.csv',parse_dates=['Date'])


td.dtypes
td['User Since']= pd.to_datetime(td['User Since']) 
td.dtypes
td.shape
td = td.drop('Full Name',axis =1)
td.isnull().mean()*100
td.head(5)
to_drop=['Tweet ID','Profile Image','Website']
td.drop(to_drop, inplace=True, axis=1)
import unicodedata
from unidecode import unidecode

def deEmojify(inputString):
    returnString = ""

    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced != '':
                returnString += replaced
            else:
                try:
                     returnString += "[" + unicodedata.name(character) + "]"
                except ValueError:
                     returnString += "[x]"

    return returnString
string = td['Tweet_Text']
# print(deEmojify(string))
td = td[~td['Tweet_Text'].isnull()]

def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')
    ReviewText = ReviewText.str.replace('(@)', ' ') 
    ReviewText = ReviewText.str.replace('(#)', ' ')
    ReviewText = ReviewText.str.replace('(0)', ' ')
    ReviewText = ReviewText.str.replace('(1)', ' ')
    ReviewText = ReviewText.str.replace('(2)', ' ')
    ReviewText = ReviewText.str.replace('(3)', ' ')
    ReviewText = ReviewText.str.replace('(4)', ' ')
    ReviewText = ReviewText.str.replace('(5)', ' ')
    ReviewText = ReviewText.str.replace('(6)', ' ')
    ReviewText = ReviewText.str.replace('(7)', ' ')
    ReviewText = ReviewText.str.replace('(8)', ' ')
    ReviewText = ReviewText.str.replace('(9)', ' ')
    ReviewText = ReviewText.str.replace('(-)', ' ')
    ReviewText = ReviewText.str.replace('(:)', ' ')
    ReviewText = ReviewText.str.replace('(/)', ' ')
    ReviewText = ReviewText.str.replace('(,)', ' ')
    ReviewText = ReviewText.str.replace('(")', ' ')
    ReviewText = ReviewText.str.replace('(-)', ' ')
    ReviewText = ReviewText.str.replace('(])', ' ')
    ReviewText = ReviewText.str.replace('(!)', ' ')
    ReviewText = ReviewText.str.replace('(_)', ' ')
    ReviewText = ReviewText.str.replace('(:)', ' ')
    ReviewText = ReviewText.str.replace('(;)', ' ')
    ReviewText = ReviewText.str.replace('(¦)', ' ')
    ReviewText = ReviewText.str.replace('(>)', ' ')
    ReviewText = ReviewText.str.replace('(<)', ' ')
    ReviewText = ReviewText.str.replace('(©)', ' ')
    ReviewText = ReviewText.str.replace('($)', ' ')
    ReviewText = ReviewText.str.replace('(^)', ' ')


  


    



    return ReviewText
td['Tweet_Text'] = preprocess(td['Tweet_Text'])
td.head()

# dframe.to_csv(“../input/demo - Sheet1.csv”)
print(f" Data Available since {td.Date.min()}")
print(f" Data Available upto {td.Date.max()}")
# print(f" Data Available since {td.User Since.min()}")
# print(f" Data Available upto {td.User Since.max()}")
td['Date'] =  pd.to_datetime(td['Date'])
cnt_srs = td['Date'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='red')
plt.xticks(rotation='vertical')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of tweets', fontsize=12)
plt.title("Number of tweets according to dates")
plt.show()
count_  = td['User Since'].dt.date.value_counts()
count_ = count_[:10,]
plt.figure(figsize=(10,5))
sns.barplot(count_.index, count_.values, alpha=0.8)
plt.title('Most accounts created according to date')
plt.xticks(rotation='vertical')
plt.ylabel('Number of accounts', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.show()
td['tweeted_day_of_week'] = td['Date'].dt.weekday_name
td['created_day_of_week'] = td['User Since'].dt.weekday_name
cnt_ = td['tweeted_day_of_week'].value_counts()
cnt_ = cnt_.sort_index() 
fig = {
  "data": [
    {
      "values": cnt_.values,
      "labels": cnt_.index,
      "domain": {"x": [0, .5]},
      "name": "Number of tweets per day",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Percentage of tweets per days of the week",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
             "text": "Percentage of Tweets according to days of the week",
                "x": 0.50,
                "y": 1
            },
        ]
    }
}
iplot(fig)
cnt_
x = 0.
y = 0.
width = 50.
height = 50.
type_list = list(td['tweeted_day_of_week'].unique())
values = [len(td[td['tweeted_day_of_week'] == i]) for i in type_list]

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = ['#2D3142','#4F5D75','#BFC0C0','#F2D7EE','#EF8354','#839788','#EEE0CB']
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 2 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}-{}".format(type_list[counter], values[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)
        
layout = dict(
    height=700, 
    width=700,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF")
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
iplot(figure, filename='squarify-treemap')
cnt_ = td['created_day_of_week'].value_counts()
cnt_ = cnt_.sort_index() 
fig = {
  "data": [
    {
      "values": cnt_.values,
      "labels": cnt_.index,
      "domain": {"x": [0, .5]},
      "name": "Number of tweets per day",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Percentage of created accounts per day",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
             "text": "Percentage of accounts created according to days of the week",
                "x": 0.50,
                "y": 1
            },
        ]
    }
}
iplot(fig)
cnt_
x = 0.
y = 0.
width = 50.
height = 50.
type_list = list(td['created_day_of_week'].unique())
values = [len(td[td['created_day_of_week'] == i]) for i in type_list]

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = ['#99B2DD','#F9DEC9','#3A405A','#494949','#FF5D73','#7C7A7A']
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 2 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}-{}".format(type_list[counter], values[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)
        
layout = dict(
    height=700, 
    width=700,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF")
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
iplot(figure, filename='squarify-tree')
td['created_at_hour'] = td['Date'].dt.hour
td['user_created_at_hour'] = td['User Since'].dt.hour
cnt_ = td['created_at_hour'].value_counts()
cnt_ = cnt_.sort_index() 
trace1 = go.Scatter(
                    x = cnt_.index,
                    y = cnt_.values,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)')
                    )

data = [trace1]
layout = dict(title = 'Number of tweets per hour',
              xaxis= dict(title= 'Tweets per hour',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
cnt_ = td['user_created_at_hour'].value_counts()
cnt_ = cnt_.sort_index() 
trace1 = go.Scatter(
                    x = cnt_.index,
                    y = cnt_.values,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(210, 113, 25, 0.8)')
                    )

data = [trace1]
layout = dict(title = 'Number of Accounts Created per hour ',
              xaxis= dict(title= 'Accounts per hour',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
print(f" Maximum number of retweets {td.Retweets.max()}")
print(f" Maximum number of favorites {td.Favorites.max()}")
td.loc[td['Retweets']==109.0,'Tweet_Text'].values
td.loc[td['Favorites']==313.0,['Tweet_text','Screen Name','Bio']].values
wordcloud__ = WordCloud(
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          max_words=250,
                          max_font_size=40, 
                          random_state=1705
                         ).generate(str(td['Screen Name'].dropna()))
def cloud_plot(wordcloud):
    fig = plt.figure(1, figsize=(20,15))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
cloud_plot(wordcloud__)
td['sentiment'] = td['Tweet_Text'].map(lambda text: TextBlob(text).sentiment.polarity)
print("5 random tweets with highest positive sentiment polarity: \n")
cL = td.loc[td.sentiment==1, ['Tweet_Text']].sample(1).values
for c in cL:
    print(c[0])
    print()
trace1 = go.Histogram(
    x = td['sentiment'],
    opacity=0.75,
    name = "Sentiment",
    marker=dict(color='rgba(122, 75, 196, 0.6)'))

data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='Histogram plot of sentiment',
                   xaxis=dict(title='Sentiment'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
cut = pd.cut(
    td['sentiment'],
    [-np.inf, -.01, .01, np.inf],
    labels=['negative', 'neutral', 'positive']
)
td['polarity'] = cut.values
td[['polarity','sentiment']][:20]
td['polarity'].value_counts()
data = [go.Scatterpolar(
  r = [td['polarity'].value_counts()[0],td['polarity'].value_counts()[1],td['polarity'].value_counts()[2]],
  theta = list(td['polarity'].unique()),
  fill = 'toself'
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 300]
    )
  ),
  showlegend = False,
  title ='Radar chart of polarities'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "Single Pokemon stats")
#td['count_sent']=td["Tweet_Text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
#Word count in each comment:
#td['count_word']=td["Tweet_Text"].apply(lambda x: len(str(x).split()))
#Unique word count
#td['count_unique_word']=td["Tweet_Text"].apply(lambda x: len(set(str(x).split())))
#Letter count
#td['count_letters']=td["Tweet_Text"].apply(lambda x: len(str(x)))
#punctuation count
#td["count_punctuations"] =td["Tweet_Text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
#td["count_words_upper"] = td["Tweet_Text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
#td["count_words_title"] = td["Tweet_Text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
#td["count_stopwords"] = td["Tweet_Text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
#Average length of the words
#td["mean_word_len"] = td["Tweet_Text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))