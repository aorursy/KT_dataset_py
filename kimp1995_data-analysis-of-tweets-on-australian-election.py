import numpy as np

import pandas as pd

import os

import string

import re

import warnings 

warnings.filterwarnings('ignore')



#plotting libraries!

import matplotlib.pyplot as plt

import seaborn as sns

from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame

%matplotlib inline





#PLOTLY

import plotly

import plotly.plotly as py

import plotly.offline as offline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

from collections import defaultdict

from plotly import tools

from plotly.graph_objs import Scatter, Figure, Layout

cf.set_config_file(offline=True)

from textblob import TextBlob

from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))

from wordcloud import WordCloud, STOPWORDS

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer

import pyLDAvis.sklearn

from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

import squarify



print(os.listdir('../input'))
twitter_data = pd.read_csv('../input/auspol2019.csv',parse_dates=['created_at','user_created_at'])

geo_data = pd.read_csv('../input/location_geocode.csv')
twitter_data.head()
geo_data.head()
twitter_data.shape
geo_data.shape
#merging two data frames based on user location

twitter_data = twitter_data.merge(geo_data, how='inner', left_on='user_location', right_on='name')
twitter_data.head()
twitter_data = twitter_data.drop('name',axis =1)
#lets check for null values

twitter_data.isnull().mean()*100
print(f" Data Available since {twitter_data.created_at.min()}")

print(f" Data Available upto {twitter_data.created_at.max()}")
#lets check latest and oldest twitter members in the dataframe

print(f" Data Available since {twitter_data.user_created_at.min()}")

print(f" Data Available upto {twitter_data.user_created_at.max()}")
print('The oldest user in the data was',twitter_data.loc[twitter_data['user_created_at'] == '2006-03-21 21:04:12', 'user_name'].values)
print('The newest user in the data was',twitter_data.loc[twitter_data['user_created_at'] == '2019-05-19 10:49:59', 'user_name'].values)
#lets explore created_at column

twitter_data['created_at'] =  pd.to_datetime(twitter_data['created_at'])

cnt_srs = twitter_data['created_at'].dt.date.value_counts()

cnt_srs = cnt_srs.sort_index()

plt.figure(figsize=(14,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')

plt.xticks(rotation='vertical')

plt.xlabel('Date', fontsize=12)

plt.ylabel('Number of tweets', fontsize=12)

plt.title("Number of tweets according to dates")

plt.show()
#lets explore user_created_at column

count_  = twitter_data['user_created_at'].dt.date.value_counts()

count_ = count_[:10,]

plt.figure(figsize=(10,5))

sns.barplot(count_.index, count_.values, alpha=0.8)

plt.title('Most accounts created according to date')

plt.xticks(rotation='vertical')

plt.ylabel('Number of accounts', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.show()
#lets derive some columns from date colums

twitter_data['tweeted_day_of_week'] = twitter_data['created_at'].dt.weekday_name

twitter_data['created_day_of_week'] = twitter_data['user_created_at'].dt.weekday_name
cnt_ = twitter_data['tweeted_day_of_week'].value_counts()

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

type_list = list(twitter_data['tweeted_day_of_week'].unique())

values = [len(twitter_data[twitter_data['tweeted_day_of_week'] == i]) for i in type_list]



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
cnt_ = twitter_data['created_day_of_week'].value_counts()

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

type_list = list(twitter_data['created_day_of_week'].unique())

values = [len(twitter_data[twitter_data['created_day_of_week'] == i]) for i in type_list]



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
#lets extract the hours from the created_at and user_created_at column

twitter_data['created_at_hour'] = twitter_data['created_at'].dt.hour

twitter_data['user_created_at_hour'] = twitter_data['user_created_at'].dt.hour
cnt_ = twitter_data['created_at_hour'].value_counts()

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
cnt_ = twitter_data['user_created_at_hour'].value_counts()

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
#most favourite and retweeted tweet

print(f" Maximum number of retweets {twitter_data.retweet_count.max()}")

print(f" Maximum number of favorites {twitter_data.favorite_count.max()}")
#lets see the tweet which has the maximum retweet count

twitter_data.loc[twitter_data['retweet_count']==6622.0,'full_text'].values
twitter_data.loc[twitter_data['favorite_count']==15559.0,['full_text','user_name','user_description']].values
#most number of occurances of a person

twitter_data.user_name.value_counts()[:5,]
#wordcloud



wordcloud__ = WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(str(twitter_data['user_screen_name'].dropna()))

def cloud_plot(wordcloud):

    fig = plt.figure(1, figsize=(20,15))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

cloud_plot(wordcloud__)
#wordcloud

wordcloud_ = WordCloud(

                          background_color='black',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(str(twitter_data['user_description'].dropna()))

def cloud_plot(wordcloud):

    fig = plt.figure(1, figsize=(20,15))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

cloud_plot(wordcloud_)
twitter_data['sentiment'] = twitter_data['full_text'].map(lambda text: TextBlob(text).sentiment.polarity)
print("5 random tweets with highest positive sentiment polarity: \n")

cL = twitter_data.loc[twitter_data.sentiment==1, ['full_text']].sample(5).values

for c in cL:

    print(c[0])

    print()
print("5 random tweets with highest nagative sentiment polarity: \n")

cL = twitter_data.loc[twitter_data.sentiment==-1, ['full_text']].sample(5).values

for c in cL:

    print(c[0])

    print()
print("5 random tweets with neutral sentiment polarity: \n")

cL = twitter_data.loc[twitter_data.sentiment==0, ['full_text']].sample(5).values

for c in cL:

    print(c[0])

    print()
trace1 = go.Histogram(

    x = twitter_data['sentiment'],

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

    twitter_data['sentiment'],

    [-np.inf, -.01, .01, np.inf],

    labels=['negative', 'neutral', 'positive']

)

twitter_data['polarity'] = cut.values

twitter_data[['polarity','sentiment']][:20]
twitter_data['polarity'].value_counts()
data = [go.Scatterpolar(

  r = [twitter_data['polarity'].value_counts()[0],twitter_data['polarity'].value_counts()[1],twitter_data['polarity'].value_counts()[2]],

  theta = list(twitter_data['polarity'].unique()),

  fill = 'toself'

)]



layout = go.Layout(

  polar = dict(

    radialaxis = dict(

      visible = True,

      range = [0, 60000]

    )

  ),

  showlegend = False,

  title ='Radar chart of polarities'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename = "Single Pokemon stats")
twitter_data['count_sent']=twitter_data["full_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Word count in each comment:

twitter_data['count_word']=twitter_data["full_text"].apply(lambda x: len(str(x).split()))

#Unique word count

twitter_data['count_unique_word']=twitter_data["full_text"].apply(lambda x: len(set(str(x).split())))

#Letter count

twitter_data['count_letters']=twitter_data["full_text"].apply(lambda x: len(str(x)))

#punctuation count

twitter_data["count_punctuations"] =twitter_data["full_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count

twitter_data["count_words_upper"] = twitter_data["full_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#title case words count

twitter_data["count_words_title"] = twitter_data["full_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords

twitter_data["count_stopwords"] = twitter_data["full_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words

twitter_data["mean_word_len"] = twitter_data["full_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
twitter_data.describe().T
sample_df = twitter_data[['count_sent','count_word','count_unique_word','count_letters','count_punctuations','count_words_upper','count_words_title','count_stopwords','mean_word_len' ]]

sns.pairplot(sample_df,palette="husl")

del sample_df
def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace





freq_dict = defaultdict(int)

for sent in twitter_data["full_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')





fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,

                          subplot_titles=["Frequent words"

                                          ])

fig.append_trace(trace0, 1, 1)



fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

iplot(fig, filename='word-plots.html')

freq_dict = defaultdict(int)

for sent in twitter_data["full_text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent bigrams"

                                          ])

fig.append_trace(trace0, 1, 1)

fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

iplot(fig, filename='word-plots')
freq_dict = defaultdict(int)

plotly.tools.set_credentials_file(username='Ratan2513', api_key='atZYQqpeRmUlL5jaST4E')

for sent in twitter_data["full_text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'green')



fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04, horizontal_spacing=0.2,

                          subplot_titles=["Frequent trigrams", 

                                          ])

fig.append_trace(trace0, 1, 1)

fig['layout'].update(height=1200, width=1500, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

py.iplot(fig, filename='word-plots')
cnt_ = twitter_data['user_location'].value_counts()

cnt_.reset_index()

cnt_ = cnt_[:20,]

trace1 = go.Bar(

                x = cnt_.index,

                y = cnt_.values,

                name = "Number of tweets on Australia polls by state.",

                marker = dict(color = 'rgba(200, 74, 55, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                )



data = [trace1]

layout = go.Layout(barmode = "group",title = 'Number of tweets on Australia polls by state.')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
data = [go.Scattermapbox(

            lat= twitter_data['lat'] ,

            lon= twitter_data['long'],

            mode='markers',

            marker=dict(

                size= 4,

                color = 'orange',

                opacity = .8,

            ),

          )]

layout = go.Layout(

    title = go.layout.Title(

        text = 'Tweets on Australia polls by state'

    ),

    geo = go.layout.Geo(

        scope = 'world',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'),

)



fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename = 'd3-cloropleth')
trace1 = go.Scattermapbox(

            lat= twitter_data.loc[twitter_data['polarity'] == 'negative','lat'] ,

            lon= twitter_data.loc[twitter_data['polarity'] == 'negative','long'],

            mode='markers',

            marker=dict(

                size= 4,

                color = 'black',

                opacity = .5,

            ),

          )

trace2= go.Scattermapbox(

            lat= twitter_data.loc[twitter_data['polarity'] == 'neutral','lat'] ,

            lon= twitter_data.loc[twitter_data['polarity'] == 'neutral','long'],

            mode='markers',

            marker=dict(

                size= 4,

                color = 'blue',

                opacity = .3,

            ),

          )

trace3= go.Scattermapbox(

            lat= twitter_data.loc[twitter_data['polarity'] == 'positive','lat'] ,

            lon= twitter_data.loc[twitter_data['polarity'] == 'positive','long'],

            mode='markers',

            marker=dict(

                size= 4,

                color = 'gold',

                opacity = .2,

            ),

          )





data = [trace1,trace2,trace3]

layout = go.Layout(

    title = go.layout.Title(

        text = 'Tweets on Australia polls according to polarity by state '

    ),

    geo = go.layout.Geo(

        scope = 'world',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(200, 125, 255)'),

)



fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename = 'd3-cloropleth-ma')
vectorizer_ = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

tweets_vectorized = vectorizer_.fit_transform(twitter_data['full_text'])
lda_ = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)

tweets_lda = lda_.fit_transform(tweets_vectorized)
def selected_topics(model, vectorizer, top_n=10):

    for idx, topic in enumerate(model.components_):

        print("Topic %d:" % (idx))

        print([(vectorizer.get_feature_names()[i], topic[i])

                        for i in topic.argsort()[:-top_n - 1:-1]]) 
print("Tweets LDA Model:")

selected_topics(lda_, vectorizer_)
pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_, tweets_vectorized, vectorizer_, mds='tsne')

dash