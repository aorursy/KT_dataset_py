import numpy as np 

import pandas as pd 

import os

import itertools



#plots

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer



from PIL import Image

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

from nltk.util import ngrams





import re

from collections import Counter



import nltk

from nltk.corpus import stopwords



import requests

import json



import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})

social=pd.read_csv('../input/the-social-dilemma-tweets/TheSocialDilemma.csv')

social.info()
import missingno as mno

mno.matrix(social)
missed = pd.DataFrame()

missed['column'] = social.columns



missed['percent'] = [round(100* social[col].isnull().sum() / len(social), 2) for col in social.columns]

missed = missed.sort_values('percent',ascending=False)

missed = missed[missed['percent']>0]



fig = sns.barplot(

    x=missed['percent'], 

    y=missed["column"], 

    orientation='horizontal'

).set_title('Missed values percent for every column')
ds = social['Sentiment'].value_counts().reset_index()

ds.columns = ['Sentiment', 'Count']

ds = ds.sort_values(['Count'],ascending=False)

#social = pd.merge(social, ds, on='user_name')



fig = sns.barplot( 

    x=ds["Sentiment"], 

    y=ds["Count"], 

    orientation='vertical'

).set_title('Sentiment of the tweets') 
ds = social['user_name'].value_counts().reset_index()

ds.columns = ['user_name', 'tweets_count']

ds = ds.sort_values(['tweets_count'],ascending=False)

social = pd.merge(social, ds, on='user_name')



fig = sns.barplot( 

    x=ds.head(20)["tweets_count"], 

    y=ds.head(20)["user_name"], 

    orientation='horizontal'

).set_title('Top 20 users by number of tweets') 



social[social['user_name']=='OurPact'][['text','Sentiment']]
social[social['user_name']=='OurPact']['Sentiment'].value_counts()
social['user_created'] = pd.to_datetime(social['user_created'])

social['year_created'] = social['user_created'].dt.year

data = social.drop_duplicates(subset='user_name', keep="first")

data = data[data['year_created']>1970]

data = data['year_created'].value_counts().reset_index()

data.columns = ['year', 'number']



fig = sns.barplot( 

    x=data["year"], 

    y=data["number"], 

    orientation='vertical'

    #title='', 

).set_title('User created year by year')
ds = social['user_location'].value_counts().reset_index()

ds.columns = ['user_location', 'count']

ds = ds[ds['user_location']!='NA']

ds = ds.sort_values(['count'],ascending=False)



fig = sns.barplot(

    

    x=ds.head(20)["count"], 

    y=ds.head(20)["user_location"], 

    orientation='horizontal'

).set_title('Top 20 user locations by number of tweets')
from plotly.offline import init_notebook_mode, iplot

def pie_count(data, field, percent_limit, title):

    

    data[field] = data[field].fillna('NA')

    data = data[field].value_counts().to_frame()



    total = data[field].sum()

    data['percentage'] = 100 * data[field]/total    



    percent_limit = percent_limit

    otherdata = data[data['percentage'] < percent_limit] 

    others = otherdata['percentage'].sum()  

    maindata = data[data['percentage'] >= percent_limit]



    data = maindata

    other_label = "Others(<" + str(percent_limit) + "% each)"

    data.loc[other_label] = pd.Series({field:otherdata[field].sum()}) 

    

    labels = data.index.tolist()   

    datavals = data[field].tolist()

    

    trace=go.Pie(labels=labels,values=datavals)

    

    layout = go.Layout(

        title = title,

        height=600,

        width=600

        )

    

    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)

    

pie_count(social, 'user_location', 0.5, 'Number of tweets per location')
ds = social['source'].value_counts().reset_index()

ds.columns = ['source', 'count']

ds = ds.sort_values(['count'],ascending=False)



fig = sns.barplot(

    x=ds.head(10)["count"], 

    y=ds.head(10)["source"], 

    orientation='horizontal', 

).set_title('Top 10 user sources by number of tweets')
social['hashtags'] = social['hashtags'].fillna('[]')

social['hashtags_count'] = social['hashtags'].apply(lambda x: len(x.split(',')))

social.loc[social['hashtags'] == '[]', 'hashtags_count'] = 0

fig = sns.scatterplot( 

    x=social['hashtags_count'], 

    y=social['tweets_count']

).set_title('Total number of tweets for users and number of hashtags in every tweet')
ds = social['hashtags_count'].value_counts().reset_index()

ds.columns = ['hashtags_count', 'count']

ds = ds.sort_values(['count'],ascending=False)

ds['hashtags_count'] = ds['hashtags_count'].astype(str) + ' tags'

fig = sns.barplot( 

    x=ds["count"], 

    y=ds["hashtags_count"], 

    orientation='horizontal'

).set_title('Distribution of number of hashtags in tweets')
social['date'] = pd.to_datetime(social['date']) 

df = social.sort_values(['date'])

df['day'] = df['date'].astype(str).str.split(' ', expand=True)[0]

df['time'] = df['date'].astype(str).str.split(' ', expand=True)[1]

df.head()



ds = df.groupby(['day', 'user_name'])['hashtags_count'].count().reset_index()

ds = ds.groupby(['day'])['user_name'].count().reset_index()

ds.columns = ['day', 'number_of_users']

ds['day'] = ds['day'].astype(str)

fig = sns.barplot( 

    x=ds['day'], 

    y=ds["number_of_users"], 

    orientation='vertical',

    #title='Number of unique users per day', 

    #width=800, 

    #height=800

).set_title('Number of unique users per day')

#fig.show()

plt.xticks(rotation=90)
ds = df['day'].value_counts().reset_index()

ds.columns = ['day', 'count']

ds = ds.sort_values('count',ascending=False)

ds['day'] = ds['day'].astype(str)

fig = sns.barplot( 

    x=ds['count'], 

    y=ds["day"], 

    orientation="horizontal",

).set_title('Tweets distribution over days present in dataset')
social['tweet_date']=pd.to_datetime(social['date']).dt.date

tweet_date=social['tweet_date'].value_counts().to_frame().reset_index().rename(columns={'index':'date','tweet_date':'count'})

tweet_date['date']=pd.to_datetime(tweet_date['date'])

tweet_date=tweet_date.sort_values('date',ascending=False)
fig=go.Figure(go.Scatter(x=tweet_date['date'],

                                y=tweet_date['count'],

                               mode='markers+lines',

                               name="Submissions",

                               marker_color='dodgerblue'))



fig.update_layout(

    title_text='Tweets per Day : ({} - {})'.format(social['tweet_date'].sort_values()[0].strftime("%d/%m/%Y"),

                                                       social['tweet_date'].sort_values().iloc[-1].strftime("%d/%m/%Y")),template="plotly_dark",

    title_x=0.5)



fig.show()
social['hour'] = social['date'].dt.hour

ds = social['hour'].value_counts().reset_index()

ds.columns = ['hour', 'count']

ds['hour'] = 'Hour ' + ds['hour'].astype(str)

fig = sns.barplot( 

    x=ds["hour"], 

    y=ds["count"], 

    orientation='vertical', 

).set_title('Tweets distribution over hours')

plt.xticks(rotation='vertical')

def split_hashtags(x): 

    return str(x).replace('[', '').replace(']', '').split(',')



tweets_df = social.copy()

tweets_df['hashtag'] = tweets_df['hashtags'].apply(lambda row : split_hashtags(row))

tweets_df = tweets_df.explode('hashtag')

tweets_df['hashtag'] = tweets_df['hashtag'].astype(str).str.lower().str.replace("'", '').str.replace(" ", '')

tweets_df.loc[tweets_df['hashtag']=='', 'hashtag'] = 'NO HASHTAG'

#tweets_df
ds = tweets_df['hashtag'].value_counts().reset_index()

ds.columns = ['hashtag', 'count']

ds = ds.sort_values(['count'],ascending=False)

fig = sns.barplot(

    x=ds.head(10)["count"], 

    y=ds.head(10)['hashtag'], 

    orientation='horizontal', 

).set_title('Top 10 hashtags')
def build_wordcloud(df, title):

    wordcloud = WordCloud(

        background_color='black',colormap="Oranges", 

        stopwords=set(STOPWORDS), 

        max_words=50, 

        max_font_size=40, 

        random_state=666

    ).generate(str(df))



    fig = plt.figure(1, figsize=(14,14))

    plt.axis('off')

    fig.suptitle(title, fontsize=16)

    fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
build_wordcloud(social['text'], 'Prevalent words in tweets for all dataset')
india_df = social.loc[social.user_location=="India"]

build_wordcloud(india_df['text'], title = 'Prevalent words in tweets from India')
def remove_tag(string):

    text=re.sub('<.*?>','',string)

    return text

def remove_mention(text):

    line=re.sub(r'@\w+','',text)

    return line

def remove_hash(text):

    line=re.sub(r'#\w+','',text)

    return line



def remove_newline(string):

    text=re.sub('\n','',string)

    return text

def remove_url(string): 

    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',string)

    return text

def remove_number(text):

    line=re.sub(r'[0-9]+','',text)

    return line

def remove_punct(text):

    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*','',text)

    return line

def text_strip(string):

    line=re.sub('\s{2,}', ' ', string.strip())

    return line

def remove_thi_amp_ha_words(string):

    line=re.sub(r'\bamp\b|\bthi\b|\bha\b',' ',string)

    return line
social['refine_text']=social['text'].str.lower()

social['refine_text']=social['refine_text'].apply(lambda x:remove_tag(str(x)))

social['refine_text']=social['refine_text'].apply(lambda x:remove_mention(str(x)))

social['refine_text']=social['refine_text'].apply(lambda x:remove_hash(str(x)))

social['refine_text']=social['refine_text'].apply(lambda x:remove_newline(x))

social['refine_text']=social['refine_text'].apply(lambda x:remove_url(x))

social['refine_text']=social['refine_text'].apply(lambda x:remove_number(x))

social['refine_text']=social['refine_text'].apply(lambda x:remove_punct(x))

social['refine_text']=social['refine_text'].apply(lambda x:remove_thi_amp_ha_words(x))

social['refine_text']=social['refine_text'].apply(lambda x:text_strip(x))



social['text_length']=social['refine_text'].str.split().map(lambda x: len(x))
fig = go.Figure(data=go.Violin(y=social['text_length'], box_visible=True, line_color='black',

                               meanline_visible=True, fillcolor='royalblue', opacity=0.6,

                               x0='Tweet Text Length'))



fig.update_layout(yaxis_zeroline=False,title="Distribution of Text length",template='ggplot2')

fig.show()
def ngram_df(corpus,nrange,n=None):

    vec = CountVectorizer(stop_words = 'english',ngram_range=nrange).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    total_list=words_freq[:n]

    df=pd.DataFrame(total_list,columns=['text','count'])

    return df

unigram_df=ngram_df(social['refine_text'],(1,1),20)

bigram_df=ngram_df(social['refine_text'],(2,2),20)

trigram_df=ngram_df(social['refine_text'],(3,3),20)
fig = make_subplots(

    rows=3, cols=1,subplot_titles=("Unigram","Bigram",'Trigram'),

    specs=[[{"type": "scatter"}],

           [{"type": "scatter"}],

           [{"type": "scatter"}]

          ])



fig.add_trace(go.Bar(

    y=unigram_df['text'][::-1],

    x=unigram_df['count'][::-1],

    marker={'color': "blue"},  

    text=unigram_df['count'],

    textposition = "outside",

    orientation="h",

    name="Months",

),row=1,col=1)



fig.add_trace(go.Bar(

    y=bigram_df['text'][::-1],

    x=bigram_df['count'][::-1],

    marker={'color': "blue"},  

    text=bigram_df['count'],

     name="Days",

    textposition = "outside",

    orientation="h",

),row=2,col=1)



fig.add_trace(go.Bar(

    y=trigram_df['text'][::-1],

    x=trigram_df['count'][::-1],

    marker={'color': "blue"},  

    text=trigram_df['count'],

     name="Days",

    orientation="h",

    textposition = "outside",

),row=3,col=1)



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='Top N Grams',xaxis_title=" ",yaxis_title=" ",

                  showlegend=False,title_x=0.5,height=1200,template="plotly_dark")

fig.show()