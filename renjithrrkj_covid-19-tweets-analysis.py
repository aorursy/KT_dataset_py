!pip install https://github.com/elyase/geotext/archive/master.zip

!pip install topojson

    
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install country_converter --upgrade

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from geotext import GeoText

import json

import topojson

import country_converter as coco

from nltk.stem import LancasterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer 

#this was part of the NLP notebook

import nltk

nltk.download('punkt')

#import sentence tokenizer

from nltk import sent_tokenize

#import word tokenizer

from nltk import word_tokenize

#list of stopwords

from nltk.corpus import stopwords

import string

#import geograpy

import emoji

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from wordcloud import WordCloud,STOPWORDS
covid_df=pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
covid_df.head(

)
covid_df.shape
Location_count=pd.DataFrame(covid_df['user_location'].value_counts())
Location_count.head()
Location_count.reset_index(inplace=True)
Location_count.rename(columns={'index':'Location','user_location':'count'},inplace=True)
Location_count.sort_values(by='count',inplace=True,ascending=False)
#Location_count
Count_graph=px.bar(x='count',y='Location',data_frame=Location_count[:15],color='Location')

Count_graph.show()
Location_count.shape
location=Location_count.loc[2]['Location']
print(GeoText(location).countries)
Location_country=Location_count.copy()
Location_country['Location']=Location_country['Location'].apply(lambda x:x.replace(',',' '))
#Location_country
Location_country['Location']=Location_country['Location'].apply(lambda x:(GeoText(x).country_mentions))
Location_country.head()
Location_country.drop(Location_country[Location_country['Location']=='[]'].index,inplace=True)
Location_country['Location']=Location_country['Location'].apply(lambda x:(x.keys()))
Location_country['Location']=Location_country['Location'].apply(lambda x:list(x))
Location_country.drop(Location_country.index[Location_country.Location.map(len)==0],inplace=True)
#Location_country
Location_country['Location']=Location_country['Location'].apply(lambda x:str(x[0]))
#Location_country
agg_func={'count':'sum'}

Location_country=Location_country.groupby(['Location']).aggregate(agg_func)
Location_country.head()
Location_country.sort_values(by=['count'],ascending=False,inplace=True)

Location_country.reset_index(inplace=True)

Location_country.columns
#Location_country['Location']=Location_country['Location'].apply(lambda x:x[2:-2])
Count_graph=px.bar(x='Location',y='count',data_frame=Location_country[:15],color='Location')

Count_graph.show()
cc = coco.CountryConverter()

Location_country['Location']=Location_country['Location'].apply(lambda x:cc.convert(names=x,to='ISO3'))
Location_country
india_states = json.load(open("../input/country-state-geo-location/countries.geo.json", "r"))

fig = px.choropleth(

    Location_country,

    locations="Location",

    geojson=india_states,

    color="count",

    #hover_name="State or union territory",

    hover_data=["count"],

    title="number of tweets from each country",

)

fig.update_geos(fitbounds="locations", visible=False)

fig.show()
tweets=pd.DataFrame(covid_df['text'])
tweets
import re

def char_is_emoji(character):

    return character in emoji.UNICODE_EMOJI

#does the text contain an emoji?

def text_has_emoji(text):

    for character in text:

        if character in emoji.UNICODE_EMOJI:

            return True

    return False

#remove the emoji

def deEmojify(inputString):

    return inputString.encode('ascii', 'ignore').decode('ascii')
punct =[]

punct += list(string.punctuation)

punct += '’'

punct.remove("'")

def remove_punctuations(text):

    for punctuation in punct:

        text = text.replace(punctuation, ' ')

    return text
def nlp(df):

    # lowercase everything

    # get rid of '\n' from whitespace

    # regex remove hyperlinks

    # removing '&gt;'

    # check for emojis

    # remov

        # lowercase everything

    df['token'] = df['text'].apply(lambda x: x.lower())

    # get rid of '\n' from whitespace 

    df['token'] = df['token'].apply(lambda x: x.replace('\n', ' '))

    # regex remove hyperlinks

    df['token'] = df['token'].str.replace('http\S+|www.\S+', '', case=False)

    # removing '&gt;'

    df['token'] = df['token'].apply(lambda x: x.replace('&gt;', ''))

    # Checking if emoji in tokens column, use for EDA purposes otherwise not necessary to keep this column

    df['emoji'] = df['token'].apply(lambda x: text_has_emoji(x))

    # Removing Emojis from tokens

    #df['token'] = df['token'].apply(lambda x: deEmojify(x))

    # remove punctuations

    #df['token'] = df['token'].apply(remove_punctuations)

    # remove ' s ' that was created after removing punctuations

    df['token'] = df['token'].apply(lambda x: str(x).replace(" s ", " "))

    return df
tweets1=(nlp(tweets))
(tweets1)
comment_words=''

for val in tweets1.token: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

  

wordcloud1 = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = STOPWORDS, 

                min_font_size = 10).generate(comment_words)

plt.figure(figsize = (10,10), facecolor = None) 

plt.imshow(wordcloud1) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
def categoriser(diction):

    if(diction['neg']>0):

        return("Negative")

    elif(diction['pos']>0):

        return('Positive')

    else:

        return('Neutral')
def SentiAnlyser(df):

    analyser= SentimentIntensityAnalyzer()

    df['sentiment']=df['token'].apply(lambda x: analyser.polarity_scores(x))

    df['sentiment']=df['sentiment'].apply(lambda x:categoriser(x))

    return df
tweets2=SentiAnlyser(tweets1)
tweets2.head()
tweets2.to_csv('./sentiment.csv')
tweets2.iloc[22]
tweet_sentiments=pd.DataFrame(tweets2['sentiment'].value_counts())

tweet_sentiments.reset_index(inplace=True)

tweet_sentiments.rename(columns={'index':'Sentiment','sentiment':'count'},inplace=True)

fig=px.pie(tweet_sentiments,values='count',names='Sentiment',title="Sentiments of Tweets")
fig.show()