

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from textblob import TextBlob

from langdetect import detect, DetectorFactory

import os

print(os.listdir("../input"))

from langdetect import detect, DetectorFactory

import re,string
pd.set_option('display.max_colwidth', -1)

df_tweets = pd.read_csv('../input/merged_1.csv', encoding='latin-1')

df_symptoms = pd.read_csv('../input/Symptom.csv', encoding='latin-1')
def relevant_tweet(df,value):

    return df[df['text'].str.contains(value)]



def remove_null(df,col):

    return df.dropna(subset=[col],inplace=True)



remove_null(df_tweets,'text')

text = df_tweets['text']



def clean_tweet(value):

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",value).split())



df_tweets['clean_tweet'] = text.apply(clean_tweet)



def check_language(value):

    DetectorFactory.seed = 0

    try:

        lang = detect(value)

        return lang

    except:

        pass

text = df_tweets['clean_tweet']

df_tweets['language'] = text.apply(check_language)
df_tweets_eng = df_tweets[df_tweets['language'] == 'en']

df_tweets_eng
def get_tweet_sentiment(tweet):

    tweet = TextBlob(tweet)

    if tweet.sentiment.polarity > 0: 

        return 'positive'

    elif tweet.sentiment.polarity == 0: 

        return 'neutral'

    else: 

        return 'negative'

tweet = df_tweets_eng['clean_tweet']

df_tweets_eng['sentiment'] = tweet.apply(get_tweet_sentiment)

df_tweets_negative = df_tweets_eng[df_tweets_eng['sentiment']=='negative']
# generate 'brands' DF

sentiment = pd.DataFrame(df_tweets_negative.text.value_counts().reset_index())

sentiment.columns = ['text', 'count']

# print(sentiment)



# merge 'df' & 'brands_count'

merged = pd.merge(df_tweets_negative, sentiment, on='text')

merged
symptom_list = df_symptoms['Symptom'].values.tolist()

def check_symptoms(tweet):

    match ={}

    lisss = tweet.split()

    for sym in symptom_list:

        if sym in lisss:

            if sym in match.keys():

                match[sym] = match[sym]+1

            else:

                match[sym] = 1

    if match:

        print(match)

        severity = df_symptoms[df_symptoms['Symptom'].isin(match.keys())][['Symptom','Severity']].values.tolist()

        sev_dict = {}

        for s in severity:

            sev_dict[s[0]] = s[1]

        print(sev_dict)



        numerator = 0

        denom = 0

        for key, value in sev_dict.items():

            numerator = numerator + (match[key] * value)

            denom = denom + match[key]



        return numerator/denom

    else:

        return 0



clean = merged['clean_tweet']

merged['index'] = clean.apply(check_symptoms)

merged[merged['index']> 0].drop_duplicates(subset='clean_tweet',keep='first')
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



text = merged[merged['index']> 0].drop_duplicates(subset='clean_tweet',keep='first')['clean_tweet'].values

stop_words = ["Nuplazid","Digoxin","bigger","right", "Xolair", "Avastin"] + list(STOPWORDS)

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = stop_words).generate(str(text))



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()