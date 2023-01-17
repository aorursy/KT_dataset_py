# Importing Libraries 



from fastai.metrics import accuracy

from fastai.text import *

from fastai import *

import gc

import os

import pandas as pd

import numpy as np
tweets = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')

print('Shape: ', tweets.shape)

tweets.head()
def check_missing_data(df):

    flag=df.isna().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum())/(df.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)
check_missing_data(tweets)
# droping 3 columns which are 90% empty



tweets = tweets.drop(columns=['airline_sentiment_gold', 'negativereason_gold', 'tweet_coord'])
tweets.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False).plot(kind='bar')

plt.title('Number of tweets wrt Airlines')

plt.show()
airlines= ['US Airways','United','American','Southwest','Delta','Virgin America']



plt.figure(1,figsize=(16, 12))

for i in airlines:

    indices= airlines.index(i)

    plt.subplot(2,3,indices+1)

    new_df=tweets[tweets['airline']==i]

    count=new_df['airline_sentiment'].value_counts()

    Index = [1,2,3]

    plt.bar(Index,count, color=['red', 'green', 'blue'])

    plt.xticks(Index,['negative','neutral','positive'])

    plt.ylabel('Mood Count')

    plt.xlabel('Mood')

    plt.title('Count of Moods of '+i)
from wordcloud import WordCloud,STOPWORDS



def generate_wordcloud(text, title): 

    wordcloud = WordCloud(relative_scaling = 1.0,stopwords = STOPWORDS, background_color='black').generate(text)

    fig,ax = plt.subplots(1,1,figsize=(14,18))

    ax.imshow(wordcloud, interpolation='bilinear')

    ax.axis("off")

    ax.margins(x=0, y=0)

    plt.title(title)

    plt.show()
words_positive = ' '.join(tweets[tweets['airline_sentiment']=='positive']['text'])

cleaned_word_positive = " ".join([word for word in words_positive.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])



generate_wordcloud(cleaned_word_positive, 'Positive Tweets')
words_negative = ' '.join(tweets[tweets['airline_sentiment']=='negative']['text'])

cleaned_word_negative = " ".join([word for word in words_negative.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])



generate_wordcloud(cleaned_word_negative, 'Negative Tweets')
words_neutral = ' '.join(tweets[tweets['airline_sentiment']=='neutral']['text'])

cleaned_word_neutral = " ".join([word for word in words_neutral.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])



generate_wordcloud(cleaned_word_negative, 'Neutral Tweets')
import re

import nltk

from nltk.corpus import stopwords



def tweet_to_words(raw_tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 



def clean_tweet_length(raw_tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return(len(meaningful_words)) 



tweets['clean_tweet']=tweets['text'].apply(lambda x: tweet_to_words(x))

tweets['Tweet_length']=tweets['text'].apply(lambda x: clean_tweet_length(x))

tweets['sentiment']=tweets['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")



# split data into training and validation set

df_trn, df_val = train_test_split(tweets, stratify = tweets['airline_sentiment'], test_size = 0.4, random_state = 678)
# Language model data

data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")
learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.4)
# train the learner object with learning rate = 1e-2

learn.fit_one_cycle(1, 1e-2)
# unfreeze the learner object and train

learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-2, 1e-5))
learn.save_encoder('ft_enc')