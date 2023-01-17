from sklearn.cross_validation import train_test_split

from stop_words import get_stop_words

import matplotlib.pyplot as plt

from __future__ import division

from collections import Counter

import pandas as pd

import numpy as np

import string

import re



#http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/   Important tutorial

#http://scikit-learn.org/stable/modules/cross_validation.html 

#https://learnpythonthehardway.org/book/ex32.html

#http://pbpython.com/pandas-list-dict.html



tweets_data_path = '../input/Tweets.csv'

pd.options.display.max_colwidth=150

tweets = pd.read_csv(tweets_data_path, header=0)

#df = tweets.copy()[['tweet_id', 'airline','text' , 'tweet_created','user_timezone','airline_sentiment']]

df = tweets.copy()[['airline', 'text','airline_sentiment']]



# print the data - chcek how It will look like 

#df.head(10)

df.tail(37)


# Define number of classes and number of tweets per class

n_class = 2

n_tweet = 2363



# Divide into number of classes

if n_class == 2:

    df_pos = df.copy()[df.airline_sentiment == 'positive'][:n_tweet]

    df_neg = df.copy()[df.airline_sentiment == 'negative'][:n_tweet]

    df_neu = pd.DataFrame()

    df = pd.concat([df_pos, df_neg], ignore_index=True).reset_index(drop=True)

elif n_class == 3:

    df_pos = df.copy()[df.airline_sentiment == 'positive'][:n_tweet]

    df_neg = df.copy()[df.airline_sentiment == 'negative'][:n_tweet]

    df_neu = df.copy()[df.airline_sentiment == 'neutral'][:n_tweet]

    df = pd.concat([df_pos, df_neg, df_neu], ignore_index=True).reset_index(drop=True)
# Define functions to process tweet text and remove stop words

def ProTweets(tweet):

    tweet = ''.join(c for c in tweet if c not in string.punctuation)

    tweet = re.sub('((www\S+)|(http\S+))', '', tweet)

    tweet = re.sub(r'\d+', '', tweet)

    tweet = re.sub(' +',' ', tweet)

    tweet = tweet.lower().strip()

    return tweet



def rmStopWords(tweet, stop_words):

    text = tweet.split()

    text = ' '.join(word for word in text if word not in stop_words)

    return text

# Get list of stop words

stop_words = get_stop_words('english')

stop_words = [''.join(c for c in s if c not in string.punctuation) for s in stop_words]

stop_words = [t.encode('utf-8') for t in stop_words]



# Preprocess all tweet data

pro_tweets = []

for tweet in df['text']:

    processed = ProTweets(tweet)

    pro_stopw = rmStopWords(processed, stop_words)

    pro_tweets.append(pro_stopw)



df['text'] = pro_tweets

df.head(10)
# Set up training and test sets by choosing random samples from classes

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['airline_sentiment'], test_size=0.33, random_state=0)



df_train = pd.DataFrame()

df_test = pd.DataFrame()



df_train['text'] = X_train

df_train['airline_sentiment'] = y_train

df_train = df_train.reset_index(drop=True)



df_test['text'] = X_test

df_test['airline_sentiment'] = y_test

df_test = df_test.reset_index(drop=True)


class MyTweetClassifier(object):



    def __init__(self, df_train):

        self.df_train = df_train

        self.df_pos = df_train.copy()[df_train.airline_sentiment == 'positive']

        self.df_neg = df_train.copy()[df_train.airline_sentiment == 'negative']

        self.df_neu = df_train.copy()[df_train.airline_sentiment == 'neutral']



    def fit(self):

        Pr_pos = df_pos.shape[0]/self.df_train.shape[0]

        Pr_neg = df_neg.shape[0]/self.df_train.shape[0]

        Pr_neu = df_neu.shape[0]/self.df_train.shape[0]

        self.Prior  = (Pr_pos, Pr_neg, Pr_neu)



        self.pos_words = ' '.join(self.df_pos['text'].tolist()).split()

        self.neg_words = ' '.join(self.df_neg['text'].tolist()).split()

        self.neu_words = ' '.join(self.df_neu['text'].tolist()).split()



        all_words = ' '.join(self.df_train['text'].tolist()).split()



        self.vocab = len(Counter(all_words))



        wc_pos = len(' '.join(self.df_pos['text'].tolist()).split())

        wc_neg = len(' '.join(self.df_neg['text'].tolist()).split())

        wc_neu = len(' '.join(self.df_neu['text'].tolist()).split())

        self.word_count = (wc_pos, wc_neg, wc_neu)

        return self



    def predict(self, df_test):

        class_choice = ['positive', 'negative', 'neutral']



        classification = []

        my_result_df = pd.DataFrame(columns=('Twitter Text', 'Sentiment Type'))

        pd.options.display.max_colwidth=150

                                 

        for tweet in df_test['text']:

            text = tweet.split()



            val_pos = np.array([])

            val_neg = np.array([])

            val_neu = np.array([])

            for word in text:

                tmp_pos = np.log(self.pos_words.count(word)+1)

                tmp_neg = np.log(self.neg_words.count(word)+1)

                tmp_neu = np.log(self.neu_words.count(word)+1)

                val_pos = np.append(val_pos, tmp_pos)

                val_neg = np.append(val_neg, tmp_neg)

                val_neu = np.append(val_neu, tmp_neu)



            denom_pos = len(text)*np.log(self.word_count[0]+self.vocab)

            denom_neg = len(text)*np.log(self.word_count[1]+self.vocab)

            denom_neu = len(text)*np.log(self.word_count[2]+self.vocab)



            val_pos = np.log(self.Prior[0]) + np.sum(val_pos) - denom_pos

            val_neg = np.log(self.Prior[1]) + np.sum(val_neg) - denom_neg

            val_neu = np.log(self.Prior[2]) + np.sum(val_neu) - denom_neu



            probability = (val_pos, val_neg, val_neu)

            

            arg_max_value= class_choice[np.argmax(probability)]

            classification.append(arg_max_value)

            data = [(tweet,arg_max_value)]

            my_result_df=pd.DataFrame(data, columns=['Twitter Text', 'Sentiment Type']).append(my_result_df, ignore_index=True)

        #return classification

        return my_result_df         

            

 
senti1= 'Seats available but not allowed to purchase on Delta flight. Somehow the airport controls the seats'

senti2= 'Miraculously landed safely on the Delta flight from hell. I was pretty sure we were going to die almost the entire flight.'

senti3='A Delta flight attendant making jokes? Suddenly I thought I was on a Southwest flight.'

senti4='Delta Flight 428 JFK-PHX has been delayed 5-6 times with no info. What is going on?'

senti5='The gate agent on my Delta flight just announced that if you have a Samsung Note 7 you shouldnt use or charge it onboard.'

senti6='VirginAmerica you know what would be amazingly awesome? BOS-FLL PLEASE!!!!!!! I want to fly with only you.'

senti7='SouthwestAir THANK YOU. I left my iPad on a plane, filled out a lost and found form. Yall found it and shipped it back. Thank you '

senti8='southwestair your attendants at the ATL airport are awesome! Very helpful with all the Cancelled Flightlations this morning.'

senti9='southwestair  This is really fantastic travel'

senti10='Delta Air  this my last travel with you '
myList = [(senti1,''),(senti2,''),(senti3,''),(senti4,''),(senti5,''),(senti6,''),(senti7,''),(senti8,''),(senti9,''),(senti10,'')]

custome_test = pd.DataFrame.from_records(myList, columns=['text', 'airline_sentiment'])

pd.options.display.max_colwidth=150

custome_test.head(10)
# Run my sentiment  classifier

tnb = MyTweetClassifier(df_train)

tnb = tnb.fit()

predict = tnb.predict(custome_test)

predict.head(10)