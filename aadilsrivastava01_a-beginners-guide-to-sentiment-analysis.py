import numpy as np # linear algebra

import pandas as pd # data processing
df = pd.read_csv('../input/train.csv')

df.head(10)
train_pos = df[df['label'] == 0]

train_neg = df[df['label'] == 1]
def clean_word(data):

    words = " ".join(data['tweet'])

    

    cleaned_words = " ".join([word for word in words.split() 

                             if 'http' not in word

                             and not word.startswith('@')

                             and not word.startswith('#')

                             and word != 'RT'])

    return cleaned_words
pos_clean_words = clean_word(train_pos)

neg_clean_words = clean_word(train_neg)
import matplotlib.pyplot as plt

%matplotlib inline

from wordcloud import WordCloud,STOPWORDS
def wcloud(cleaned_words):

    wordcloud = WordCloud(stopwords=STOPWORDS,

                         background_color='black',

                         width=3000,

                          height=2500

                         ).generate(cleaned_words)

    return wordcloud
pos_wcloud = wcloud(pos_clean_words)

neg_wcloud = wcloud(neg_clean_words)
print('Non racist tweets')



plt.figure(1,figsize=(12,12))

plt.imshow(pos_wcloud)

plt.axis('off')

plt.show()
print('Racist tweets')



plt.figure(1,figsize=(12,12))

plt.imshow(neg_wcloud)

plt.axis('off')

plt.show()
df['label'].value_counts(normalize = True).plot.bar()
import re

import nltk

from nltk.corpus import stopwords
def clean_tweet_words(tweet):

    alpha_only = re.sub("[^a-zA-Z]",' ',tweet) #"[^a-zA-Z]" this regex will remove any non-alphabetical char as they are not significant

    words = alpha_only.lower().split()

    stop = set(stopwords.words('english'))

    #from the dataframe we can see 'user' word is quite common in the tweets, which is basically used for tagging someone in the tweet

    #so I will be removing that

    stop.add('user')

    sig_words = [word for word in words if not word in stop]

    return(" ".join(sig_words))
df['clean_tweet']  = df['tweet'].apply(lambda tweet: clean_tweet_words(tweet))



df.head(10)
from sklearn.model_selection import train_test_split



train,test = train_test_split(df,test_size = 0.2,random_state=0)
train_clean_tweet = []

for tweet in train['clean_tweet']:

    train_clean_tweet.append(tweet)

test_clean_tweet = []

for tweet in test['clean_tweet']:

    test_clean_tweet.append(tweet)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
svc_pipe = Pipeline([('tfidf',TfidfVectorizer()),('svc', LinearSVC(random_state=0,max_iter=5000))])

nb_pipe = Pipeline([('tfidf',TfidfVectorizer()),('nb', MultinomialNB())])
svc_pipe.fit(train_clean_tweet,train['label'])

nb_pipe.fit(train_clean_tweet,train['label'])
pred_svc = svc_pipe.predict(test_clean_tweet)

pred_nb = nb_pipe.predict(test_clean_tweet)
from sklearn.metrics import accuracy_score, confusion_matrix
print('SVC')

print(accuracy_score(test['label'],pred_svc))

print('\n')

print(confusion_matrix(test['label'],pred_svc))

print('\n')
print('Naive Bayes Classifier')

print(accuracy_score(test['label'],pred_nb))

print('\n')

print(confusion_matrix(test['label'],pred_nb))

print('\n')