import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
dataset = pd.read_csv('../input/sentiment-analysis-on-financial-tweets/stockerbot-export1.csv')
dataset.head()
dataset=dataset.drop('id',axis=1)
dataset.isnull().sum()
dataset['url']=dataset['url'].fillna('http://www.NULL.com')
plt.figure(figsize=(15,6))

dataset['source'].value_counts()[:10].plot(kind='barh',color=sns.color_palette('summer',30))

plt.title('Source with most number of tweets')
plt.figure(figsize=(15,6))

dataset['url'].value_counts()[:10].plot(kind='barh',color=sns.color_palette('summer',30))
plt.figure(figsize=(15,6))

dataset['company_names'].value_counts()[:30].plot(kind='bar',color=sns.color_palette('summer',30))
#com nltk

pat1= r'@[A-Za-z0-9]+'

pat2= r'https?://[A-Za-z0-9./]+'

combined_pat=r'|'.join((pat1,pat2))

pat3= r'[^a-zA-Z]'

combined_pat2=r'|'.join((combined_pat,pat3))
len(dataset['text'])
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps= PorterStemmer()

cleaned_tweets = []



for i in range(0,len(dataset['text'])):

    tweets = re.sub(combined_pat2,' ',dataset['text'][i])

    tweets=tweets.lower()

    tweets=tweets.split()

    tweets = [ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]

    tweets = ' '.join(tweets)

    cleaned_tweets.append(tweets)

cleaned_tweets[:10]
dataset.columns
dataset['cleaned_tweets'] = cleaned_tweets
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

for tweet in cleaned_tweets[:10]:

    print(tweet)

    s = sia.polarity_scores(tweet)

    for k in sorted(s):

        print('{0}: {1}, '.format(k, s[k]), end='')

        print()
def findpolarity(data):

    sid = SentimentIntensityAnalyzer()

    polarity = sid.polarity_scores(data)

    if(polarity['compound'] >= 0.2):  

        sentiment = 1

    if(polarity['compound'] <= -0.2):

        sentiment = -1 

    if(polarity['compound'] < 0.2 and polarity['compound'] >-0.2):

        sentiment = 0     

    return(sentiment)
findpolarity(cleaned_tweets[0])
sentiment = []

for i in range(0, len(cleaned_tweets)):

    s = findpolarity(cleaned_tweets[i])

    sentiment.append(s)
len(sentiment)
len(cleaned_tweets)
tweet_sentiment = pd.DataFrame()

tweet_sentiment['cleaned_tweets'] = cleaned_tweets

tweet_sentiment['sentiment'] = sentiment
tweet_sentiment.to_csv('tweet_sentiment.csv', index=False)
tweet_sentiment.shape[0]
positive_tweet = []

negative_tweet = []

neutral_tweet = []



for i in range(0, tweet_sentiment.shape[0]):

    if tweet_sentiment['sentiment'][i] == 0:

        neutral_tweet.append(tweet_sentiment['cleaned_tweets'][i])

    elif tweet_sentiment['sentiment'][i] == 1:

        positive_tweet.append(tweet_sentiment['cleaned_tweets'][i])

    elif tweet_sentiment['sentiment'][i] == -1:

        negative_tweet.append(tweet_sentiment['cleaned_tweets'][i])
negative_tweet[:10]
!pip install wordcloud
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))



    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(positive_tweet)

show_wordcloud(neutral_tweet)

show_wordcloud(negative_tweet)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(tweet_sentiment['cleaned_tweets']).toarray()

y = tweet_sentiment['sentiment']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

score = accuracy_score(y_test, y_pred)
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred, title="{} Confusion Matrix",

                normalize=True,figsize=(6,6),text_fontsize='large')

plt.show()
score