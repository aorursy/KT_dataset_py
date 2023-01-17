import pandas as pandasInstance

import numpy as numpyInstance

import matplotlib.pyplot as matplotlibInstance

import seaborn as seabornInstance
%matplotlib inline
tweets_Data = pandasInstance.read_csv('../input/Tweets.csv')
tweets_Data.head()
tweets_Data.info()
tweets_Data.describe()
matplotlibInstance.figure(figsize=(20,10))

matplotlibInstance.tight_layout()

seabornInstance.countplot(x='airline',data=tweets_Data)
groupedByTimeZone = tweets_Data.groupby('user_timezone')
numberOfTweetsFromDifferentTiemZones = groupedByTimeZone['user_timezone'].count()
topFiveTimeZonesWithMostTweets = numberOfTweetsFromDifferentTiemZones.sort_values(ascending=False).head(5)
topFiveTimeZonesWithMostTweets
tweets_Data['airline_sentiment_confidence'].hist(color='green',figsize=(20,10))
tweets_Data['Tweet Length'] = tweets_Data['text'].apply(len)
tweets_Data[tweets_Data['Tweet Length'] == 186]['text'].iloc[0]
tweets_Data.hist(column='Tweet Length',by='airline_sentiment',figsize=(20,10),color='red',bins=100)
import nltk

from nltk.corpus import stopwords
import string
tweets_text = tweets_Data['text']
def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
withoutStopWordsandPunctuations = tweets_Data['text'].apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(tweets_Data['text'])
print(len(bow_transformer.vocabulary_))
tweets_bow = bow_transformer.transform(tweets_Data['text'])
print('Shape of Sparse Matrix: ', tweets_bow.shape)

print('Amount of Non-Zero occurences: ', tweets_bow.nnz)
sparsity = (100.0 * tweets_bow.nnz / (tweets_bow.shape[0] * tweets_bow.shape[1]))

print('sparsity: {}'.format((sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer = TfidfTransformer().fit(tweets_bow)
tweets_tfidf = tfidf_transformer.transform(tweets_bow)

print(tweets_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(tweets_tfidf, tweets_Data['airline_sentiment'])
all_predictions = spam_detect_model.predict(tweets_tfidf)

print(all_predictions)
from sklearn.metrics import classification_report

print (classification_report(tweets_Data['airline_sentiment'], all_predictions))