# Loading packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from datetime import datetime
# Loading data:
tweets = pd.read_csv('https://transfer.sh/bJEJT/twcs.csv')
# Let's see what does the dataset contain
tweets.head()
# Pick only inbound tweets that aren't in reply to anything
first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
print('Found {} first inbound messages.'.format(len(first_inbound)))

# Merge in all tweets in response
tweet_data = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')
print("Found {} responses.".format(len(tweet_data)))

# Filter out cases where reply tweet isn't from company
tweet_data = tweet_data[tweet_data.inbound_y ^ True]

# Let's see what happened
print("Found {} responses from companies.".format(len(tweet_data)))
print("Tweets Preview:")
# Let's check what's inside our dataset
tweet_data.info() 
# Seems that the column 'in_response_to_tweet_id_x' includes only missing values, let's check how many of them do we have in the whole dataset
tweet_data.isnull().sum()
# Let's drop this column
tweet_data.drop(['in_response_to_tweet_id_x'], axis='columns', inplace=True)
#Changing timestamp format
tweet_data['outbound_time_'] = pd.to_datetime(tweet_data['created_at_x'], format='%a %b %d %H:%M:%S +0000 %Y')
tweet_data['inbound_time'] = pd.to_datetime(tweet_data['created_at_y'], format='%a %b %d %H:%M:%S +0000 %Y')

#Calculating time between between outbound response and inbound message
tweet_data['response_time'] = tweet_data['inbound_time'] - tweet_data['outbound_time_']

# And how does it look like after the changes:
tweet_data.head()
# Let's change the variables we'll be using for networks to floats, as they have numeric values
tweet_data['tweet_id_x'] = pd.to_numeric(tweet_data['tweet_id_x'])
tweet_data['tweet_id_y'] = pd.to_numeric(tweet_data['tweet_id_y'])
tweet_data['author_id_x'] = pd.to_numeric(tweet_data['author_id_x'])
#Making sure the data type is a timedelta/duration
print('from ' + str(tweet_data['response_time'].dtype))

#Making it easier to later do averages by converting to a float datatype
tweet_data['converted_time'] = tweet_data['response_time'].astype('timedelta64[s]') / 60

print('to ' + str(tweet_data['converted_time'].dtype))
# Getting the average response time per company.
tweet_data.groupby('author_id_y')['converted_time'].mean()
# Getting the average response time per company.
author_grouped = tweet_data.groupby('author_id_y')
# Let's see the top 20 support providers 
top_support_providers = set(author_grouped.agg('count')
                                .sort_values(['tweet_id_x'], ascending=[0])
                                .index[:20]
                                .values)
tweet_data \
    .loc[tweet_data.author_id_y.isin(top_support_providers)] \
    .groupby('author_id_y') \
    .tweet_id_x.count() \
    .sort_values() \
    .plot('barh', title='Top 20 Brands by Volume')
# To select rows whose column value is in an iterable array, which we'll define as array, you can use isin:
array = ['Delta', 'AmericanAir', 'SouthwestAir', 'British_Airways']
# tweet_data_airline will now be the new dataframe only containing the 4 biggest airlines in support
tweet_data_airline = tweet_data.loc[tweet_data['author_id_y'].isin(array)]
# Viewing the dataframe
tweet_data_airline.head()
# Getting the shape of the new defined df. 
tweet_data_airline.shape
#I saw it says 94 mins is the average time it takes for a response. This does not seem realistic.
#Focusing in on the airlines and taking out outliers.

# Delta Airline
delta = tweet_data_airline[tweet_data_airline['author_id_y'] == 'Delta']
delta_times = delta['converted_time']

delta_times.dropna()

def remove_outlier(delta_times):
    q1 = delta_times.quantile(0.25)
    q3 = delta_times.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = delta_times.loc[(delta_times > fence_low) & (delta_times < fence_high)]
    return df_out

no_outliers = remove_outlier(delta_times)

import matplotlib.pyplot as plt
hist_plot = no_outliers.plot.hist(bins=50)
hist_plot.set_title('Delta Support Response Time')
hist_plot.set_xlabel('Mins to Response')
hist_plot.set_ylabel('Frequency')
plt.show()

print('Delta\'s average response time is ' + str(round(no_outliers.mean(),2)) + ' minutes.' )

# AmericanAir
americanair = tweet_data_airline[tweet_data_airline['author_id_y'] == 'AmericanAir']
americanair_times = americanair['converted_time']

americanair_times.dropna()

def remove_outlier(americanair_times):
    q1 = americanair_times.quantile(0.25)
    q3 = americanair_times.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out1 = americanair_times.loc[(americanair_times > fence_low) & (americanair_times < fence_high)]
    return df_out1

no_outliers1 = remove_outlier(americanair_times)

import matplotlib.pyplot as plt
hist_plot = no_outliers1.plot.hist(bins=50)
hist_plot.set_title('AmericanAir Support Response Time')
hist_plot.set_xlabel('Mins to Response')
hist_plot.set_ylabel('Frequency')
plt.show()

print('AmericanAir\'s average response time is ' + str(round(no_outliers1.mean(),2)) + ' minutes.' )

# SouthwestAir
southwestair = tweet_data_airline[tweet_data_airline['author_id_y'] == 'SouthwestAir']
southwestair_times = southwestair['converted_time']

southwestair_times.dropna()

def remove_outlier(southwestair_times):
    q1 = southwestair_times.quantile(0.25)
    q3 = southwestair_times.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out2 = southwestair_times.loc[(southwestair_times > fence_low) & (southwestair_times < fence_high)]
    return df_out2

no_outliers2 = remove_outlier(southwestair_times)

import matplotlib.pyplot as plt
hist_plot = no_outliers2.plot.hist(bins=50)
hist_plot.set_title('SouthwestAir Support Response Time')
hist_plot.set_xlabel('Mins to Response')
hist_plot.set_ylabel('Frequency')
plt.show()

print('SouthwestAir\'s average response time is ' + str(round(no_outliers2.mean(),2)) + ' minutes.' )

# British_Airways
british_airways = tweet_data_airline[tweet_data_airline['author_id_y'] == 'British_Airways']
british_airways_times = british_airways['converted_time']

british_airways_times.dropna()

def remove_outlier(british_airways_times):
    q1 = british_airways_times.quantile(0.25)
    q3 = british_airways_times.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out3 = british_airways_times.loc[(british_airways_times > fence_low) & (british_airways_times < fence_high)]
    return df_out3

no_outliers3 = remove_outlier(british_airways_times)

import matplotlib.pyplot as plt
hist_plot = no_outliers3.plot.hist(bins=50)
hist_plot.set_title('British Airways Support Response Time')
hist_plot.set_xlabel('Mins to Response')
hist_plot.set_ylabel('Frequency')
plt.show()

print('British Airways\'s average response time is ' + str(round(no_outliers3.mean(),2)) + ' minutes.' )

# Tokenizing sentences
from nltk.tokenize import sent_tokenize

# Tokenizing words
from nltk.tokenize import word_tokenize

# Tokenizing Tweets!
from nltk.tokenize import TweetTokenizer
!pip install gensim

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder #labelencoder for crosstab
import collections   # when calling Counter I would then use collections.Counter()
import collections as collect # collect.Counter
from collections import Counter
from gensim.corpora.dictionary import Dictionary #corpora dicionary for later analysis
from gensim.models.tfidfmodel import TfidfModel 
tknzr = TweetTokenizer(preserve_case=False)

# Parsing the tweets using the tweet column
text_tokenized = [tknzr.tokenize(text) for text in tweet_data_airline['text_x']]
# Lemmarize and remove stopwords
english_stopwords = stopwords.words('english')
english_stopwords.append('rt') #we also want to remove the retweet sign from the tweets. It's easyer just to add it to the list of english stopwords

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

tweet_data_airline['tokenized'] = [tknzr.tokenize(text) for text in tweet_data_airline['text_x']]
tweet_data_airline['tokenized'] = tweet_data_airline['tokenized'].map(lambda t: [word.lower().strip() for word in t if word.isalpha()])
tweet_data_airline['tokenized'] = tweet_data_airline['tokenized'].map(lambda t: [wordnet_lemmatizer.lemmatize(word) for word in t if word not in english_stopwords])
# Viewing the data
tweet_data_airline['tokenized'].head()
# Figuring out frequently used words
def tokenize_tweets_for_counter(tweets):
    """Get all of the tokens in a set of tweets"""
    twt = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False) #makes lowercase
    tokens = [token for tweet in tweets for token in twt.tokenize(tweet)]
    return(tokens)
# Figuring out frequently used words
my_temp_tweet_tokens = tokenize_tweets_for_counter(tweet_data_airline['text_x'])
from nltk.corpus import stopwords

# lowercasing
cleaned_word_tokenized = [word.lower().strip() for word in my_temp_tweet_tokens]
# replacing some unwanted things
cleaned_word_tokenized = [word.replace('(','').replace(')','') for word in cleaned_word_tokenized if word.isalpha()]
# removing stopwords
cleaned_word_tokenized = [word for word in cleaned_word_tokenized if word not in english_stopwords]
# removing RT

pos_count = Counter(cleaned_word_tokenized)
pos_count.most_common(10)
# We start by creating a Dictionary from the tweets
dictionary = Dictionary(tweet_data_airline['tokenized'])
# Create a Corpus: corpus
corpus = [dictionary.doc2bow(mytweet) for mytweet in tweet_data_airline['tokenized']]
# Create and fit a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]
# We import the model
from gensim.models.lsimodel import LsiModel

# And we fit it on the tfidf_corpus pointing to the dictionary as reference and the number of topics.
lsi = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
# Inspecting the topics
lsi.show_topics(num_topics=10)
# We can use the trained model to transform the corpus
lsi_corpus = lsi[tfidf_corpus]
# Load the MatrixSimilarity
from gensim.similarities import MatrixSimilarity

# Create the document-topic-matrix
document_topic_matrix = MatrixSimilarity(lsi_corpus)
document_topic_matrix = document_topic_matrix.index
pd.DataFrame(document_topic_matrix.dot(document_topic_matrix.T))
!pip install basemap
import nltk
nltk.download('vader_lexicon')
# We borrowed this code from: https://datascienceplus.com/twitter-analysis-with-python/?fbclid=IwAR3JrPUK6iaU8IOdD4-Sdsf3XTc4g71BhHU5y6xiv99dnhQKcvTNCcLu-Zw
import re
import warnings

#Visualisation

import matplotlib
import seaborn as sns
from IPython.display import display
#from mpl_toolkits.basemap import Basemap
#from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer

#nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
# Lemmatizing tweets/preprocessing them  
tweet_data_airline['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweet_data_airline['text_x']]       
vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)
X = vectorizer.fit_transform(tweet_data_airline['text_lem'].str.upper())
sid = SentimentIntensityAnalyzer()
tweet_data_airline['sentiment_compound_polarity']=tweet_data_airline.text_lem.apply(lambda x:sid.polarity_scores(x)['compound'])
tweet_data_airline['sentiment_neutral']=tweet_data_airline.text_lem.apply(lambda x:sid.polarity_scores(x)['neu'])
tweet_data_airline['sentiment_negative']=tweet_data_airline.text_lem.apply(lambda x:sid.polarity_scores(x)['neg'])
tweet_data_airline['sentiment_pos']=tweet_data_airline.text_lem.apply(lambda x:sid.polarity_scores(x)['pos'])
tweet_data_airline['sentiment_type']=''
tweet_data_airline.loc[tweet_data_airline.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweet_data_airline.loc[tweet_data_airline.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweet_data_airline.loc[tweet_data_airline.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets_sentiment = tweet_data_airline.groupby(['sentiment_type'])['sentiment_neutral'].count()
tweets_sentiment.rename("",inplace=True)
explode = (1, 0, 0)
plt.subplot(221)
tweets_sentiment.transpose().plot(kind='barh',figsize=(20, 20))
plt.title('Sentiment Analysis: All airlines 1', bbox={'facecolor':'0.8', 'pad':0})
plt.subplot(222)
tweets_sentiment.plot(kind='pie',figsize=(20, 20),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)
plt.title('Sentiment Analysis: All airlines 2', bbox={'facecolor':'0.8', 'pad':0})
plt.show()
# To select rows whose column value is in an iterable array, which we'll define as array, you can use isin:
array1 = ['Delta']
# tweet_data_airline will now be the new dataframe only containing delta airlines in support
tweet_data_delta = tweet_data_airline.loc[tweet_data_airline['author_id_y'].isin(array1)]

# To select rows whose column value is in an iterable array, which we'll define as array, you can use isin:
array2 = ['AmericanAir']
# tweet_data_airline will now be the new dataframe only containing american airlines in support
tweet_data_americanair = tweet_data_airline.loc[tweet_data_airline['author_id_y'].isin(array2)]

# To select rows whose column value is in an iterable array, which we'll define as array, you can use isin:
array3 = ['SouthwestAir']
# tweet_data_airline will now be the new dataframe only containing southwest airlines in support
tweet_data_southwestair = tweet_data_airline.loc[tweet_data_airline['author_id_y'].isin(array3)]

# To select rows whose column value is in an iterable array, which we'll define as array, you can use isin:
array4 = ['British_Airways']
# tweet_data_airline will now be the new dataframe only containing british airways in support
tweet_data_britishairways = tweet_data_airline.loc[tweet_data_airline['author_id_y'].isin(array4)]
tweet_data_delta['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweet_data_delta['text_x']]       
vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)
X = vectorizer.fit_transform(tweet_data_delta['text_lem'].str.upper())
sid = SentimentIntensityAnalyzer()
tweet_data_delta['sentiment_compound_polarity']=tweet_data_delta.text_lem.apply(lambda x:sid.polarity_scores(x)['compound'])
tweet_data_delta['sentiment_neutral']=tweet_data_delta.text_lem.apply(lambda x:sid.polarity_scores(x)['neu'])
tweet_data_delta['sentiment_negative']=tweet_data_delta.text_lem.apply(lambda x:sid.polarity_scores(x)['neg'])
tweet_data_delta['sentiment_pos']=tweet_data_delta.text_lem.apply(lambda x:sid.polarity_scores(x)['pos'])
tweet_data_delta['sentiment_type']=''
tweet_data_delta.loc[tweet_data_delta.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweet_data_delta.loc[tweet_data_delta.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweet_data_delta.loc[tweet_data_delta.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets_sentiment1 = tweet_data_delta.groupby(['sentiment_type'])['sentiment_neutral'].count()
tweets_sentiment1.rename("",inplace=True)
explode = (1, 0, 0)
plt.subplot(221)
tweets_sentiment1.transpose().plot(kind='barh',figsize=(20, 20))
plt.title('Sentiment Analysis 1 for Delta', bbox={'facecolor':'0.8', 'pad':0})
plt.subplot(222)
tweets_sentiment1.plot(kind='pie',figsize=(20, 20),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)
plt.title('Sentiment Analysis 2 for Delta', bbox={'facecolor':'0.8', 'pad':0})
plt.show()
tweet_data_americanair['text_lem1'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweet_data_americanair['text_x']]       
vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)
X = vectorizer.fit_transform(tweet_data_americanair['text_lem1'].str.upper())
sid = SentimentIntensityAnalyzer()
tweet_data_americanair['sentiment_compound_polarity']=tweet_data_americanair.text_lem1.apply(lambda x:sid.polarity_scores(x)['compound'])
tweet_data_americanair['sentiment_neutral']=tweet_data_americanair.text_lem1.apply(lambda x:sid.polarity_scores(x)['neu'])
tweet_data_americanair['sentiment_negative']=tweet_data_americanair.text_lem1.apply(lambda x:sid.polarity_scores(x)['neg'])
tweet_data_americanair['sentiment_pos']=tweet_data_americanair.text_lem1.apply(lambda x:sid.polarity_scores(x)['pos'])
tweet_data_americanair['sentiment_type']=''
tweet_data_americanair.loc[tweet_data_americanair.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweet_data_americanair.loc[tweet_data_americanair.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweet_data_americanair.loc[tweet_data_americanair.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets_sentiment1 = tweet_data_americanair.groupby(['sentiment_type'])['sentiment_neutral'].count()
tweets_sentiment1.rename("",inplace=True)
explode = (1, 0, 0)
plt.subplot(221)
tweets_sentiment1.transpose().plot(kind='barh',figsize=(20, 20))
plt.title('Sentiment Analysis 1 for AmericanAir', bbox={'facecolor':'0.8', 'pad':0})
plt.subplot(222)
tweets_sentiment1.plot(kind='pie',figsize=(20, 20),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)
plt.title('Sentiment Analysis 2 for AmericanAir', bbox={'facecolor':'0.8', 'pad':0})
plt.show()
tweet_data_southwestair['text_lem3'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweet_data_southwestair['text_x']]       
vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)
X = vectorizer.fit_transform(tweet_data_southwestair['text_lem3'].str.upper())
sid = SentimentIntensityAnalyzer()
tweet_data_southwestair['sentiment_compound_polarity']=tweet_data_southwestair.text_lem3.apply(lambda x:sid.polarity_scores(x)['compound'])
tweet_data_southwestair['sentiment_neutral']=tweet_data_southwestair.text_lem3.apply(lambda x:sid.polarity_scores(x)['neu'])
tweet_data_southwestair['sentiment_negative']=tweet_data_southwestair.text_lem3.apply(lambda x:sid.polarity_scores(x)['neg'])
tweet_data_southwestair['sentiment_pos']=tweet_data_southwestair.text_lem3.apply(lambda x:sid.polarity_scores(x)['pos'])
tweet_data_southwestair['sentiment_type']=''
tweet_data_southwestair.loc[tweet_data_southwestair.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweet_data_southwestair.loc[tweet_data_southwestair.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweet_data_southwestair.loc[tweet_data_southwestair.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets_sentiment3 = tweet_data_southwestair.groupby(['sentiment_type'])['sentiment_neutral'].count()
tweets_sentiment3.rename("",inplace=True)
explode = (1, 0, 0)
plt.subplot(221)
tweets_sentiment3.transpose().plot(kind='barh',figsize=(20, 20))
plt.title('Sentiment Analysis 1 for SouthwestAir', bbox={'facecolor':'0.8', 'pad':0})
plt.subplot(222)
tweets_sentiment3.plot(kind='pie',figsize=(20, 20),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)
plt.title('Sentiment Analysis 2 for SouthwestAir', bbox={'facecolor':'0.8', 'pad':0})
plt.show()
tweet_data_britishairways['text_lem2'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweet_data_britishairways['text_x']]       
vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)
X = vectorizer.fit_transform(tweet_data_britishairways['text_lem2'].str.upper())
sid = SentimentIntensityAnalyzer()
tweet_data_britishairways['sentiment_compound_polarity']=tweet_data_britishairways.text_lem2.apply(lambda x:sid.polarity_scores(x)['compound'])
tweet_data_britishairways['sentiment_neutral']=tweet_data_britishairways.text_lem2.apply(lambda x:sid.polarity_scores(x)['neu'])
tweet_data_britishairways['sentiment_negative']=tweet_data_britishairways.text_lem2.apply(lambda x:sid.polarity_scores(x)['neg'])
tweet_data_britishairways['sentiment_pos']=tweet_data_britishairways.text_lem2.apply(lambda x:sid.polarity_scores(x)['pos'])
tweet_data_britishairways['sentiment_type']=''
tweet_data_britishairways.loc[tweet_data_britishairways.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweet_data_britishairways.loc[tweet_data_britishairways.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweet_data_britishairways.loc[tweet_data_britishairways.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets_sentiment2 = tweet_data_britishairways.groupby(['sentiment_type'])['sentiment_neutral'].count()
tweets_sentiment2.rename("",inplace=True)
explode = (1, 0, 0)
plt.subplot(221)
tweets_sentiment2.transpose().plot(kind='barh',figsize=(20, 20))
plt.title('Sentiment Analysis 1 for British Airways', bbox={'facecolor':'0.8', 'pad':0})
plt.subplot(222)
tweets_sentiment2.plot(kind='pie',figsize=(20, 20),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)
plt.title('Sentiment Analysis 2 for British Airways', bbox={'facecolor':'0.8', 'pad':0})
plt.show()
# Grouping tweets by class to see the amoung of each class denotation
t_class = tweet_data_airline.groupby(['sentiment_type'], as_index=False).count()
# Print results of t_class
print(t_class)
# Creating a array to save results
negative_tweets = []

# Take negative tweets and tokenized words and add it to our negative_tweets
for x in tweet_data_airline[tweet_data_airline['sentiment_type'] == 'NEGATIVE']['tokenized']:
    negative_tweets.extend(x)
    
negative_tweets_tfidf = tfidf[dictionary.doc2bow(negative_tweets)]

negative_tweets_tfidf = sorted(negative_tweets_tfidf, key=lambda w: w[1], reverse=True)

# Print the top 10 weighted words
for term_id, weight in negative_tweets_tfidf[:10]:
    print(dictionary.get(term_id), weight)
positive_tweets = []

for x in tweet_data_airline[tweet_data_airline['sentiment_type'] == 'POSITIVE']['tokenized']:
    positive_tweets.extend(x)
    
positive_tweets_tfidf = tfidf[dictionary.doc2bow(positive_tweets)]

positive_tweets_tfidf = sorted(positive_tweets_tfidf, key=lambda w: w[1], reverse=True)

# Print the top 10 weighted words
for term_id, weight in positive_tweets_tfidf[:10]:
    print(dictionary.get(term_id), weight)
neutral_tweets = []

for x in tweet_data_airline[tweet_data_airline['sentiment_type'] == 'NEUTRAL']['tokenized']:
    negative_tweets.extend(x)
    
neutral_tweets_tfidf = tfidf[dictionary.doc2bow(negative_tweets)]

neutral_tweets_tfidf = sorted(negative_tweets_tfidf, key=lambda w: w[1], reverse=True)

# Print the top 10 weighted words
for term_id, weight in neutral_tweets_tfidf[:10]:
    print(dictionary.get(term_id), weight)
# Let's recode our labels to numeric values
mapping = {'NEGATIVE': 0, 'POSITIVE': 1, 'NEUTRAL': 2}
# We use our mapping to change the value in our dataframe/column
tweet_data_airline['sentiment_type'] = tweet_data_airline['sentiment_type'].map(mapping)
# Let's make a quick view of change is complete before moving on to ML
tweet_data_airline.sentiment_type.unique()
# Creating a list to save results from the three classifier models
classifier_results = [0,0,0]
# define y and X 
y = tweet_data_airline['sentiment_type']

X = document_topic_matrix

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
# Let's fit a simple linear regression model

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5)

print("Score on train set " + str(accuracies.mean()))
print("Std deviation " + str(accuracies.std()))
classifier_results[0] = regressor.score(X_test, y_test)
print("Score on test set " + str(classifier_results[0]))
from sklearn.preprocessing import LabelEncoder #labelencoder for crosstab
# Encode the labels to numbers
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Creating a pandas DataFrame and cross-tabulation
real_tweets = labelencoder_y.inverse_transform(y_test)
predicted_tweets = labelencoder_y.inverse_transform(y_pred)
df = pd.DataFrame({'real_tweets': real_tweets, 'predicted_tweets': predicted_tweets}) 
# Let's create a crosstab with results comparing classified tweets to train set
pd.crosstab(df.real_tweets, df.predicted_tweets)
# Let's see how does it look like with accuracies for each class
from sklearn.metrics import classification_report
y_true = df.real_tweets
y_pred = df.predicted_tweets
target_names = ['Negative', 'Positive', 'Neutral']
print(classification_report(y_true, y_pred, target_names=target_names))
# Let's fit a simple linear gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier
regressor = GradientBoostingClassifier()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5)

print("Score on train set " + str(accuracies.mean()))
print("Std deviation " + str(accuracies.std()))
classifier_results[1] = regressor.score(X_test, y_test)
print("Score on test set " + str(classifier_results[1]))
# Encode the labels to numbers
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Creating a pandas DataFrame and cross-tabulation
real_tweets = labelencoder_y.inverse_transform(y_test)
predicted_tweets = labelencoder_y.inverse_transform(y_pred)
df = pd.DataFrame({'real_tweets': real_tweets, 'predicted_tweets': predicted_tweets}) 
pd.crosstab(df.real_tweets, df.predicted_tweets)
# Let's see how does it look like with accuracies for each class
from sklearn.metrics import classification_report
y_true = df.real_tweets
y_pred = df.predicted_tweets
target_names = ['Negative', 'Positive', 'Neutral']
print(classification_report(y_true, y_pred, target_names=target_names))
#  Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 22)
classifier.fit(X_train, y_train)

# Predicting the test set results from the test-set inputs
y_pred = classifier.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)

print("Score on train set " + str(accuracies.mean()))
print("Std deviation " + str(accuracies.std()))
classifier_results[2] = regressor.score(X_test, y_test)
print("Score on test set " + str(classifier_results[2]))
# Encode the labels to numbers
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Creating a pandas DataFrame and cross-tabulation
real_tweets = labelencoder_y.inverse_transform(y_test)
predicted_tweets = labelencoder_y.inverse_transform(y_pred)
df = pd.DataFrame({'real_tweets': real_tweets, 'predicted_tweets': predicted_tweets}) 
pd.crosstab(df.real_tweets, df.predicted_tweets)
# Let's see how does it look like with accuracies for each class
from sklearn.metrics import classification_report
y_true = df.real_tweets
y_pred = df.predicted_tweets
target_names = ['Negative', 'Positive', 'Neutral']
print(classification_report(y_true, y_pred, target_names=target_names))
final_data = pd.DataFrame({'cat': ['Logistic Regresion', 'Gradient Boosting', 'Random Forest'], 'val': [classifier_results[0], classifier_results[1], classifier_results[2]]})
ax = sns.barplot(x = 'val', y = 'cat', 
              data = final_data)
ax.set(xlabel='Performance of classifying models in procentage', ylabel='models')
plt.show()
import networkx as nx
import itertools
import copy
edgelist = tweet_data_airline[['tweet_id_x','tweet_id_y']]
nodelist = tweet_data_airline[['tweet_id_y',	'author_id_x']]
# Creating empty graph
g = nx.Graph()
# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1])
# Let's check what's inside edge list
elrow.head()
# Adding node attributes
for i, nlrow in nodelist.iterrows():
      g.node[nlrow['tweet_id_y']].update(nlrow[1:].to_dict())
# Node list example
print(nlrow)
# Let's check the number of nodes and edges
print('number of edges: {}'.format(g.number_of_edges()))
print('number of nodes: {}'.format(g.number_of_nodes()))
# Let's download our modified dataset:
from google.colab import files

tweet_data_airline.to_csv('airlines.csv')
files.download('airlines.csv')