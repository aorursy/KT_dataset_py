#Libraries
!pip install tweepy
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import pandas as pd
import csv
import re #regular expression
from textblob import TextBlob
import string
import preprocessing as p
import os
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re,string,unicodedata
!pip install contractions
import contractions #import contractions_dict
from bs4 import BeautifulSoup
%matplotlib inline


#Importing text processing libraries
import spacy
import spacy.cli
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#downloading wordnet/punkt dictionary
nltk.download('wordnet')

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 100)
!pip install preprocessor
!pip install tweet-preprocessor
from preprocessor.api import clean
#Twitter credentials 
consumer_key = '*****'
consumer_secret = '****'
access_key= '***-****'
access_secret = '*******'
#pass twitter credentials to tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
#file location
Crude_Oil_Twitter_Analysis = "/Users/dhruvarora/Downloads/Twitter Analysis - Oil/TwitterData-US Oil Prices.csv"
COLS = ['id', 'created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang',
'favorite_count', 'retweet_count', 'original_author',   'possibly_sensitive', 'hashtags',
'user_mentions', 'place', 'place_coord_boundaries']
#set two date variables for date range
start_date = '2020-01-01'
end_date = '2020-01-14'
#HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)
#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)
#Clean tweet
def clean_tweets(tweet):
 
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
#after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
#replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
#remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
#filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
#looping through conditions
    for w in word_tokens:
#check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
    #print(word_tokens)
    #print(filtered_sentence)return tweet
def write_tweets(keyword, file):
    # If the file exists, then read the existing data from the CSV file.
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.DataFrame(columns=COLS)
    #page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search, q=keyword,
                              count=1000, include_rts=False, since=start_date).pages(200):
        for status in page:
            new_entry = []
            status = status._json

            ## check whether the tweet is in english or skip to the next tweet
            if status['lang'] != 'en':
                continue

            #when run the code, below code replaces the retweet amount and
            #no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                   status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue


           #tweepy preprocessing called for basic preprocessing
            clean_text = clean(status['text'])

            #call clean_tweet method for extra preprocessing
            filtered_tweet=clean_tweets(clean_text)

            #pass textBlob method for sentiment calculations
            blob = TextBlob(filtered_tweet)
            Sentiment = blob.sentiment

            #seperate polarity and subjectivity in to two variables
            polarity = Sentiment.polarity
            subjectivity = Sentiment.subjectivity

            #new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['text'],filtered_tweet, Sentiment,polarity,subjectivity, status['lang'],
                          status['favorite_count'], status['retweet_count']]

            #to append original author of the tweet
            new_entry.append(status['user']['screen_name'])

            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)

            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)

            #get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)

            try:
                coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            new_entry.append(coordinates)

            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a' ,encoding='utf-8')
    df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")
#declare keywords as a query for three categories
CrudeOil_keywords = 'US Oil Prices'
#write_tweets(CrudeOil_keywords,  Crude_Oil_Twitter_Analysis)#
#Loading Dataset
url = '../input/twitterdataus-oil-pricescsv/TwitterData-US Oil Prices.csv'
raw_data = pd.read_csv(url, header='infer')
data = raw_data[['original_text','original_author','retweet_count','polarity','subjectivity']]
data = data.rename(columns={'original_text': 'text', 'original_author': 'screenName', 'retweet_count':'retweetCount'})
#Resetting Index
data.reset_index(drop=True, inplace=True)
#Backup of the newly created dataset
data_backup = data.copy()
#lowering cases
data['text'] = data['text'].str.lower()
#stripping leading spaces (if any)
data['text'] = data['text'].str.strip()
# Removing HTML tags
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

#apply to the dataset
data['text'] = data['text'].apply(strip_html_tags)
# Remove URL and links
def strip_url(text):
    strip_url_text = re.sub(r'http\S+', '', text)
    return strip_url_text

#Applying the dataset
data['text'] = data['text'].apply(strip_url)


#removing punctuations
from string import punctuation

def remove_punct(text):
    for punctuations in punctuation:
        text = text.replace(punctuations, '')
        return text

#apply to the dataset
data['text'] = data['text'].apply(remove_punct)
#function to remove special characters
def remove_special_chars(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

#applying the function on the clean dataset
data['text'] = data['text'].apply(remove_special_chars)
#function to remove macrons & accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#applying the function on the clean dataset
data['text'] = data['text'].apply(remove_accented_chars) 
#Function to expand contractions
def expand_contractions(con_text):
    con_text = contractions.fix(con_text)
    return con_text

#applying the function on the clean dataset
data['text'] = data['text'].apply(expand_contractions)
#creating a new column in the dataset for word count
data ['word_count'] = data['text'].apply(lambda x:len(str(x).split(" ")))
#Taking Backup
data_clean = data.copy()
#function to remove stopwords
def remove_stopwords(text, is_lower_case=False):
    stopword_list = set(stopwords.words('english'))
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#applying the function
data ['text'] = data['text'].apply(remove_stopwords) 
#Function for stemming
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

#applying the function
data['Stemd_text'] = data['text'].apply(simple_stemmer)
#rearranging columns
data = data[['screenName','text','Stemd_text','retweetCount','word_count','polarity','subjectivity']]
#Taking Backup
data_preproc = data.copy()
from textblob import TextBlob
#function to perform Textblob Sentiment Analyis
def sentiment_analysis(text):
    polarity = round(TextBlob(text).sentiment.polarity, 3)
    sentiment_categories = ['positive','negative','neutral']
    if polarity > 0:
        return sentiment_categories[0]
    elif polarity < 0:
        return sentiment_categories[1]
    else:
        return sentiment_categories[2]  
        
#Apply to the Stemd_Text
data['Sentiments'] = [sentiment_analysis(txt) for txt in data['Stemd_text']]
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(data.word_count, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Word Count')
plt.ylabel('Tweet Count')
plt.title('Histogram of Word Count')
plt.show();
#Creating a Count Plot
sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize=(8,8))
ax = sns.countplot(x="Sentiments", data=data)
plt.title('Sentiments Count')
plt.ylabel('Count')
plt.xlabel('Sentiments')
data.head()
#Displaying Negative Tweets with a high retweet count
data[(data.Sentiments == 'negative') & (data.retweetCount > 40)]
