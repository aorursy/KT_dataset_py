#installing contractions library

!pip -q install contractions
#Generic Data Processing & Visualization Libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re,string,unicodedata

import contractions #import contractions_dict

from bs4 import BeautifulSoup

%matplotlib inline





#Importing text processing libraries

import spacy

import spacy.cli

import nltk

from nltk.tokenize.toktok import ToktokTokenizer

from nltk.tokenize import word_tokenize

from nltk.stem.lancaster import LancasterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords



#downloading wordnet/punkt dictionary

nltk.download('wordnet')

nltk.download('punkt')

nltk.download('stopwords')



import warnings

warnings.filterwarnings("ignore")



pd.set_option('display.max_columns', 100)
#Loading Dataset

url = '../input/sentiment-analysis-in-energy-crude-oil/tweets.csv'

raw_data = pd.read_csv(url, header='infer')
raw_data.shape
#Creating a seperate dataset with specific columns.

data = raw_data[['text','screenName','retweetCount']]

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

data = data[['screenName','text','Stemd_text','retweetCount','word_count']]
#Taking Backup

data_preproc = data.copy()
#Import Textblob Library

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
plt.figure(figsize=(10,6))

sns.boxenplot(x='Sentiments', y='word_count', data=data)

plt.show();
plt.figure(figsize=(10,6))

sns.stripplot(x='Sentiments', y='retweetCount', data=data)

plt.show();
data[(data.Sentiments == 'negative') & (data.retweetCount > 35)]