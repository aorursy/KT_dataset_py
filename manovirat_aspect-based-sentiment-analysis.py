#general imports

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import seaborn as sns # plotting

import matplotlib.pyplot as plt # plotting

%matplotlib inline

import os # accessing directory structure



#NLP processing imports

from wordcloud import WordCloud

import nltk

from nltk.corpus import stopwords

from nltk import sent_tokenize, word_tokenize

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.tokenize import RegexpTokenizer

from stop_words import get_stop_words

import re

import spacy



###Vader Sentiment

#To install vaderSentiment

!pip install vaderSentiment 

from vaderSentiment import vaderSentiment

from textblob import TextBlob

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



####Lemmatization

from nltk.stem import WordNetLemmatizer

# Lemmatize with POS Tag

from nltk.corpus import wordnet
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/twcs.csv")
data.shape
data = data.loc[:10000]
data.shape
data.head()
pd.set_option('display.max_colwidth', -1)
data.head(10)
nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns')
#DataTypes

data.dtypes
data["text"] = data["text"].astype(str)
def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+|@[^\s]+')

    return url_pattern.sub(r'', text)
data["textclean"] = data["text"].apply(lambda text: remove_urls(text))
data.head()
top_N = 100 #top 100 words



#convert list of list into text

a = data['textclean'].str.lower().str.cat(sep=' ')



# removes punctuation,numbers and returns list of words

b = re.sub('[^A-Za-z]+', ' ', a)
#remove all the stopwords from the text

stop_words = list(get_stop_words('en'))         

nltk_words = list(stopwords.words('english'))   

stop_words.extend(nltk_words)
word_tokens = word_tokenize(b) # Tokenization

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:

    if w not in stop_words:

        filtered_sentence.append(w)
# Remove characters which have length less than 2  

without_single_chr = [word for word in filtered_sentence if len(word) > 2]



# Remove numbers

cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



# 1. Init Lemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_output = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in cleaned_data_title]

lemmatized_output = [word for word in lemmatized_output if not word.isnumeric()]
word_dist = nltk.FreqDist(lemmatized_output)

top100_words = pd.DataFrame(word_dist.most_common(top_N),

                    columns=['Word', 'Frequency'])
plt.figure(figsize=(10,10))

sns.set_style("whitegrid")

ax = sns.barplot(x="Frequency",y="Word", data=top100_words.head(10))
def wc(data,bgcolor,title):

    plt.figure(figsize = (80,80))

    wc = WordCloud(background_color = bgcolor, max_words = 100,  max_font_size = 50)

    wc.generate(' '.join(data))

    plt.imshow(wc)

    plt.axis('off')
wc(lemmatized_output,'black','Common Words' )
sent_analyser = SentimentIntensityAnalyzer()

def sentiment(text):

    return (sent_analyser.polarity_scores(text)["compound"])
data["Polarity"] = data["textclean"].apply(sentiment)
data.head()
data.dtypes
def senti(data):

    if data['Polarity'] >= 0.05:

        val = "Positive"

    elif data['Polarity'] <= -0.05:

        val = "Negative"

    else:

        val = "Neutral"

    return val
data['Sentiment'] = data.apply(senti, axis=1)
plt.figure(figsize=(10,10))

sns.set_style("whitegrid")

ax = sns.countplot(x="Sentiment", data=data, 

                  palette=dict(Neutral="blue", Positive="Green", Negative="Red"))
#import spacy

nlp = spacy.load("en_core_web_sm")
def pos(text):

    doc = nlp(text)

    # You want list of Verb tokens 

    aspects = [token.text for token in doc if token.pos_ == "NOUN"]

    return aspects
data["Aspects"] = data["textclean"].apply(pos)
data.head()