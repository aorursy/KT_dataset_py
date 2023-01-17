import numpy as np 
import pandas as pd
import re  # for removing unnecessary tags
import nltk
from bs4 import BeautifulSoup # for removing the HTML tags
from nltk.corpus import stopwords # for removing stopwords
from nltk.tokenize import word_tokenize

pd.set_option("display.max_colwidth", 500)
train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = '\t')
train.head()
train.sentiment.value_counts()
##cleaning of data for sentiment analysis

# Removing Stop words from Reviews
stop_words = set(stopwords.words('english'))

# LEMMATIZING
from nltk import WordNetLemmatizer
lemma = WordNetLemmatizer()


def cleaning(line):
    
    #Removing unecessary html tags
    soup = BeautifulSoup(line,"lxml")
    line = soup.get_text()
    
    # Removing punctuations and special characters
    line = re.sub(r"[^\w\s]","",line)
    
    #tokenizing of the review
    tokens = word_tokenize(line)
    
    #eliminating the stop words
    tokens = [x for x in tokens if x not in stop_words]
    
    #lemmatizing the tokens
    tokens = [lemma.lemmatize(x, pos = 'v') for x in tokens]
    
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    line = ' '.join(tokens)
    return line


train["review_new"]=train.review.map(cleaning)
train.head()