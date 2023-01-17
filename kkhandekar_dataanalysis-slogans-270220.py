#installing contractions library

!pip install contractions
#Generic Data Processing & Visualization Libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re,string,unicodedata

import contractions #import contractions_dict

%matplotlib inline
#Importing text processing libraries

import spacy

import spacy.cli

#from spacy.matcher import Matcher

#from spacy.tokens import Span



import nltk

import collections

from nltk.tokenize.toktok import ToktokTokenizer

from nltk.tokenize import word_tokenize

from nltk.stem.lancaster import LancasterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.util import ngrams



#downloading wordnet/punkt dictionary

nltk.download('wordnet')

nltk.download('punkt')

nltk.download('stopwords')
#Loading Data

data = pd.read_csv('../input/slogan-dataset/sloganlist.csv', header='infer')
data.shape
#checking for null / missing values

data.isna().sum()
#function to count words

def word_count(x):

  count = len(str(x).split(" "))

  return count



#data ['word_count'] = data['Slogan'].apply(lambda x:len(str(x).split(" ")))

data ['word_count'] = data['Slogan'].apply(word_count)
#lowering cases

data['Slogan'] = data['Slogan'].str.lower()



#stripping leading spaces (if any)

data['Slogan'] = data['Slogan'].str.strip()
#removing punctuations

from string import punctuation



def remove_punct(text):

  for punctuations in punctuation:

    text = text.replace(punctuations, '')

  return text



#apply to the dataset

data['Slogan'] = data['Slogan'].apply(remove_punct)
#function to remove special characters

def remove_special_chars(text, remove_digits=True):

  pattern = r'[^a-zA-z0-9\s]'

  text = re.sub(pattern, '', text)

  return text



#applying the function on the clean dataset

data['Slogan'] = data['Slogan'].apply(remove_special_chars)
#function to remove macrons & accented characters

def remove_accented_chars(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text



#applying the function on the clean dataset

data['Slogan'] = data['Slogan'].apply(remove_accented_chars)  
#Function to expand contractions

def expand_contractions(con_text):

  con_text = contractions.fix(con_text)

  return con_text



#applying the function on the clean dataset

data['Slogan'] = data['Slogan'].apply(expand_contractions) 
#Dataset Backup

data_backup = data.copy()
stopword_list = set(stopwords.words('english'))

tokenizer = ToktokTokenizer()
#function to remove stopwords

def remove_stopwords(text, is_lower_case=False):

    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    if is_lower_case:

        filtered_tokens = [token for token in tokens if token not in stopword_list]

    else:

        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)    

    return filtered_text



#applying the function

data['Slogan_Norm'] = data['Slogan'].apply(remove_stopwords)  
data ['word_count_norm'] = data['Slogan_Norm'].apply(word_count)
data.head()
plt.style.use('seaborn-deep')

plt.figure(figsize=(10,10))

plt.grid(True)

x = data['word_count']

y = data['word_count_norm']

plt.hist([x,y], label=['Word Count Distribution','Normalized Word Count Distribution'])

plt.legend(loc='upper right')

plt.title('Word Count Distribution')

plt.show()