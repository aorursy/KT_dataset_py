# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind') #use this visual style for plots
!pip install seaborn #install this module
import seaborn as sns #import this library with this abbreviation
!pip install --upgrade pip
!pip install spacy
import spacy
sp = spacy.load('en_core_web_sm')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.feature_extraction import text #text processing
from sklearn.feature_extraction.text import CountVectorizer #spliting and numeric representation (Bag-of-words/n-grams)
from sklearn.feature_extraction.text import TfidfTransformer #calculating word importance score (TF/IDF)
#check out the data 
df=pd.read_csv("/kaggle/input/hcde530/comments20200605.csv")
df.head() #load the first 5 rows of the D&S dataframe using this method
print(df.shape) #show me the shape of this dataframe 
print(df.columns)
#convert epoch time to python timestamp format
import datetime #import this module 
df['published'] = df['comment_publish_date'].apply(lambda x: datetime.datetime.fromtimestamp(x)) #update the argument from pd.datetime to datetime.datetime per datetime documentation
#c.head() #check the output
#remove unnecessary columns

c=df.drop(['video_id','commenter_channel_url','comment_publish_date', 'commenter_rating', 'comment_id','commenter_channel_id','commenter_channel_display_name','comment_parent_id','collection_date',], axis=1)
print(c.shape)
c.head()
#clean up newlines from text column
cleaned = c.replace(r'\\n',' ', regex=True) 
cleaned = c.replace('\n','', regex=True)
#expand how much of the text column I can read 
pd.set_option('display.max_colwidth', None)
print(cleaned.text[50:60])
#code from Rafal 
import nltk
from nltk.stem.porter import *
from nltk import word_tokenize

#Create own tokenization (method of splitting text into individual word tokens)
#Problem 3 - Fix typos by replacing words, e.g represent 'the best' and 'good' the same way
class LemmaTokenizer:
    def __init__(self):
        self.sp = spacy.load('en_core_web_sm') #load english language data
    def __call__(self, doc):
        replacements = {'goood': 'good'} #Problem 3: replace specific tokens, e.g., common typos
        tokens = self.sp(doc) #tokenize the document (split into words) - doc is one sentence
        return [replacements.get(t.lemma_,t.lemma_) for t in tokens] #replace some tokens

class PorterTokenizer: 
    def __init__(self):
        self.stemmer = PorterStemmer() #Create porter tokenizer
    def __call__(self, doc):
        replacements = {'goood': 'good'} #Problem 3: replace specific tokens, e.g., common typos
        return [replacements.get(self.stemmer.stem(t), self.stemmer.stem(t)) for t in word_tokenize(doc)]  
#Converting words to word ids using Scikit-learn CountVectorizer
count_vect = CountVectorizer( 
    tokenizer=LemmaTokenizer(),
    ngram_range=(1,2)
)#create the CountVectorizer object
count_vect.fit(cleaned['text'].values) #fit into our dataset

#Get a list of unique words found in the document (vocabulary)
word_list = count_vect.get_feature_names()
print(len(word_list))

#Check all the words that were extracted and their ids:
for word_id, word in enumerate(word_list):
    print(str(word_id)+" -> "+str(word_list[word_id])) #Show ID and word

#Transform our dataset from words to ids
word_counts = count_vect.transform(df['text'].values) #Transform the text into numbers (bag of words features)
print("\nSize (sentenes x words):", word_counts.shape) #Display the size of our array (list of lists)
print("Representation of our sentences as an array of ids:")
#Check how text sentences from our data were replaced by numbers
print(word_counts.toarray()) #See what it looks like