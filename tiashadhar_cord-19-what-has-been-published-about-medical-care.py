# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Importing relevant libraries 



import numpy as np

import gensim

import os, pickle, nltk,re,json, math

import pandas as pd



import spacy

from spacy.matcher import Matcher



import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



from tqdm import tqdm



nlp = spacy.load("en_core_web_sm")

## Browsing the data directory



print(os.listdir("../input/CORD-19-research-challenge"))
## Reading the Metadata csv

## Metadata consists of all the possible pdf information



metadata = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")

metadata.head()
## Keeping relevant columns for deep dive 



df2 = metadata.drop(columns = ['sha', 'source_x', 'pmcid', 'license', 'Microsoft Academic Paper ID', 'WHO #Covidence'])

df2.head()



# Step 1 - Finding out those docs which are relevant to our Task from Abstract & Title



## Keeping data only which contains an abstract

df3 = df2.dropna(subset=['abstract'])

df3.head()

# Additional required libraraies



from spacy.tokenizer import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from collections import Counter

nltk.download('stopwords')

import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

import re

import string



#CountVectorizer() for counting word tokens and creating document term matrix needed for Gensim. Also require the text package to modify 

#stopwords according to what we see in the EDA



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import text
# Function for getting the all .json paths



def getListOfFiles(dirName):

    # create a list of file and sub directories 

    # names in the given directory 

    listOfFile = os.listdir(dirName)

    allFiles = list()

    # Iterate over all the entries

    for entry in listOfFile:

        # Create full path

        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory 

        if os.path.isdir(fullPath):

            allFiles = allFiles + getListOfFiles(fullPath)

        else:

            if fullPath[-5:] == '.json': allFiles.append(fullPath)

                

    return allFiles



#getting all the paths into a list

paths = getListOfFiles('/kaggle/input/CORD-19-research-challenge')
# The below function will extract the abstract and body text of a .json file



def clean_json(path):

    #loads json file into a string

    with open(path) as f:

        dic = json.load(f)

        

    try:    

        paper_id = dic['paper_id']

    

        txt = dic['abstract'][0]['text']

    except:

        #import pdb;pdb.set_trace()

        txt= ""

    for i in range(len(dic['body_text'])):

        par = dic['body_text'][i]['text']

        txt = txt + ' ' + par

    return paper_id, txt
## Getting the paper IDs and texts for all the files 



ids = []

texts = []

for path in paths:

    paper_id, txt = clean_json(path)

    ids.append(paper_id) #keep track of the paper's id so that I can merge the dataframes eventually

    texts.append(txt)

        

ids_and_texts = pd.DataFrame(data=list(zip(ids,texts)),columns=['sha','text'])



ids_and_texts.head()
# This function will make the links ready to pop in your browser

def doi_url(doi):

    if not(isinstance(doi,str)): return '-'

    return f'http://{doi}' if doi.startswith('doi.org') else f'http://doi.org/{doi}'



metadata.doi = metadata.doi.apply(doi_url)
#all articles with sha i.e. contains the full text

full_data = metadata.loc[~metadata.sha.isnull()]



#all articles without sha (doesn't have the full text)

not_sha = metadata.loc[metadata.sha.isnull()]
# Using the abstract as the text of the article.

not_sha['text'] = not_sha['abstract']



# If the article doesn't have the abstract I'll use the title

idx = not_sha.loc[not_sha.text.isnull()].index

not_sha.text.loc[idx] = not_sha.title.loc[idx]



# if it doesn't have a title I'll drop it

not_sha.drop(not_sha.text.isnull().index,inplace=True)
#now merging the metadata with the dataframe that has the full texts

complete_data = full_data.merge(ids_and_texts,on=['sha'],how='inner')

complete_data = pd.concat((complete_data,not_sha))

complete_data = complete_data.reset_index()





## Converting columns to string

complete_data['abstract'] = complete_data['abstract'].astype(str)

complete_data['text'] = complete_data['text'].astype(str)



## Adding word count columns



complete_data['abstract_word_count'] = complete_data['abstract'].apply(lambda x: len(x.strip().split()))

complete_data['body_word_count'] = complete_data['text'].apply(lambda x: len(x.strip().split()))





## Keeping a copy and moving forward

data = complete_data.copy()



complete_data.head()
## Subset of the data to extract relevant topics



#Create new dataframe that just store the title and abstract to work on

text_df = complete_data[['sha','title','abstract', 'text']].copy()

text_df.head(7)

# Defining a text cleaner to do the cleaning we need - lower case, and replace all punctuation, numbers with empty strings

# Note the explict casting of the title elements into str is needed to use string operations without an error

def text_cleaner(text):

    text = str(text).lower()

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'[^a-z ]','',text)

    return text
### Bringing all abstract, title & text columns words in lower format



text_df.title = text_df.title.apply(lambda x: text_cleaner(x))

text_df = text_df[text_df['title'].notna()]



text_df.abstract = text_df.abstract.apply(lambda x: text_cleaner(x))

text_df = text_df[text_df['abstract'].notna()]



text_df.text = text_df.text.apply(lambda x: text_cleaner(x))

text_df = text_df[text_df['text'].notna()]





text_df.head()
#### FILTERING DATA FROM LOOKING INTO WORDS FROM ABSTRACT 



# Vectorizing and removing stop words 



tfidf = TfidfVectorizer(max_features=5000, stop_words=nltk.corpus.stopwords.words('english'))

X = tfidf.fit_transform(text_df.abstract)



# Scaling all the features and ensuring that all the features have 0 mean and unit standard deviation



scaler = StandardScaler()

X = scaler.fit_transform(X.todense())





### Now extracting  keywords from each text 



def get_keywords(X,tfidf,k=200):

    '''

    X: is the features matrix

    tfidf: is the tfidf object used to vectorize the texts

    k: maximum number of keywords for each text

    '''



    feature_names = tfidf.get_feature_names()

    keywords = []

    ponts_tfidf = []

    for i in range(X.shape[0]):

        text_vector = X[i]

        idxs = np.array(text_vector.argsort()[-k:][::-1]).T #getting the index of the most important words (with more tfidf ponctuation)

        s=''

        for j in range(k):            

            # To make sure there are no useless words

            if text_vector[idxs[j]] != 0:

                s = s + feature_names[idxs[j]] + ','

        keywords.append(s)

    return keywords



## Extracting the keywords from the above method



keywords = get_keywords(X,tfidf)



## Appending all the keywords to the full data 

text_df['abs_keywords'] = keywords



text_df.head(7) 
#Identify top words by aggregating the table

top_words = dtm.sum(axis = 0).sort_values(ascending = False)

print(top_words[0:50])
### Filtering docs based on relevant phrases and words



## Task Breakups  



#Resources to support skilled nursing facilities and long term care facilities.






