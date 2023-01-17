# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

#text processing libraries
import re
import string
import nltk
from nltk.corpus import stopwords
#XG boost 
import xgboost as xgb
from xgboost import XGBClassifier
#sklearn
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV


#matplotliB
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir('../input/'))
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
print('Training data shape', train.shape)
train.head()
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print('Test data shape', test.shape)
test.head()
# let's working on Traning dataset EDA and cleaning
train.isnull().sum()
test.isnull().sum()
train['target'].value_counts()
sns.barplot(train['target'].value_counts().index,train['target'].value_counts(),palette='rocket')
disaster_tweets = train[train['target']==1]['text']
disaster_tweets.values[1]
non_disaster_tweets = train[train['target']==0]['text']
non_disaster_tweets.values[2]
sns.barplot(y=train['keyword'].value_counts()[:20].index, x=train['keyword'].value_counts()[:20], orient='h')
train.loc[train['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
train.head()
train['location'].notnull().head()
train['location'].replace({'United States':'USA',
                           'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "California":'USA',
                             "Chicago, IL":'USA',
                             "Chicago":'USA',
                            "New York, NY":'USA',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "Earth":'Worldwide',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India"},inplace=True)

sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],
            orient='h')
train['text'][:5]
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?]', '',text)
    text = re.sub('https?://\|S+',"",text)
    text = re.sub('<.*?>+',"",text)
    text = re.sub('[%s]' % re.escape(string.punctuation),"",text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text
train['text'] = train['text'].apply(lambda x: clean_text(x))
test['text'] = test['text'].apply(lambda x: clean_text(x))
train['text'].head()
#tokenization
# Tokenizing the training and the test set
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
train['text'].head()
#stopwords removal
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words
train['text'] = train['text'].apply(lambda x: remove_stopwords(x))
test['text'] = test['text'].apply(lambda x : remove_stopwords(x))
train.head()

## Stemming and Lemmatization examples
'''text = "feet cats wolves talked"

tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)

# Stemmer
stemmer = nltk.stem.PorterStemmer()
print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))

# Lemmatizer
lemmatizer=nltk.stem.WordNetLemmatizer()
print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))'''
def combine_text(list_of_words):
    combined_text = ''.join(list_of_words)
    return combined_text
train['text'] = train['text'].apply(lambda x : combine_text(x))
test['text'] = test['text'].apply(lambda x: combine_text(x))
train['text']
train.head()
'''def text_preocessing(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text'''
#count_vectorizer = CountVectorizer()


