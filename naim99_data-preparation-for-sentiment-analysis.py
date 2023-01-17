import pandas as pd

import numpy as np 



import pandas as pd

# Import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

import nltk 

import string

import re

import numpy as np

from wordcloud import WordCloud 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # function for splitting data to train and test sets



import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier



from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

import seaborn as sns



import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from nltk.tokenize import TweetTokenizer



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score, f1_score

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

train= pd.read_csv("../input/train.csv", sep=',', error_bad_lines=False , encoding ='ISO-8859-1')

test = pd.read_csv("../input/test.csv", sep=',', error_bad_lines=False , encoding ='ISO-8859-1')



train
train = train.rename(columns={"SentimentText;;;;;;;;;;;;": "SentimentText"}) 

print(train.columns.tolist())
print(train.columns.tolist())
test = test.rename(columns={"SentimentText;;;;;;;;;;;;;;;;;;;": "SentimentText"}) 
test
print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('ItemID')['SentimentText'].count().mean()))

print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('ItemID')['SentimentText'].count().mean()))
print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.ItemID.unique())))

print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.ItemID.unique())))
train['SentimentText'] = train['SentimentText'].astype(str) 

test['SentimentText'] = test['SentimentText'].astype(str) 

print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['SentimentText'].apply(lambda x: len(x.split())))))

print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['SentimentText'].apply(lambda x: len(x.split())))))
train = train[train['SentimentText'].notna()]

train = train[train['Sentiment'].notna()]

train
value = train['Sentiment'].value_counts()

value
train
train_1 = train[~train['Sentiment'].isin(["0","1"])]

value_counts_sentiment  = train_1['Sentiment'].value_counts()

trainG = pd.merge(train,train_1, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

trainG
train_1
value_counts_sentiment  = trainG['Sentiment'].value_counts()

value_counts_sentiment 
test = test[test['SentimentText'].notna()]
patterns= ['[^!.?]+']



for tweet in trainG['SentimentText'] :

    #Convert to lower case

         tweet = tweet.lower()

         #Convert www.* or https?://* to URL

         tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)

         #Convert @username to AT_USER

         tweet = re.sub('@[^\s]+','AT_USER',tweet)

         #Remove additional white spaces

         tweet = re.sub('[\s]+', ' ', tweet)

         #Replace #word with word

         tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

         #trim

         tweet = tweet.strip('\'"')

         

         

test['SentimentText'] = test['SentimentText'].astype(str)

for tweet in test['SentimentText'] :

    #Convert to lower case

         tweet = tweet.lower()

         #Convert www.* or https?://* to URL

         tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)

         #Convert @username to AT_USER

         tweet = re.sub('@[^\s]+','AT_USER',tweet)

         #Remove additional white spaces

         tweet = re.sub('[\s]+', ' ', tweet)

         #Replace #word with word

         tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

         #trim

         tweet = tweet.strip('\'"')       

for tweet in trainG['SentimentText'] :

    for p in patterns:

        tweet = re.findall(p, tweet)

    
trainG
import re

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")



for string in trainG['SentimentText']:

    

    string = string.lower().replace("<br />", " ")

    string = re.sub(strip_special_chars, "", string.lower())    

    

trainG
trainG.shape
test.shape
trainG['SentimentText'] = trainG['SentimentText'].astype(str)

trainG = trainG[trainG['SentimentText'].notna()]
#Clean text from noise

def clean_text(text):

    #Filter to allow only alphabets

    text = re.sub(r'[^a-zA-Z\']', ' ', text)

    

    #Remove Unicode characters

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    

    #Convert to lowercase to maintain consistency

    text = text.lower()

       

    return text


trainG['SentimentText'] = trainG['SentimentText'].apply(lambda x:clean_text(x))
def tokenize(text): 

    tknzr = TweetTokenizer()

    return tknzr.tokenize(text)



def stem(doc):

    return (stemmer.stem(w) for w in analyzer(doc))



en_stopwords = set(stopwords.words("english")) 



vectorizer = CountVectorizer(

    analyzer = 'word',

    tokenizer = tokenize,

    lowercase = True,

    ngram_range=(1, 1),

    stop_words = en_stopwords)


#data=pd.concat([trainG,test],sort=False).reset_index(drop=True)
#data
#data['SentimentText'] = data['SentimentText'].astype(str) 
#Clean text from noise

def clean_text(text):

    #Filter to allow only alphabets

    text = re.sub(r'[^a-zA-Z\']', ' ', text)

    

    #Remove Unicode characters

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    

    #Convert to lowercase to maintain consistency

    text = text.lower()

       

    return text
#data['SentimentText']  =data['SentimentText'] .apply(lambda x:clean_text(x))
data = trainG.copy()

data.drop('ItemID', axis=1, inplace=True)
data.head()
print(data.isnull().any())
data.dtypes
data['Sentiment'] = data['Sentiment'].astype('category')

print(type(data['Sentiment'][0]))
data['label_id'] = data['Sentiment'].cat.codes

data['label_id'].head()
data
test.tail()
trainG.to_excel("trainG_clean.xlsx", sheet_name='global', index=False)

test.to_excel("test_clean.xlsx",sheet_name = "gloabl" , index = False) 