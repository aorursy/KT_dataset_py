# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
test.head()
import seaborn as sns

import pandas as pd



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif
def punc(df,stri):

    df[stri] = df[stri].str.lower().str.replace('[^a-z]', ' ')

    df[stri] = df[stri].str.lower().str.replace('[^\w\s]',' ')

    df[stri] = df[stri].str.lower().str.replace(r'<.*?>',' ')

    df[stri] = df[stri].str.lower().str.replace(' br ',' ')

    return df
train=punc(train,"question1")

train=punc(train,"question2")

#test=punc(test,"question2")

#test=punc(test,"question1")
train.head()
train.isna().sum()
test.isna().sum()
test=test.dropna()

train=train.dropna()
test.head()
import nltk

from nltk.corpus import stopwords



stop = stopwords.words('english')
def stopi(df,stri): 

    df[stri] = df[stri].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    return df
train=stopi(train,"question1")

train=stopi(train,"question2")

#test=stopi(test,"question2")

#test=stopi(test,"question1")
train.info()
train.head()
from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()

def port(df,stri):

    df[stri] = df[stri].apply(lambda x: " ".join(ps.stem(word) for word in x.split()))

    return df
train=port(train,"question1")

train=port(train,"question2")
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

train['question1'] = train['question1'].apply(lambda x: " ".join(wordnet_lemmatizer.lemmatize(word) for word in x.split()))

train['question2'] = train['question2'].apply(lambda x: " ".join(wordnet_lemmatizer.lemmatize(word) for word in x.split()))
train.head()
from difflib import SequenceMatcher

a = "Dump Administration Dismisses Surgeon General Vivek Murthy (http)PUGheO7BuT5LUEtHDcgm"

b = "Dump Administration Dismisses Surgeon General Vivek Murthy (http)avGqdhRVOO"

ratio = SequenceMatcher(None, a, b).ratio()

ar=[]

for index, row in train.iterrows():

    #print(row['c1'], row['c2'])

    a=SequenceMatcher(None,row['question1'],row['question2']).ratio()

    ar.append(a)

    
train['simi']=ar
train.head()
train['sim'] = np.where(train['simi']==1.0, 1, 0)
train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words ='english', smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
import re, math

from collections import Counter



WORD = re.compile(r'\w+')



def get_cosine(vec1, vec2):

    intersection = set(vec1.keys()) & set(vec2.keys())

    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])

    sum2 = sum([vec2[x]**2 for x in vec2.keys()])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:

        return 0.0

    else:

        return float(numerator) / denominator



def text_to_vector(text):

    words = WORD.findall(text)

    return Counter(words)



text1 = 'This is a foo bar sentence .'

text2 = 'This sentence is similar to a foo bar sentence .'



vector1 = text_to_vector(text1)

vector2 = text_to_vector(text2)



cosine = get_cosine(vector1, vector2)



ar=[]

for index, row in train.iterrows():

    #print(row['c1'], row['c2'])

    vector1 = text_to_vector(row['question1'])

    vector2 = text_to_vector(row['question2'])

    a=get_cosine(vector1, vector2)

    ar.append(a)

    
train['simi1']=ar
train.head()

x=train.ix[:,3:5]
y=train.is_duplicate
x.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
sample = X_train["question1"] + "," + X_train["question1"]



def my_tokenizer(s):

    return s.split(",")



vect = CountVectorizer(max_features=1500)

vect = CountVectorizer(analyzer='word',tokenizer=my_tokenizer, ngram_range=(1, 3), min_df=1) 

train1 = vect.fit_transform(sample.values)
train1
sample = X_test["question1"] + "," + X_test["question1"]



def my_tokenizer(s):

    return s.split(",")





test1 = vect.transform(sample.values)
test1
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(train1, y_train)

predictions = clf.predict(test1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

accuracy
from sklearn.linear_model import LogisticRegression

log=LogisticRegression(penalty='l2',C=.00001)

log.fit(train1,y_train)

y_pred = log.predict(test1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy
sample = X_train[['question1','question2']]

sample = X_train.apply(lambda col: col.str.strip())

import scipy.sparse as sp

vect = CountVectorizer(max_features=1500,ngram_range=(1, 3))

train1 = sp.hstack(sample.apply(lambda col: vect.fit_transform(col)))
sample = X_test[['question1','question2']]

sample = X_test.apply(lambda col: col.str.strip())

import scipy.sparse as sp

test1 = sp.hstack(sample.apply(lambda col: vect.transform(col)))
from sklearn.linear_model import LogisticRegression

log=LogisticRegression(penalty='l2',C=.00001)

log.fit(train1,y_train)

y_pred = log.predict(test1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(train1, y_train)

predictions = clf.predict(test1)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

accuracy