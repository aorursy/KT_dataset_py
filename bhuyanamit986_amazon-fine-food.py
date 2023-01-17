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
%matplotlib inline

import numpy as np 

import pandas as pd

import sqlite3 as sql

import nltk

import string

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



#using SQLite table to read data.

con=sql.connect('../input/amazon-fine-food-reviews/database.sqlite')
data=pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score!=3

""", con)
data.head()
#Partitioning data into positive (1) class or negative(0) class as class is not provided

def partition(x):

    if x<3:

        return 0

    return 1



#now divide the reviews into positive and negative reviews

review_data=data['Score']

positiveNegative=review_data.map(partition)

data['target']=positiveNegative
data.head()
#Let's first sort our dataset

sorted_dataset=data.sort_values('ProductId', axis=0, ascending=True)

sorted_dataset.head()
#Now let's deduplication of our dataset as the it may be possible that a same product have different variants then 

#it will have the same reviews but the product id's will be different so it is wiseful to remove those duplicated values



final_data=sorted_dataset.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep='first', inplace=True)
sorted_dataset.shape
corrupt_data=pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score!=3 AND HelpfulnessNumerator>HelpfulnessDenominator

ORDER BY ProductId

""", con)

corrupt_data
sorted_dataset=sorted_dataset[sorted_dataset.HelpfulnessNumerator<=sorted_dataset.HelpfulnessDenominator]
sorted_dataset["Score"].value_counts()
#counts the number of positive and negative reviews

sorted_dataset["target"].value_counts()
count_vec=CountVectorizer()

final_counts=count_vec.fit_transform(sorted_dataset["Text"].values)
sorted_dataset["Text"][138688]
count_vec.vocabulary_ 

"""creates a dictionary of all the words of the documents that are fitted to the bag of words which constitutes of word 

and a random index provided to each word"""
final_counts.get_shape()
without_stop_words=CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2))
vec_wo_stop=without_stop_words.fit_transform(sorted_dataset.Text.values)
vec_wo_stop.get_shape()
without_stop_words.vocabulary_
stop=nltk.corpus.stopwords.words('english') #set of stop words.

stemmer=nltk.stem.SnowballStemmer('english') #used for stemming english words
import re
def cleanhtml(sentence):

    cleanr=re.compile('<.*?>')

    filtered=re.sub(cleanr, ' ', sentence)

    return filtered
def cleanpunc(sentence):

    cleaned=re.sub(r'[?|''|}|{|"|!|#]', r'', sentence)

    cleaned=re.sub(r'[.|:|;|&|,|/|\|)|(]', r' ', cleaned)

    return cleaned
stemmer.stem('most beautiful')
def preprocessor(sentence):

    cleanr=re.compile('<.*?>')

    filtered=re.sub(cleanr, ' ', sentence)

    cleaned=re.sub(r'[?|''|}|{|"|!|#]', r'', filtered)

    cleaned=re.sub(r'[.|:|;|&|,|/|\|)|(]', r' ', cleaned)

    return cleaned

    
def tokenizer(sentence):

    text=[]

    for w in sentence.split():

        if w not in stop:

            text.append(w)

        

    return [stemmer.stem(word.lower()).encode('utf8') for word in text]

    
sorted_dataset["Text"]=sorted_dataset["Text"].apply(preprocessor)
sorted_dataset["Text"][138706]
sorted_dataset["Text"]=sorted_dataset["Text"].apply(tokenizer)
def make_str(lis):

    return b' '.join(lis)
sorted_dataset["Text"]=sorted_dataset["Text"].apply(make_str)
sorted_dataset["Text"][138706]
from sklearn.model_selection import train_test_split
data_x, data_y=train_test_split(sorted_dataset, test_size=0.2, random_state=0)
print(data_x.shape, data_y.shape)
tfidf=TfidfVectorizer(ngram_range=(1,2))
X_transformed=tfidf.fit_transform(data_x.Text)
tfidf.vocabulary_
X_transformed.get_shape()
X_train=data_x['Text']

X_test=data_y['Text']

y_train=data_x['target']

y_test=data_y['target']
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
tfidf=TfidfVectorizer(ngram_range=(1,2))
lr_tfidf=Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))], verbose=1)
lr_tfidf.fit(X_train, y_train)
y_hat=lr_tfidf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, y_hat)
cmat=confusion_matrix(y_test, y_hat)
plt.figure(figsize=(16, 12))

sns.heatmap(cmat)

plt.xlabel('True label')

plt.ylabel('Predicted label')
cmat
y_test.value_counts()