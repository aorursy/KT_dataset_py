# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re

import pickle 
#import mglearn
import time


from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
print(os.listdir("../input"))
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.

#corpus exploration
train.head()





#corpus exploration
test.head()









#corpus exploration
len(train['question_text'])
#corpus exploration
len(test['question_text'])







#some operations to clean the corpus
count_train=[]
bag_of_words=[]
count_vec = CountVectorizer(stop_words="english", analyzer='word',ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=4)
train_x=[]
train_y=[]
for j in range(len(train['question_text'])):
    train_x.append(train['question_text'][j])



#some operations to clean the corpus
from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')
#Preprocess the text in training and testing
processed_train = []
for doc in train_x:
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_train.append(stemmed)




#save the results
train['processed_train']=processed_train
train.head()



#save the results
row_lst = []
for lst in train.loc[:,'processed_train']:
    text = ''
    for word in lst:
        text = text + ' ' + word
    row_lst.append(text)

train['final_processed_test'] = row_lst

train.head()











count_test=[]
bag_of_words=[]
count_vec = CountVectorizer(stop_words="english", analyzer='word',ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=4)
test_x=[]
test_y=[]
for j in range(len(test['question_text'])):
    test_x.append(test['question_text'][j])


from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')
#Preprocess the text in training and testing
processed_test = []
for doc in test_x:
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_test.append(stemmed)


test['processed_test']=processed_test
test.head()


row_lst = []
for lst in test.loc[:,'processed_test']:
    text = ''
    for word in lst:
        text = text + ' ' + word
    row_lst.append(text)

test['final_processed_test'] = row_lst

test.head()


                    
















def format_sentence(sent):

    return ({ word: True for word in nltk.word_tokenize(sent)})






#prepare the corpus for the naive method
def load_training_set():

    training = []
    for i in range(len(train['final_processed_test'])):
        if (train['target'][i]==1):
            training.append([format_sentence(train['final_processed_test'][i]), 1])
        else:
            training.append([format_sentence(train['final_processed_test'][i]), 0])
    return training














training = load_training_set()


#naive Bayes
from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(training)
def word_feats(words):
    return dict([(word, True) for word in words])

#prepare the prediction
predicted = []
for k in range(len(test['question_text'])):
    predicted.append(classifier.classify(word_feats(test['question_text'][k])))






















#prepare the submission
submission = pd.DataFrame({"qid": test['qid'],"prediction":predicted })
submission.to_csv("submission.csv", index=False)
















































































































































































































































































































































































































































































