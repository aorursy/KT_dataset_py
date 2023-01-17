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
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import linear_model

import numpy as np

from sklearn.model_selection import train_test_split

import scipy

from sklearn.metrics import log_loss

import xgboost as xgb

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/questions.csv")

df.head()
df.groupby("is_duplicate")['id'].count().plot.bar()
df.isnull().sum()

df = df.dropna(how='any',axis=0) 
df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)
a = 0 

for i in range(a,a+10):

    print(df.question1[i])

    print(df.question2[i])

    print()
SPECIAL_TOKENS = {

    'quoted': 'quoted_item',

    'non-ascii': 'non_ascii_word',

    'undefined': 'something'

}



def clean(text, stem_words=True):

    import re

    from string import punctuation

    from nltk.stem import SnowballStemmer

    from nltk.corpus import stopwords

    

    def pad_str(s):

        return ' '+s+' '

    

    if pd.isnull(text):

        return ''



#    stops = set(stopwords.words("english"))

    # Clean the text, with the option to stem words.

    

    # Empty question

    

    if type(text) != str or text=='':

        return ''



    # Clean the text

    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly

    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)

    text = re.sub("\'ve", " have ", text)

    text = re.sub("can't", "can not", text)

    text = re.sub("n't", " not ", text)

    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)

    text = re.sub("\'re", " are ", text)

    text = re.sub("\'d", " would ", text)

    text = re.sub("\'ll", " will ", text)

    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)

    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)

    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)

    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)

    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)

    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)

    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)

    text = re.sub("[c-fC-F]\:\/", " disk ", text)

    

    # remove comma between numbers, i.e. 15,000 -> 15000

    

    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    

#     # all numbers should separate from words, this is too aggressive

    

#     def pad_number(pattern):

#         matched_string = pattern.group(0)

#         return pad_str(matched_string)

#     text = re.sub('[0-9]+', pad_number, text)

    

    # add padding to punctuations and special chars, we still need them later

    

    text = re.sub('\$', " dollar ", text)

    text = re.sub('\%', " percent ", text)

    text = re.sub('\&', " and ", text)

    

#    def pad_pattern(pattern):

#        matched_string = pattern.group(0)

#       return pad_str(matched_string)

#    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text) 

        

    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) # replace non-ascii word with special word

    

    # indian dollar

    

    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)

    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)

    

    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)

    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)

    text = re.sub(r" india ", " India ", text)

    text = re.sub(r" switzerland ", " Switzerland ", text)

    text = re.sub(r" china ", " China ", text)

    text = re.sub(r" chinese ", " Chinese ", text) 

    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)

    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)

    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)

    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  

    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 

    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)

    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)

    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 

    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)

    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)

    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)

    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)

    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)

    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)

    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)

    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)

    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)

    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)

    text = re.sub(r" III ", " 3 ", text)

    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)

    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)

    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)

    

    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"

    

    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)

  

    

    # Remove punctuation from text

    text = ''.join([c for c in text if c not in punctuation]).lower()

       # Return a list of words

    return text
df['question1'] = df['question1'].apply(clean)

df['question2'] = df['question2'].apply(clean)
a = 0 

for i in range(a,a+10):

    print(df.question1[i])

    print(df.question2[i])

    print()
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(pd.concat((df['question1'],df['question2'])).unique())

trainq1_trans = count_vect.transform(df['question1'].values)

trainq2_trans = count_vect.transform(df['question2'].values)

labels = df['is_duplicate'].values

X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))

y = labels

X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 

xgb_prediction = xgb_model.predict(X_valid)
from sklearn.metrics import f1_score, classification_report, accuracy_score



print('training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))

print('validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))

print(classification_report(y_valid, xgb_prediction))
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(pd.concat((df['question1'],df['question2'])).unique())

trainq1_trans = tfidf_vect.transform(df['question1'].values)

trainq2_trans = tfidf_vect.transform(df['question2'].values)

labels = df['is_duplicate'].values

X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))

y = labels

X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)
from sklearn.metrics import f1_score, classification_report, accuracy_score

xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 

xgb_prediction = xgb_model.predict(X_valid)

print('word level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))

print('word level tf-idf validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))

print(classification_report(y_valid, xgb_prediction))
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(pd.concat((df['question1'],df['question2'])).unique())

trainq1_trans = tfidf_vect_ngram.transform(df['question1'].values)

trainq2_trans = tfidf_vect_ngram.transform(df['question2'].values)

labels = df['is_duplicate'].values

X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))

y = labels

X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 

xgb_prediction = xgb_model.predict(X_valid)

print('n-gram level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))

print('n-gram level tf-idf validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))

print(classification_report(y_valid, xgb_prediction))
from sklearn.metrics import f1_score, classification_report, accuracy_score

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(pd.concat((df['question1'],df['question2'])).unique())

trainq1_trans = tfidf_vect_ngram_chars.transform(df['question1'].values)

trainq2_trans = tfidf_vect_ngram_chars.transform(df['question2'].values)

labels = df['is_duplicate'].values

X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))

y = labels

X_train,X_valid,y_train,y_valid = train_test_split(X,y, test_size = 0.33, random_state = 42)

xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train) 

xgb_prediction = xgb_model.predict(X_valid)

print('character level tf-idf training score:', f1_score(y_train, xgb_model.predict(X_train), average='macro'))

print('character level tf-idf validation score:', f1_score(y_valid, xgb_model.predict(X_valid), average='macro'))

print(classification_report(y_valid, xgb_prediction))