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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize,sent_tokenize

from bs4 import BeautifulSoup

import re,string,unicodedata

from nltk.tokenize.toktok import ToktokTokenizer

from nltk.stem import LancasterStemmer,WordNetLemmatizer

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from string import punctuation

from nltk import pos_tag

from nltk.corpus import wordnet

import keras

from keras.layers import Dense,LSTM

from keras.models import Sequential
df = pd.read_csv("../input/amazon-music-reviews/Musical_instruments_reviews.csv")
df.head()
df.isna().sum()
df.reviewText.fillna("",inplace = True)
del df['reviewerID']

del df['asin']

del df['reviewerName']

del df['helpful']

del df['unixReviewTime']

del df['reviewTime']
df.head()
df['text'] = df['reviewText'] + ' ' + df['summary']

del df['reviewText']

del df['summary']
df.head()
df.overall.value_counts()
def sentiment_rating(rating):

    # Replacing ratings of 1,2,3 with 0 (not good) and 4,5 with 1 (good)

    if(int(rating) == 1 or int(rating) == 2 or int(rating) == 3):

        return 0

    else: 

        return 1

df.overall = df.overall.apply(sentiment_rating) 
df.overall.value_counts()
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def get_simple_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            pos = pos_tag([i.strip()])

            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))

            final_text.append(word.lower())

    return " ".join(final_text)
df.text = df.text.apply(lemmatize_words)
x_train,x_test,y_train,y_test = train_test_split(df.text,df.overall,test_size = 0.2 , random_state = 0)
good = x_train[y_train[y_train == 1].index]

bad = x_train[y_train[y_train == 0].index]

x_train.shape,good.shape,bad.shape
plt.figure(figsize = (20,20)) # Text Reviews with Poor Ratings

wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(bad))

plt.imshow(wc,interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # # Text Reviews with Good Ratings

wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(good))

plt.imshow(wc,interpolation = 'bilinear')
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))

#transformed train reviews

cv_train_reviews=cv.fit_transform(x_train)

#transformed test reviews

cv_test_reviews=cv.transform(x_test)



print('BOW_cv_train:',cv_train_reviews.shape)

print('BOW_cv_test:',cv_test_reviews.shape)
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))

#transformed train reviews

tv_train_reviews=tv.fit_transform(x_train)

#transformed test reviews

tv_test_reviews=tv.transform(x_test)

print('Tfidf_train:',tv_train_reviews.shape)

print('Tfidf_test:',tv_test_reviews.shape)
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)

#Fitting the model for Bag of words

lr_bow=lr.fit(cv_train_reviews,y_train)

print(lr_bow)

#Fitting the model for tfidf features

lr_tfidf=lr.fit(tv_train_reviews,y_train)

print(lr_tfidf)
#Predicting the model for bag of words

lr_bow_predict=lr.predict(cv_test_reviews)

##Predicting the model for tfidf features

lr_tfidf_predict=lr.predict(tv_test_reviews)
#Accuracy score for bag of words

lr_bow_score=accuracy_score(y_test,lr_bow_predict)

print("lr_bow_score :",lr_bow_score)

#Accuracy score for tfidf features

lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)

print("lr_tfidf_score :",lr_tfidf_score)
#Classification report for bag of words

lr_bow_report=classification_report(y_test,lr_bow_predict,target_names=['0','1'])

print(lr_bow_report)



#Classification report for tfidf features

lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['0','1'])

print(lr_tfidf_report)
#training the model

mnb=MultinomialNB()

#fitting the nb for bag of words

mnb_bow=mnb.fit(cv_train_reviews,y_train)

print(mnb_bow)

#fitting the nb for tfidf features

mnb_tfidf=mnb.fit(tv_train_reviews,y_train)

print(mnb_tfidf)
#Predicting the model for bag of words

mnb_bow_predict=mnb.predict(cv_test_reviews)

#Predicting the model for tfidf features

mnb_tfidf_predict=mnb.predict(tv_test_reviews)
#Accuracy score for bag of words

mnb_bow_score=accuracy_score(y_test,mnb_bow_predict)

print("mnb_bow_score :",mnb_bow_score)

#Accuracy score for tfidf features

mnb_tfidf_score=accuracy_score(y_test,mnb_tfidf_predict)

print("mnb_tfidf_score :",mnb_tfidf_score)
mnb_bow_report = classification_report(y_test,mnb_bow_predict,target_names = ['0','1'])

print(mnb_bow_report)

mnb_tfidf_report = classification_report(y_test,mnb_tfidf_predict,target_names = ['0','1'])

print(mnb_tfidf_report)
model = Sequential()

model.add(Dense(units = 75 , activation = 'relu' , input_dim = cv_train_reviews.shape[1]))

model.add(Dense(units = 50 , activation = 'relu'))

model.add(Dense(units = 25 , activation = 'relu'))

model.add(Dense(units = 10 , activation = 'relu')) 

model.add(Dense(units = 1 , activation = 'sigmoid'))

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()
model.fit(cv_train_reviews,y_train , epochs = 10)
model.evaluate(cv_test_reviews,y_test)[1]