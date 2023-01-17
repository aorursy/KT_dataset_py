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

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("../input/medium-post-titles/medium_post_titles.csv")
df.head()
del df['category']

del df['title']
df.head()
df.isna().sum()
df.dropna(inplace = True)
df.subtitle.value_counts()
df.subtitle_truncated_flag.value_counts()
df.subtitle_truncated_flag.replace(False,0,inplace = True)

df.subtitle_truncated_flag.replace(True,1,inplace = True)
text_truncated = df[df.subtitle_truncated_flag == 1].subtitle.values[:80000]

text_nottruncated = df[df.subtitle_truncated_flag == 0].subtitle.values[:80000]
plt.figure(figsize = (20,20))

wc = WordCloud(max_words = 3000 , min_font_size = 3 , width = 1600 , height = 800 ,stopwords = STOPWORDS).generate(str(text_truncated))

plt.imshow(wc,interpolation = 'bilinear')
truncated_keys = list(wc.process_text(str(text_truncated)).keys())
train_text = df.subtitle[:80000]

test_text = df.subtitle[80000:]

train_category = df.subtitle_truncated_flag[:80000]

test_category = df.subtitle_truncated_flag[80000:]
predictions = []

for i in test_text:

    x = i.split()

    for j in x:

        if j in truncated_keys:

            predictions.append(1)

            break

        else:

            predictions.append(0)

            break

len(predictions)            
len(test_category)
count = 0

for i in range(len(predictions)):

    test_category = list(test_category)

    if(predictions[i] == int(test_category[i])):

        count += 1

print(count)        
accuracy = (count/len(predictions))* 100

accuracy
print("Accuracy by using WordCloud is : " , accuracy)
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

    return final_text        
df.subtitle = df.subtitle.apply(lemmatize_words)
def join_text(text):

    string = ''

    for i in text:

        string += i.strip() +' '

    return string    
df.subtitle = df.subtitle.apply(join_text)
df.head()
def int_converter(text):

    return int(text)

df.subtitle_truncated_flag = df.subtitle_truncated_flag.apply(int_converter)
train_message = df.subtitle[:80000]

test_message = df.subtitle[80000:]

train_category = df.subtitle_truncated_flag[:80000]

test_category = df.subtitle_truncated_flag[80000:]
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,2))

#transformed train reviews

cv_train_reviews=cv.fit_transform(train_message)

#transformed test reviews

cv_test_reviews=cv.transform(test_message)



print('BOW_cv_train:',cv_train_reviews.shape)

print('BOW_cv_test:',cv_test_reviews.shape)
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))

#transformed train reviews

tv_train_reviews=tv.fit_transform(train_message)

#transformed test reviews

tv_test_reviews=tv.transform(test_message)

print('Tfidf_train:',tv_train_reviews.shape)

print('Tfidf_test:',tv_test_reviews.shape)
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)

#Fitting the model for Bag of words

lr_bow=lr.fit(cv_train_reviews,train_category)

print(lr_bow)

#Fitting the model for tfidf features

lr_tfidf=lr.fit(tv_train_reviews,train_category)

print(lr_tfidf)
#Predicting the model for bag of words

lr_bow_predict=lr.predict(cv_test_reviews)

##Predicting the model for tfidf features

lr_tfidf_predict=lr.predict(tv_test_reviews)
#Accuracy score for bag of words

lr_bow_score=accuracy_score(test_category,lr_bow_predict)

print("lr_bow_score :",lr_bow_score)

#Accuracy score for tfidf features

lr_tfidf_score=accuracy_score(test_category,lr_tfidf_predict)

print("lr_tfidf_score :",lr_tfidf_score)
#Classification report for bag of words

lr_bow_report=classification_report(test_category,lr_bow_predict,target_names=['0','1'])

print(lr_bow_report)



#Classification report for tfidf features

lr_tfidf_report=classification_report(test_category,lr_tfidf_predict,target_names=['0','1'])

print(lr_tfidf_report)
#training the model

mnb=MultinomialNB()

#fitting the nb for bag of words

mnb_bow=mnb.fit(cv_train_reviews,train_category)

print(mnb_bow)

#fitting the nb for tfidf features

mnb_tfidf=mnb.fit(tv_train_reviews,train_category)

print(mnb_tfidf)
#Predicting the model for bag of words

mnb_bow_predict=mnb.predict(cv_test_reviews)

#Predicting the model for tfidf features

mnb_tfidf_predict=mnb.predict(tv_test_reviews)
#Accuracy score for bag of words

mnb_bow_score=accuracy_score(test_category,mnb_bow_predict)

print("mnb_bow_score :",mnb_bow_score)

#Accuracy score for tfidf features

mnb_tfidf_score=accuracy_score(test_category,mnb_tfidf_predict)

print("mnb_tfidf_score :",mnb_tfidf_score)
mnb_bow_report = classification_report(test_category,mnb_bow_predict,target_names = ['0','1'])

print(mnb_bow_report)

mnb_tfidf_report = classification_report(test_category,mnb_tfidf_predict,target_names = ['0','1'])

print(mnb_tfidf_report)