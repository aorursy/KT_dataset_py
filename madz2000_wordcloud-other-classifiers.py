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
df = pd.read_csv("../input/onion-or-not/OnionOrNot.csv")
df.head()
df.isna().sum()
df.label.value_counts()
x = df[:20000]

true_text = x[x.label == 1].text

false_text = x[x.label == 0].text
plt.figure(figsize = (20,20))

wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(true_text))

plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20))

wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(false_text))

plt.imshow(wc , interpolation = 'bilinear')
final_truetext = wc.process_text(" ".join(true_text))

final_falsetext = wc.process_text(" ".join(false_text))
final_truetext = sorted(final_truetext.items(),key = 

             lambda kv:(kv[1], kv[0]))

final_falsetext = sorted(final_falsetext.items(),key = 

             lambda kv:(kv[1], kv[0]))
len(final_truetext)
len(final_falsetext)
final_truetext = final_truetext[-3000:]

final_falsetext = final_falsetext[-3000:]
text_true = []

text_false = []

for i in range(3000):

    text_true.append(final_truetext[i][0])

    text_false.append(final_falsetext[i][0])

text_true[:5],text_false[:5]    
predictions = []

for i in test_text:

    x = i.split()

    for j in x:

        if j in text_true and j not in text_false:

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

    if(predictions[i] == test_category[i]):

        count += 1

print(count)
accuracy = (count/len(predictions))*100

accuracy
print("Accuracy using WordCloud is : ", accuracy , "%")
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
df.text = df.text.apply(lemmatize_words)
def join_text(text):

    string = ''

    for i in text:

        string += i.strip() +' '

    return string    
df.text = df.text.apply(join_text)
train_text = df.text[:20000]

train_category = df.label[:20000]

test_text = df.text[20000:]

test_category = df.label[20000:]
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))

#transformed train reviews

cv_train_reviews=cv.fit_transform(train_text)

#transformed test reviews

cv_test_reviews=cv.transform(test_text)



print('BOW_cv_train:',cv_train_reviews.shape)

print('BOW_cv_test:',cv_test_reviews.shape)
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))

#transformed train reviews

tv_train_reviews=tv.fit_transform(train_text)

#transformed test reviews

tv_test_reviews=tv.transform(test_text)

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