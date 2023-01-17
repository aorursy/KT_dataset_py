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

import base64

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

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

from matplotlib.pyplot import imread
from os import path

from PIL import Image
df = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
df.head()
df.isna().sum()
df.fillna("",inplace = True)
df.head()
d = path.dirname(_file_) if "_file_" in locals() else os.getcwd()

img_coloring = np.array(Image.open(path.join("../input/clothing2/","clothing2.jpg")))
x = df['Review Text'].values

wc = WordCloud(background_color = 'white',max_words = 2000, mask = img_coloring,max_font_size = 40,stopwords = STOPWORDS,random_state = 42,width = 1600 ,height = 800).generate(" ".join(x))
image_colors = ImageColorGenerator(img_coloring)

plt.figure(figsize = (20,20))

plt.imshow(wc.recolor(color_func = image_colors), interpolation="bilinear") # WordCloud in the form of a shirt
df['Text'] = df['Review Text'] + df['Title'] + df['Division Name'] + df['Department Name'] + df['Class Name']

del df['Review Text']

del df['Title']

del df['Division Name']

del df['Department Name']

del df['Class Name']
df.head()
df['Recommended IND'].value_counts()
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
df.Text = df.Text.apply(lemmatize_words)
def join_text(text):

    return " ".join(text)
df.Text = df.Text.apply(join_text)
train_message,test_message,train_category,test_category = train_test_split(df.Text,df['Recommended IND'])
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))

#transformed train reviews

cv_train_reviews=cv.fit_transform(train_message)

#transformed test reviews

cv_test_reviews=cv.transform(test_message)



print('CV_train:',cv_train_reviews.shape)

print('CV_test:',cv_test_reviews.shape)
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))

#transformed train reviews

tv_train_reviews=tv.fit_transform(train_message)

#transformed test reviews

tv_test_reviews=tv.transform(test_message)

print('Tfidf_train:',tv_train_reviews.shape)

print('Tfidf_test:',tv_test_reviews.shape)
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)

#Fitting the model for Bag of words

lr_cv=lr.fit(cv_train_reviews,train_category)

#Fitting the model for tfidf features

lr_tfidf=lr.fit(tv_train_reviews,train_category)
#Predicting the model for bag of words

lr_cv_predict=lr.predict(cv_test_reviews)

##Predicting the model for tfidf features

lr_tfidf_predict=lr.predict(tv_test_reviews)
#Accuracy score for bag of words

lr_cv_score=accuracy_score(test_category,lr_cv_predict)

print("lr_bow_score :",lr_cv_score)

#Accuracy score for tfidf features

lr_tfidf_score=accuracy_score(test_category,lr_tfidf_predict)

print("lr_tfidf_score :",lr_tfidf_score)
lr_cv_report = classification_report(test_category,lr_cv_predict,target_names=['0','1'])

print(lr_cv_report)

lr_tfidf_report = classification_report(test_category,lr_tfidf_predict,target_names=['0','1'])

print(lr_tfidf_report)
cm_lr_cv = confusion_matrix(test_category,lr_cv_predict)

cm_lr_cv
cm_lr_cv = pd.DataFrame(cm_lr_cv, index=[0,1], columns=[0,1])

cm_lr_cv.index.name = 'Actual'

cm_lr_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_lr_cv,cmap= "Blues",annot = True, fmt='')
cm_tfidf_lr = confusion_matrix(test_category,lr_tfidf_predict)

cm_tfidf_lr
cm_lr_tfidf = pd.DataFrame(cm_tfidf_lr, index=[0,1], columns=[0,1])

cm_lr_tfidf.index.name = 'Actual'

cm_lr_tfidf.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_lr_tfidf,cmap= "Blues",annot = True, fmt='')
#training the model

mnb=MultinomialNB()

#fitting the nb for bag of words

mnb_cv=mnb.fit(cv_train_reviews,train_category)

#fitting the nb for tfidf features

mnb_tfidf=mnb.fit(tv_train_reviews,train_category)
#Predicting the model for bag of words

mnb_cv_predict=mnb.predict(cv_test_reviews)

#Predicting the model for tfidf features

mnb_tfidf_predict=mnb.predict(tv_test_reviews)
#Accuracy score for bag of words

mnb_cv_score=accuracy_score(test_category,mnb_cv_predict)

print("mnb_cv_score :",mnb_cv_score)

#Accuracy score for tfidf features

mnb_tfidf_score=accuracy_score(test_category,mnb_tfidf_predict)

print("mnb_tfidf_score :",mnb_tfidf_score)
mnb_cv_report = classification_report(test_category,mnb_cv_predict,target_names=['0','1'])

print(mnb_cv_report)

mnb_tfidf_report = classification_report(test_category,mnb_tfidf_predict,target_names=['0','1'])

print(mnb_tfidf_report)
cm_cv_mnb = confusion_matrix(test_category,mnb_cv_predict)

cm_cv_mnb
cm_mnb_cv = pd.DataFrame(cm_cv_mnb, index=[0,1], columns=[0,1])

cm_mnb_cv.index.name = 'Actual'

cm_mnb_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_mnb_cv,cmap= "Blues",annot = True, fmt='')
cm_tfidf_mnb = confusion_matrix(test_category,mnb_tfidf_predict)

cm_tfidf_mnb
cm_mnb_tfidf = pd.DataFrame(cm_tfidf_mnb, index=[0,1], columns=[0,1])

cm_mnb_tfidf.index.name = 'Actual'

cm_mnb_tfidf.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_mnb_tfidf,cmap= "Blues",annot = True, fmt='')