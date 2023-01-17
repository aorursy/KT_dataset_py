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

from keras.models import Sequential

from keras.layers import Dense,LSTM
df = pd.read_csv("../input/kindle-reviews/kindle_reviews.csv")
df.head()
# Leaving the text columns , finding the correlation between different features and their influence on score 

plt.figure(figsize = (10,10))

corr = df.corr()

sns.heatmap(corr , mask=np.zeros_like(corr, dtype=np.bool) , cmap=sns.diverging_palette(-100,0,as_cmap=True) , square = True)
del df['Unnamed: 0']

del df['asin']

del df['helpful']

del df['reviewTime']

del df['reviewerID']

del df['reviewerName']

del df['unixReviewTime']
df.isna().sum()
df.overall.count()
df.head()
df['reviewText'] = df['reviewText'] + ' ' + df['summary']

del df['summary']
df.isna().sum()
df['reviewText'].fillna("",inplace = True)
df.head()
df.overall.value_counts()
def review_sentiment(rating):

    # Replacing rating of 1,2,3 with 0(not good) and 4,5 with 1(good) 

    if(rating == 1 or rating == 2 or rating == 3):

        return 0

    else:

        return 1
df.overall = df.overall.apply(review_sentiment)
df.head()
df.overall.value_counts()
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def clean_text(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            word = i.strip().lower()

            final_text.append(word)

    return " ".join(final_text) 
df['reviewText'] = df['reviewText'].apply(clean_text)
x_train,x_test,y_train,y_test = train_test_split(df.reviewText,df.overall , random_state = 0)
cv=CountVectorizer(min_df=0,max_df=1,ngram_range=(1,1))

#transformed train reviews

cv_train_reviews=cv.fit_transform(x_train)

#transformed test reviews

cv_test_reviews=cv.transform(x_test)



print('BOW_cv_train:',cv_train_reviews.shape)

print('BOW_cv_test:',cv_test_reviews.shape)
mnb = MultinomialNB()

mnb.fit(cv_train_reviews,y_train)
mnb_pred = mnb.predict(cv_test_reviews)

accuracy_score(y_test,mnb_pred)
cv_report = classification_report(y_test,mnb_pred,target_names = ['0','1'])

print(cv_report)
cm_cv = confusion_matrix(y_test,mnb_pred)

cm_cv
cm_cv = pd.DataFrame(cm_cv, index=[0,1], columns=[0,1])

cm_cv.index.name = 'Actual'

cm_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')