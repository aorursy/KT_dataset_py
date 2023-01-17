# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
music = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
music.head()
del music['reviewerID']
del music['asin']
del music['reviewerName']
del music['helpful']
del music['unixReviewTime']
del music['reviewTime']
del music['summary']
music=music.dropna()
def sentiment_rating(rating):
    if int(rating)<=3:
        return 0
    else: 
        return 1
music.overall = music.overall.apply(sentiment_rating) 
x_train,x_test,y_train,y_test = train_test_split(music['reviewText'], music.overall, test_size=0.33, random_state=0)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB())])
model = pipe.fit(x_train,y_train)
prediction = model.predict(x_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cv_train_reviews=cv.fit_transform(x_train)
cv_test_reviews=cv.transform(x_test)
mb=MultinomialNB()
mb_bow=mb.fit(cv_train_reviews,y_train)
mb_predict = mb_bow.predict(cv_test_reviews)
score = accuracy_score(y_test,mb_predict)
print("score :",score)
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)
lr_bow = lr.fit(cv_train_reviews,y_train)
lr_predict = lr_bow.predict(cv_test_reviews)
lr_score = accuracy_score(y_test,lr_predict)
print("score :",lr_score)
lr_report=classification_report(y_test,lr_predict,target_names=['0','1'])
print(lr_report)