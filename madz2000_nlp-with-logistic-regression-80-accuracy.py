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
#importing the training data

df=pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')

df.head()
#Summary of the dataset

df.describe()
def strip_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text()



#Removing the square brackets

def remove_between_square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)

# Removing URL's

def remove_between_square_brackets(text):

    return re.sub(r'http\S+', '', text)

#Removing the noisy text

def denoise_text(text):

    text = strip_html(text)

    text = remove_between_square_brackets(text)

    return text

#Apply function on review column

df['Text']=df['Text'].apply(denoise_text)
plt.figure(figsize = (10,10))

corr = df.corr()

sns.heatmap(corr , mask=np.zeros_like(corr, dtype=np.bool) , cmap=sns.diverging_palette(-100,0,as_cmap=True) , square = True)
del df['Id']

del df['Time']

del df['UserId']

del df['ProductId']

del df['HelpfulnessNumerator']

del df['HelpfulnessDenominator']
df.head()
df['Text'] = df['Text'] + ' ' + df['Summary'] + ' ' + df['ProfileName']

del df['Summary']

del df['ProfileName']

df.head()
def score_sentiment(score):

    # Replace scores of 1,2,3 with 0 (not good) and 4,5 with 1 (good)

    if(score == 1 or score == 2 or score == 3):

        return 0

    else:

        return 1

df.Score = df.Score.apply(score_sentiment)    
df.head()
df.isna().sum()
df.Text.fillna("",inplace = True)
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
stemmer = PorterStemmer()

def stem_text(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            word = stemmer.stem(i.strip())

            final_text.append(word)

    return " ".join(final_text)
df.Text = df.Text.apply(stem_text)
plt.figure(figsize = (20,20)) # Text with bad/average reviews

wc = WordCloud(max_words = 3000 , width = 1600 , height = 800).generate(" ".join(df[df.Score == 0].Text))

plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text with good reviews

wc = WordCloud(max_words = 3000 , width = 1600 , height = 800).generate(" ".join(df[df.Score ==1].Text))

plt.imshow(wc , interpolation = 'bilinear')
x_train,x_test,y_train,y_test = train_test_split(df.Text,df.Score,random_state = 0)
cv=CountVectorizer(min_df=0,max_df=1,ngram_range=(1,2))

#transformed train reviews

cv_train_reviews=cv.fit_transform(x_train)

#transformed test reviews

cv_test_reviews=cv.transform(x_test)



print('BOW_cv_train:',cv_train_reviews.shape)

print('BOW_cv_test:',cv_test_reviews.shape)
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)

#Fitting the model for Bag of words

lr_bow=lr.fit(cv_train_reviews,y_train)

print(lr_bow)
#Predicting the model for bag of words

lr_bow_predict=lr.predict(cv_test_reviews)
#Accuracy score for bag of words

lr_bow_score=accuracy_score(y_test,lr_bow_predict)

print("lr_bow_score :",lr_bow_score)
#Classification report for bag of words

lr_bow_report=classification_report(y_test,lr_bow_predict,target_names=['0','1'])

print(lr_bow_report)