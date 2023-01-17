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
import pandas as pd

df=pd.read_csv('/kaggle/input/reviews/Restaurant_Reviews.tsv',sep='\t')
df.head(5)


# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.



#import nltk

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



# BeautifulSoup libraray

from bs4 import BeautifulSoup 



import re # regex



#keras

from keras.preprocessing.text import one_hot,Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense,Flatten,Embedding,Input,Dropout

from keras.models import Model

from keras.utils import to_categorical

df.isna().sum()
from nltk.corpus import stopwords

def clean_reviews(review):

    

# 1. Removing html tags

    review_text = BeautifulSoup(review).get_text()



# 2. Retaining only alphabets.

    review_text = re.sub("[^a-zA-Z]"," ",review_text)



# 3. Converting to lower case and splitting

    word_tokens= review_text.lower().split()



# 4. Remove stopwords

    stop_words= set(stopwords.words("english"))     

    word_tokens= [w for w in word_tokens if not w in stop_words]



    cleaned_review=" ".join(word_tokens)

    return cleaned_review



df['Review']=df['Review'].apply(clean_reviews)





df.head(5)
blanks = []  # start with an empty list



for i,lb,rv in df.itertuples():  # iterate over the DataFrame

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
from sklearn.model_selection import train_test_split



X = df['Review']

y = df['Liked']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC



# Naïve Bayes:

text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),('clf', MultinomialNB())])



# Linear SVC:

text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf_nb
text_clf_nb.fit(X_train, y_train)
predictions = text_clf_nb.predict(X_test)
from sklearn import metrics

print(metrics.confusion_matrix(y_test,predictions))
# Print a classification report

print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))
text_clf_lsvc.fit(X_train, y_train)
predictions = text_clf_lsvc.predict(X_test)
from sklearn import metrics

print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))