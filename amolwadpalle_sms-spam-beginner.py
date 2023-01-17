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
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding = 'latin-1')

data.head()
cols_to_del = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']

data.drop(cols_to_del, axis = 1, inplace = True)
data.rename(columns = {'v1' : 'type', 'v2' : 'text'}, inplace = True)

data.head()
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

stemer = PorterStemmer()

stop_words = stopwords.words('english')

clean_text = []

for i in range(0, len(data.text)):

    review = re.sub('[^a-zA-Z]',' ', data.text[i])

    review = review.lower()

    review = review.split()

    review = [stemer.stem(word) for word in review if word not in set(stop_words)]

    review = ' '. join(review)

    clean_text.append(review)

# create Bag of Words

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(max_features = 5000)

X = cv.fit_transform(clean_text).toarray()
data['spam'] = data['type'].map({'spam' :1, 'ham' : 0})

data.head()
y = data.spam

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import MultinomialNB

my_nmb = MultinomialNB()

my_nmb.fit(X_train, y_train)

predictions = my_nmb.predict(X_test)
from sklearn.metrics import accuracy_score



print (accuracy_score(y_test, predictions))
from sklearn.metrics import confusion_matrix

confusion_m = confusion_matrix(y_test,predictions)

confusion_m