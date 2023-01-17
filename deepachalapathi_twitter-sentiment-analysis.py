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

import matplotlib.pyplot as plt

import pandas as pd

dataset_train = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')

dataset_test = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/test.csv')
X_train = dataset_train['tweet']

y_train = dataset_train['label']

X_test = dataset_test['tweet']
print(X_test[0])
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
print(len(dataset_test))
def clean_text(dataset):

    corpus = []

    for i in range(len(dataset)):

        review = re.sub('[^a-zA-Z]',' ',dataset[i])

        review = review.lower().split()



        le = WordNetLemmatizer()



        all_stopwords = stopwords.words('english')

        review = [le.lemmatize(word) for word in review if not word in set(all_stopwords)]

        review = ' '.join(review)

        corpus.append(review)

    return corpus

X_train = clean_text(X_train)

X_test = clean_text(X_test)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary = True)

X_train = cv.fit_transform(X_train).toarray()

X_test = cv.transform(X_test).toarray()
from sklearn.model_selection import train_test_split

x_train, x_test, Y_train, Y_test = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(x_train, Y_train)
y_pred = classifier.predict(x_test)

print (len(y_pred))

#print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(Y_test, y_pred)

print(cm)

accuracy_score(Y_test, y_pred)
pred_Y = classifier.predict(X_test)



dataset_test['pred_label'] = pred_Y



dataset_test[dataset_test.pred_label == 1] #print the rows with predict 1


