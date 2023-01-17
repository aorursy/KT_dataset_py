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
messages= pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
messages
messages = messages.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages = messages.rename(columns={"v1":"label", "v2":"message"})
messages
messages.label = messages.label.map( {'spam': 1, 'ham': 0} ).astype(int)
messages
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

wordnet=WordNetLemmatizer()
corpus = []

for i in range(len(messages.message)):

    review = re.sub('[^a-zA-Z]', ' ', messages.message[i])

    review = review.lower()

    review = review.split()

    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
corpus
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 3000)

X = cv.fit_transform(corpus).toarray()
X
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,messages.label,test_size=0.2,random_state=0)
from sklearn.naive_bayes import MultinomialNB

spam_detection_model= MultinomialNB().fit(X_train,Y_train)
y_pred = spam_detection_model.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test,y_pred))
from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(Y_test,y_pred)
from sklearn.linear_model import PassiveAggressiveClassifier

linear_clf = PassiveAggressiveClassifier(max_iter=20)

linear_clf.fit(X_train,Y_train)

y_pred = linear_clf.predict(X_test)

accuracy_score(Y_test,y_pred)