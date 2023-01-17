# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('../input/fakenewsdetection/news.csv')

df.shape

df.head()
labels = df.label

labels.head()
#Split the dataset for training and testing

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)



#Fit and transform train set and test set

tfidf_train = tfidf_vectorizer.fit_transform(x_train)

tfidf_test = tfidf_vectorizer.transform(x_test)
#Initialize the PassiveAggeressiveClassifier 

pac = PassiveAggressiveClassifier(max_iter = 50)

pac.fit(tfidf_train, y_train)



#Predict on the test set and calculate accuracy 

y_pred = pac.predict(tfidf_test)

score = accuracy_score(y_test, y_pred)

print(f'Accuracy: {round(score*100,2)}%')
result = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

data = result.flatten()
my_labels = ['True Positives','True Negatives','False Positives', 'False Negatives']

plt.pie(data,labels=my_labels,autopct='%1.1f%%')

plt.title('Fake News Confusion Matrix')

plt.axis('equal')

plt.show()