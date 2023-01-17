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
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import re
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true.head()
fake.head()
true['subject'].value_counts()
true['subject'].value_counts().plot(kind="bar")
fake['subject'].value_counts()
fake['subject'].value_counts().plot(kind="bar")
true['category'] = 1

fake['category'] = 0
df = pd.concat([true,fake]) 
df.head()
df.tail()
df.shape
df['text'].isnull().values.any()
df['category'].value_counts().plot(kind="bar")
num_true = df['category'].value_counts()[1]

num_false = df['category'].value_counts()[0]
category_names = ['True', 'Fake']

sizes = [num_true, num_false]

plt.figure(figsize=(2,2), dpi=220)

plt.pie(sizes, labels = category_names, textprops={'fontsize': 6}, startangle=90, autopct = '%1.1f%%', explode=[0, 0.1])

plt.show()
df['text']
df['text'] = df['text'].apply((lambda y:re.sub("http://\S+"," ", y)))

df['text'] = df['text'].apply((lambda x:re.sub("\@", " ",x.lower())))
df
vectorizer = CountVectorizer(stop_words='english')
%%time



all_news = vectorizer.fit_transform(df.text)
all_news.shape
vectorizer.vocabulary_
X_train, X_test,y_train,y_test = train_test_split(all_news, df.category, test_size=0.3, random_state = 88)
X_train.shape
classifier = MultinomialNB()
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))