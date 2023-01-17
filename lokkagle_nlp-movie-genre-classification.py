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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/kaggle_movie_train.csv')

data.head()
data.info()
data.drop(columns= 'id', inplace= True)

data.head()
data.isna().any()
data['genre'].value_counts()
plt.style.use('seaborn')

data['genre'].value_counts().plot(kind = 'bar')

plt.show()


genre_mapping = {'other': 0, 'action': 1, 'romance': 2, 'horror': 3, 'sci-fi': 4, 'comedy': 5,'thriller': 6, 'drama': 7,'adventure': 8}

genre_mapping
data['genre'] = data['genre'].map(genre_mapping)

data.head()
data['text'][0]
import nltk

nltk.download('stopwords')

import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
corpus = []

ps = PorterStemmer()



for i in range(0, data.shape[0]):

    text = re.sub(pattern= '[^a-zA-Z]', repl= ' ', string= data['text'][i])

    text = text.lower()

    text = text.split()

    text = [ words for words in text if words not in set(stopwords.words('english'))]

    text = [ps.stem(words)  for words in text]

    text = ' '.join(text)

    corpus.append(text)
corpus[:5]
genre_mapping
scifi_words = []

romance_words = []

thriller_words = []



for i in list(data[data['genre']==4].index):

    scifi_words.append(corpus[i])



for i in list(data[data['genre']==2].index):

    romance_words.append(corpus[i])



for i in list(data[data['genre']==6].index):

    thriller_words.append(corpus[i])



scifi = ''

romance = ''

thriller = ''

for i in range(0, 3):

    scifi += scifi_words[i]

    romance += romance_words[i]

    thriller += thriller_words[i]
from wordcloud import WordCloud

import matplotlib.pyplot as plt



wc = WordCloud(background_color='white', width=3000, height=2500).generate(scifi)

plt.figure(figsize=(8,8))

plt.imshow(wc)

plt.axis('off')

plt.show()


wc = WordCloud(background_color='white', width=3000, height=2500).generate(romance)

plt.figure(figsize=(8,8))

plt.imshow(wc)

plt.axis('off')

plt.show()
wc = WordCloud(background_color='white', width=3000, height=2500).generate(thriller)

plt.figure(figsize=(8,8))

plt.imshow(wc)

plt.axis('off')

plt.show()
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=10000, ngram_range=(1,2))

X = cv.fit_transform(corpus).toarray()


y = data['genre'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = nb_classifier.predict(X_test)
# Calculating Accuracy

from sklearn.metrics import accuracy_score

score_ = accuracy_score(y_test, y_pred)

print("Accuracy score is: {}%".format(round(score_*100,2)))
y_test[:5]
y_pred[:5]
# model performance

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test, y_pred))