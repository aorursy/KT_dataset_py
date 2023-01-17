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
# data loading

data = pd.read_csv('/kaggle/input/Restaurant_Reviews.tsv', delimiter= '\t', quoting = 3)

data.head()
data.info()


data.isna().any()
plt.style.use('seaborn')

sns.countplot(x = 'Liked', data = data)

plt.show()
data['Liked'].value_counts() # balanced class attribute
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
corpus = []

ps = PorterStemmer()



for i in range(data.shape[0]):

    text = re.sub(pattern= '[^a-zA-Z]', repl= '', string= data['Review'][i])

    text = text.lower()

    text = text.split()

    text = [words for words in text if words not in set(stopwords.words('english'))]

    text = [ps.stem(words) for words in text]

    text = ' '.join(text)

    corpus.append(text)
corpus[:10]
# split the data

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 1500)

X = cv.fit_transform(corpus).toarray()

y = data.iloc[:, -1].values



    
# train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# model building

from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train, y_train)
# model evaluation with cross validation score

from sklearn.model_selection import cross_val_score

for i in [5,10]:

    CV_score = cross_val_score(estimator= MultinomialNB(), X = X_train,y = y_train, cv = i )

    print('CV score: {} for cv = {}'.format(CV_score, i))
# model evaluation with model.score

print('traing score: {}'.format(nb_classifier.score(X_train, y_train)))

print('testing score: {}'.format(nb_classifier.score(X_test, y_test)))
# predictions

y_pred = nb_classifier.predict(X_test)
# model performance metrics

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
y_test[:10]
y_pred[:10]