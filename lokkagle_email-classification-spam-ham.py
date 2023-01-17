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
data = pd.read_csv('/kaggle/input/spam or ham.csv', engine= 'python')

data.head()
data.info()
sns.countplot(x = 'mail type', data= data)

plt.show()
data['message'][:5]
data.isna().any()
data['mail type'].value_counts()
import nltk

nltk.download('stopwords')

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
corpus = []

ps = PorterStemmer()



for i in range(data.shape[0]):

    text = re.sub(pattern= '[^A-Za-z]', repl= ' ', string = data['message'][i])

    text = text.lower()

    text = text.split()

    words = [words for words in text if words not in set(stopwords.words('english'))]

    words = [ps.stem(word) for word in words]

    words = ' '.join(words)

    corpus.append(words)

    
corpus[:10] 
# bag of words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 3000)

X = cv.fit_transform(corpus).toarray()

X
# extracting dependent variable 

y = pd.get_dummies(data['mail type'])

y
y = y.iloc[:, 0].values

y
# train and test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
# model building

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)
print('training score :{}'.format(nb.score(X_train, y_train)))

print('testing score :{}'.format(nb.score(X_test, y_test)))
y_pred = nb.predict(X_test)
# error metrics

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

print(cm)
# other way of seeing confusion matrix

pd.crosstab(y_test, y_pred, rownames=['actual values'], colnames=['predicted values'])
print(classification_report(y_test, y_pred))
## tfidf approach

# bag of words model

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range= (2,3), max_features= 3000)

X = cv.fit_transform(corpus).toarray()

X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
nb = MultinomialNB()

nb.fit(X_train, y_train)
print('training score :{}'.format(nb.score(X_train, y_train)))

print('testing score :{}'.format(nb.score(X_test, y_test)))
y_pred = nb.predict(X_test)
# other way of seeing confusion matrix

pd.crosstab(y_test, y_pred, rownames=['actual values'], colnames=['predicted values'])