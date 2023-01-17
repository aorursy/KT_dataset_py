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
# loading data

data = pd.read_csv('/kaggle/input/kaggle_fake_train.csv')

data.head()
# lets build the model on title and label

data_filtered = data.copy(deep= True)

data_filtered.drop(columns = ['id', 'author', 'text'],axis = 1, inplace = True)

data_filtered.head()
# visualizing target class

plt.style.use('seaborn')

sns.countplot(x = 'label', data = data_filtered)

plt.show()
# checking for missing values

data_filtered.isna().sum()
# dropping missing values

data_filtered.dropna(inplace= True)

data_filtered.isna().sum()
print('original data shape: {}'.format(data.shape))

print('shape of the data after handling nulls : {}'.format(data_filtered.shape))
data_filtered.head()
data_filtered.reset_index(inplace = True)

data_filtered.head()
import nltk

import re

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
corpus = []

ps = PorterStemmer()



for i in range(0,data_filtered.shape[0]):



  # Cleaning special character from the news-title

  title = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data_filtered.title[i])



  # Converting the entire news-title to lower case

  title = title.lower()



  # Tokenizing the news-title by words

  words = title.split()



  # Removing the stopwords

  words = [word for word in words if word not in set(stopwords.words('english'))]



  # Stemming the words

  words = [ps.stem(word) for word in words]



  # Joining the stemmed words

  title = ' '.join(words)



  # Building a corpus of news-title

  corpus.append(title)
corpus[:10]
# bag of words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 1000, ngram_range= (1,3))

X = cv.fit_transform(corpus).toarray()

X
print(X.ndim)

print(X.shape)
y = data_filtered['label']

y
# train and test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
X_test
# model

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

model = nb.fit(X_train,y_train)


print('training score: {}'.format(model.score(X_train, y_train)))

print('testing score: {}'.format(model.score(X_test, y_test)))
# predictions

y_pred = model.predict(X_test)
# performance metrics

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

cm
# other way

pd.crosstab(y_test, y_pred, rownames=['actual'], colnames= ['predicted'])
print(classification_report(y_test, y_pred))
# model accuracy

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))


print(y_test[:10])

print(y_pred[:10])