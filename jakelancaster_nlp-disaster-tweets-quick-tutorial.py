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
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
trainDf = pd.read_csv('../input/nlp-getting-started/train.csv')
trainDf.head()
# Here we use the .info() funtion to show some basic information about the dataset such as the datatypes, value counts & column counts.

trainDf.info()
trainDf['target'].value_counts()
sns.countplot(trainDf['target'])
print('The percentages of each target is:\n',(trainDf['target'].value_counts()/trainDf.shape[0])*100)
trainDf.isnull().sum()
trainDf['keyword'].fillna('no_keyword', inplace=True)
trainDf['location'].fillna('no_location', inplace=True)
trainDf.isna().sum()
trainDf['keyword'].value_counts().plot(kind='bar')
trainDf['keyword'].value_counts()[:10].plot(kind='bar')
trainDf['location'].value_counts()[:10].plot(kind='bar')
trainDf['location'].value_counts()[1:11].plot(kind='bar')
nltk.download('stopwords')
print(stopwords.words('english')[:10])
sents = []
for i in range(0,trainDf.shape[0]):
  tweets = re.sub('[^a-zA-Z]', ' ',trainDf['text'][i])
  tweets = tweets.lower()
  tweets = tweets.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  tweets = [ps.stem(word) for word in tweets if not word in set(all_stopwords)]
  tweets = ' '.join(tweets)
  sents.append(tweets)
sents[:5]
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(sents).toarray()
X[:10]
y=trainDf.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = MultinomialNB()
classifier.fit(X_train, Y_train)
prediction=classifier.predict(X_test)
cm = confusion_matrix(Y_test,prediction)
print(cm)
accuracy_score(Y_test,prediction)
test = pd.read_csv('../input/nlp-getting-started/test.csv')
test.head()
test['keyword'].fillna('no_keyword', inplace=True)
test['location'].fillna('no_location', inplace=True)
test.isnull().sum()
sentsT = []
for i in range(0,test.shape[0]):
  tweets = re.sub('[^a-zA-Z]', ' ',test['text'][i])
  tweets = tweets.lower()
  tweets = tweets.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  tweets = [ps.stem(word) for word in tweets if not word in set(all_stopwords)]
  tweets = ' '.join(tweets)
  sentsT.append(tweets)
sentsT[:5]
cv1 = CountVectorizer(max_features = 1000)
x_test = cv1.fit_transform(sentsT).toarray()
x_test[:10]
Y_test=test.iloc[:,-1].values
print(Y_test)
predictions=classifier.predict(x_test)
print(predictions)
sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.head()
sample_submission.to_csv('sumbission.csv', index=False)
