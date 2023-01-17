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
import numpy as np

import pandas as pd

import re

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

import string

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from tqdm import tqdm_notebook as tqdm
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)
train.head()
train.isnull().sum()
train['target'].value_counts()
print('%no insincere questions{:.2f}'.format(train['target'].value_counts()[0]/len(train)))

print('% insincere questions{:.2f}'.format(train['target'].value_counts()[1]/len(train)))
cv = TfidfVectorizer(sublinear_tf = True,stop_words = 'english', ngram_range = (1,2), max_features = 4000, token_pattern = '(\S+)')
X_train, X_val, y_train, y_val = train_test_split(train["question_text"], train['target'], test_size = 0.25)
df_train = cv.fit_transform(X_train)

df_val = cv.transform(X_val)

df_test = cv.transform(test["question_text"])
lSvc = LinearSVC()

lSvc.fit(df_train,y_train)

lSvc.score(df_train,y_train)
lSvc.score(df_val,y_val)
logreg = LogisticRegression()

logreg.fit(df_train,y_train)

logreg.score(df_train,y_train)
y_pred = logreg.predict(df_test)
submission = test

submission.head()
submission['prediction'] = y_pred

submission.prediction = submission.prediction.astype(int)

submission.drop('question_text', axis = 1, inplace = True)

submission.head()
submission.to_csv('submission.csv', index=False)