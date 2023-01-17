# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head(5)
from sklearn.feature_extraction.text import CountVectorizer

sample_texts = ['мама мыла раму', 'я люблю мама', 'раму мыла я']
count_vec = CountVectorizer()
sample_texts_vectorized = count_vec.fit_transform(sample_texts)
sample_texts_vectorized
# это наши тексты
print(sample_texts)
# это имена колонок 
print(count_vec.get_feature_names())
# Это вывод нашего векторайзера
print(sample_texts_vectorized.toarray())
%%time

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train['Text'])

X_train
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='sag', max_iter=20)

# solver - чтобы было быстрее, но в ущерб качеству
# max_iter - чтобы было быстрее, но в ущерб качеству
%%time

classifier.fit(X_train, train['Sentiment'])
test = pd.read_csv('../input/test.csv')

X_test = vectorizer.transform(test['Text'])
X_test
predictions = classifier.predict_proba(X_test)
predictions
# первый столбец - вероятность первого класса - негативного
# второй столбец - вероятность второго класса - позитивного
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
submission['Sentiment'] = predictions[:, 1]
submission.head()
submission.to_csv('simple_submission_baseline.csv', index=False)

