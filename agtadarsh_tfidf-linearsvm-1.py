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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import contractions
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.question_text = train.question_text.apply(lambda x: contractions.fix(x))
test.question_text = test.question_text.apply(lambda x:contractions.fix(x))
tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words='english')
X_train_dtm = tfidf.fit_transform(train.question_text)
X_test_dtm = tfidf.transform(test.question_text)
X_test_dtm
#Linear SVC
print("\nLinear SVC")
logreg = LinearSVC()
# train the model using X_train_dtm
%time logreg.fit(X_train_dtm, train.target)
# make class predictions for X_test_dtm
y_pred = logreg.predict(X_test_dtm)
submission = test
submission.head()

submission['prediction'] = y_pred
submission.prediction = submission.prediction.astype(int)
submission.drop('question_text', axis = 1, inplace = True)
submission.head()
submission.to_csv('submission.csv', index=False)
