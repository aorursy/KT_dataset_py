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
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
data.head()
data['difficulty'].value_counts()
data['target'].value_counts().to_frame().T
pd.crosstab(data['difficulty'], data['target'])
data['ciphertext'].apply(len).describe()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
vectorizer = CountVectorizer(
    analyzer = 'char',
    lowercase = False,
    ngram_range=(1, 6))

estimator = SGDClassifier(loss='hinge', max_iter=1000, random_state=0,
                          tol=1e-3, n_jobs=-1)
model = Pipeline([('selector', 
                   FunctionTransformer(
                       lambda x: x['ciphertext'], validate=False)),
                  ('vectorizer', vectorizer), 
                  ('tfidf', TfidfTransformer()),
                  ('estimator', estimator)])
X = data.drop('target', axis=1)
y = data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import f1_score, classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
f1_score(y_test, y_pred, average='macro')
def get_f1_score(difficulty):
    score = f1_score(y_test[X_test['difficulty'] == difficulty], 
                     y_pred[X_test['difficulty'] == difficulty], average='macro') 
    return score
print("f1_score per difficulty")
for i in range(1, 5):
    print("Difficulty: {} ==> {:5f}".format(i, get_f1_score(i)))
model.fit(X, y)
test_pred = model.predict(test)
subm['Predicted'] = test_pred
subm.to_csv('submission.csv', index=False)