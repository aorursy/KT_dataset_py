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
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

train_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = 'english')

X_train = tfidf.fit_transform(train_df['text'])

X_test = tfidf.transform(test_df['text'])

X_train[0].todense()

y = train_df['target']

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(knn, X_train, y, cv = 5)

print("cross validation scores: \n {}".format(scores))
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



lgreg = LogisticRegression(solver = 'lbfgs')

scores = cross_val_score(lgreg, X_train, y, cv = 5)

print("Logreg cros_val_score: \n {}".format(scores))



svc = SVC(gamma = 'auto')

score = cross_val_score(svc, X_train, y, cv = 5)

score
rf = RandomForestClassifier(max_depth = 5, n_estimators = 100)

score = cross_val_score(rf, X_train, y, cv = 5)

score
lgreg = LogisticRegression(solver = 'lbfgs')

lgreg.fit(X_train, y)



sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission['target'] = lgreg.predict(X_test)

sample_submission.head()
sample_submission.to_csv("submission.csv", index = False)