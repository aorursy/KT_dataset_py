!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@nlp
import sys

sys.path.append('/kaggle/working')
%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex3 import *

print("\nSetup complete")
review_data = pd.read_csv('../input/nlp-course/yelp_ratings.csv', index_col=0)

review_data.head()
import spacy



# Need to load the large model to get the vectors

nlp = spacy.load('en_core_web_lg')
# We just want the vectors so we can turn off other models in the pipeline

with nlp.disable_pipes():

    vectors = np.array([nlp(review.text).vector for idx, review in review_data[:100].iterrows()])
# Loading all document vectors from file

vectors = np.load('../input/nlp-course/review_vectors.npy')
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment, test_size=0.1, random_state=1)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



model = LogisticRegression(random_state=1)

model.fit(X_train, y_train)



accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

print(accuracy)
from sklearn.linear_model import LogisticRegressionCV

c_vals = [0.1, 1, 10, 100, 1000, 10000]

model = LogisticRegressionCV(Cs=c_vals, scoring='accuracy',

                             cv=5, max_iter=10000,

                             random_state=1).fit(X_train, y_train)
print(f'Model test accuracy: {model.score(X_test, y_test)}')
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
scores, models = [], []

for c in c_vals:

    clf = LinearSVC(C=c, random_state=1, dual=False, max_iter=10000)

    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

    scores.append(cv_score.mean())

    models.append(clf)
max_score = np.array(scores).argmax()

model = models[max_score].fit(X_train, y_train)

print(f'Model test accuracy: {model.score(X_test, y_test)}')