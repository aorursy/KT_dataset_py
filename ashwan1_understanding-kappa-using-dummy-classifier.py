import numpy as np

import pandas as pd

from sklearn.dummy import DummyClassifier

from sklearn.metrics import cohen_kappa_score



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train.head()
dummy_clf = DummyClassifier() # Default strategy is 'stratified'

dummy_clf.fit(train.id_code, train.diagnosis) # Inputs doesn't matter, it's dummy

train_predictions = dummy_clf.predict(train.id_code)

print(f"Score: {dummy_clf.score(train.id_code, train.diagnosis)}")

print(f"Cohen kappa score: {cohen_kappa_score(train_predictions, train.diagnosis, weights='quadratic')}")
test = pd.read_csv('../input/test.csv')

test.head()
predictions = dummy_clf.predict(test.id_code)

submissions = pd.read_csv('../input/sample_submission.csv')

submissions['diagnosis'] = predictions

submissions.head()
submissions.to_csv('submission.csv', index=False)