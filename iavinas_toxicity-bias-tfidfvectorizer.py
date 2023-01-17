import numpy as np 

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv' , index_col='id')

test = pd.read_csv('../input/test.csv' , index_col='id')

train.head()
Vect = TfidfVectorizer()

label = np.where(train['target'] >= 0.5 , 1, 0)

train = Vect.fit_transform(train['comment_text'])

test = Vect.transform(test['comment_text'])
print(train.shape , test.shape , label.shape)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(train , label)

pre = clf.predict_proba(test)
sub = pd.read_csv('../input/sample_submission.csv')
sub['prediction'] = pre[:,1]

sub.to_csv('submission.csv', index=False)
pre