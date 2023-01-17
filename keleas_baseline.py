import numpy as np 

import pandas as pd 

import warnings

warnings.simplefilter('ignore')
train_x = pd.read_csv('../input/train_x.csv', index_col=0, header=None)

train_y = pd.read_csv('../input/train_y.csv', index_col=0)

test_x = pd.read_csv('../input/test_x.csv', index_col=0, header=None)
## посмотрим как выглядят метки



train_y.head(3)
mappping_type = {'Bird': 0, 'Airplane': 1}

train_y = train_y.replace({"target": mappping_type})
train_y.head(3)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
%%time



parameters = {'penalty':('l1', 'l2'), 'C':np.linspace(0.01, 1, 10)}

clf = LogisticRegression(n_jobs=-1)

lr_clf = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')

lr_clf.fit(train_x[:100], train_y[:100])
lr_clf.best_estimator_
best_lr_clf = LogisticRegression(penalty='l1', C=0.12)



best_lr_clf.fit(train_x[:100], train_y[:100])

predict_y = best_lr_clf.predict_proba(test_x)
sample = pd.DataFrame(np.array([[i, x.argmax()] for i, x in enumerate(predict_y)]), columns=['id', 'target'])



mappping_type_inv = {0: 'Bird', 1: 'Airplane'}

sample = sample.replace({'target': mappping_type_inv})
sample.head()
sample.to_csv('submit.csv', index=False)