import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier



import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
train_df.drop(columns=['ID_code', 'target'], inplace=True)

test_df.drop(columns=['ID_code'], inplace=True)
train_df['is_test'] = 0

test_df['is_test'] = 1

df = pd.concat([train_df, test_df], axis = 0)

X = df.drop(columns=['is_test'])

y = df['is_test']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

print('train2 shape:', X_train.shape, 'test2 shape:', X_test.shape)
#Let's use simple Random Forest as Classifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

y_test_score = rfc.predict_proba(X_test)
import scikitplot as skplt

import matplotlib.pyplot as plt



skplt.metrics.plot_roc_curve(y_test, y_test_score)

plt.show()
#AUC score is about 0.5

print('AUC score: ', round(roc_auc_score(y_true=y_test, y_score=y_test_score[:,1]), 4))