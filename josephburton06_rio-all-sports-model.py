import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd



import sport_helpers

from sport_helpers import create_top_sport

from sport_helpers import create_enc



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn import tree



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline
df = create_top_sport()
create_enc(df, ['nationality', 'sex', 'sport'])
df = df[df.sport != 'athletics']
train, test = train_test_split(df, test_size=.3, random_state=123, stratify=df[['sport_enc']])
y_train = train[['sport_enc']]

y_test = test[['sport_enc']]
X_train = train[['sex_enc', 'height', 'weight', 'age', 'nationality_enc']]

X_test = test[['sex_enc', 'height', 'weight', 'age', 'nationality_enc']]
clf = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=123).fit(X_train, y_train)
y_pred = clf.predict(X_train)

y_pred_proba = clf.predict_proba(X_train)
print('Accuracy of Decision Tree classifier on training set: {:.6f}'

     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.6f}'

     .format(clf.score(X_test, y_test)))