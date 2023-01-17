import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd



import nationality_helpers

from nationality_helpers import create_top_medalist



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import tree



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline
df = create_top_medalist()
df.shape
df.nationality.nunique()
df.groupby('sport')['country_count'].agg('count')
df.describe()
train, test = train_test_split(df, test_size=.3, random_state=123, stratify=df[['nationality']])
y_train = train[['nationality']]

y_test = test[['nationality']]
X_train = train[['sex', 'height', 'weight', 'age', 'sport']]

X_test = test[['sex', 'height', 'weight', 'age', 'sport']]
X_train = train[['sex', 'height', 'weight', 'age']]

X_test = test[['sex', 'height', 'weight', 'age']]
clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=123).fit(X_train, y_train)
y_pred = clf.predict(X_train)

y_pred_proba = clf.predict_proba(X_train)
print('Accuracy of Decision Tree classifier on training set: {:.6f}'

     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on training set: {:.6f}'

     .format(clf.score(X_test, y_test)))