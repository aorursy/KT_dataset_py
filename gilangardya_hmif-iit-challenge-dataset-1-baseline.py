import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# from xgboost import XGBClassifier # install dahulu

%matplotlib inline
df = pd.read_csv('../input/train-data-1.csv')
df.columns
plt.title('Proporsi kelas')
sns.countplot(df['akreditasi'])
X = df.drop(['id', 'akreditasi'], axis=1)
y = df['akreditasi']
X_dummy = pd.get_dummies(X) # disc: bukan cara terbaik, ini hanya agar mudah
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.2, random_state=496)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
print(classification_report(y_train, y_train_pred))
print('accuracy', accuracy_score(y_train, y_train_pred))
print('mae', mean_absolute_error(y_train, y_train_pred))
y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred))
print('accuracy', accuracy_score(y_test, y_test_pred))
print('mae', mean_absolute_error(y_test, y_test_pred))
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
clf.fit(X_full, y_full)
y_full_pred = clf.predict(X_full)
print(classification_report(y_full, y_full_pred))
print('accuracy', accuracy_score(y_full, y_full_pred))
print('mae', mean_absolute_error(y_full, y_full_pred))
# test_data = pd.read_csv('../input/test-data-1.csv')
# test_data.head()
# test_data = test_data.drop(['id'], axis=1)
# test_data_dummy = pd.get_dummies(test_data)
# dummy_absent = set(X_full.columns) - set(test_data_dummy.columns)
# for col in dummy_absent:
#     test_data_dummy[col] = 0
# test_data_dummy = test_data_dummy[X_full.columns]
# test_data_dummy.head()
# test_data_pred = clf.predict(test_data_dummy)
# test_data_pred
# submission = pd.read_csv('../input/sample-submission-1.csv')
# submission['akreditasi'] = test_data_pred
# submission.to_csv('submission-1.csv', index=False)
