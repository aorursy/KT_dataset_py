import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.shape
df.head()
df.describe()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss
df = df.drop(['Time'], axis=1)
from sklearn.preprocessing import StandardScaler
amt = np.array(df['Amount'])

amt = amt.reshape(-1,1)

amt.shape

amt[:10]
sc = StandardScaler()

df['Amount'] = sc.fit_transform(amt)
df.head()
x = df.drop('Class', axis=1)

y = df['Class']

x.shape, y.shape
lr = LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))
smote = SMOTE()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

x_train, y_train = smote.fit_sample(x_train, y_train)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
np.bincount(y_train), np.bincount(y_test)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))
nm = NearMiss()

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

x_train, y_train = nm.fit_sample(x_train, y_train)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
np.bincount(y_train), np.bincount(y_test)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))