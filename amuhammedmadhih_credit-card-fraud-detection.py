import pandas as pd

import seaborn as sns

import numpy as np



filename = "/kaggle/input/creditcardfraud/creditcard.csv"

data = pd.read_csv(filename)

data.head()
fraud = data.loc[data['Class'] == 1]

normal = data.loc[data['Class'] == 0]

fraud.sum()
len(fraud)
len(normal)

sns.relplot(x = "Amount", y = "Time", hue="Class", data = data)
from sklearn import linear_model

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
x = data.iloc[:,:-1]

y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.35)
clf = linear_model.LogisticRegression(C=1e5)

clf.fit(x_train, y_train)
y_pred = np.array(clf.predict(x_test))

y=np.array(y_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y, y_pred))

print(classification_report(y_test, y_pred))