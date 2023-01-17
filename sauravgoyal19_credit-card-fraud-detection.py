import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")
data.head()
data.tail()
fraud = data.loc[data['Class'] == 1]

normal = data.loc[data['Class'] == 0]
print(fraud)
len(fraud)
len(normal)
sns.relplot(x='Amount',y='Time', hue='Class', data =data)
from sklearn import linear_model

from sklearn.model_selection import train_test_split
x = data.iloc[:,:-1]

y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35)
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(x_train, y_train)
y_pred = np.array(clf.predict(x_test))

y = np.array(y_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y, y_pred))
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
clf.predict(x_test)
pred = np.array(clf.predict(x_test))
ans = []

for i in range(len(pred)):

    if pred[i]==1:

        ans.append("Fraud")

    else:

            ans.append("True")
ans
sns.relplot(x='Amount',y='Time', data=x_test)