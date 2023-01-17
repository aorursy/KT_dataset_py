# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


d=pd.read_csv("../input/train.csv")
received=pd.to_datetime(d["Date-received"])
sent=pd.to_datetime(d["Date-sent-to-company"])
d=d.drop(columns=["Date-sent-to-company","Date-received","Complaint-ID","Consumer-complaint-summary"])
Y=d["Complaint-Status"]
Y=pd.get_dummies(Y)
d=d.drop(columns="Complaint-Status")
d.head()
d=pd.get_dummies(d)
d['days between received and sent'] = (sent-received).dt.days
d.head()
d=d.drop(columns=["Complaint-reason_Account terms and changes","Complaint-reason_Advertising",
                  "Complaint-reason_Incorrect exchange rate",
                  "Complaint-reason_Problem with an overdraft",
                  "Complaint-reason_Was approved for a loan, but didn't receive the money"])

X=d
clf = DecisionTreeClassifier(criterion = 'entropy')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15)
clf.fit(X_train, y_train)
y_pred =  clf.predict(X_test)
from sklearn.metrics import accuracy_score,f1_score
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
print('f1 weighted Score on train data: ', f1_score(y_true=y_train, y_pred=clf.predict(X_train),average='weighted'))
print('f1 weighted Score on test data: ', f1_score(y_true=y_test, y_pred=y_pred,average='weighted'))

t=pd.read_csv("../input/test.csv")
t_received=pd.to_datetime(t["Date-received"])
t_sent=pd.to_datetime(t["Date-sent-to-company"])
idn=t["Complaint-ID"]
t=t.drop(columns=["Date-sent-to-company","Date-received","Complaint-ID","Consumer-complaint-summary"])
t.head()
t=pd.get_dummies(t)
t['days between received and sent'] = (t_sent-t_received).dt.days
t=t.drop(columns=["Complaint-reason_Can't stop withdrawals from your bank account",
                  "Complaint-reason_Problem with cash advance"])
y_pred =  clf.predict(t)
t.head()

for i in range(5):
    print(y_pred[i])
out = pd.DataFrame({'Closed':y_pred[:,0],'Closed with explanation':y_pred[:,1]
                   ,'Closed with monetary relief':y_pred[:,2],'Closed with non-monetary relief':y_pred[:,3]
                   ,'Untimely response':y_pred[:,4]})
out.head()
out = out.idxmax(axis=1)
out.columns = ['Complaint-Status']
out.name="Complaint-Status"
out.head()
idn.head()
result = pd.concat([idn, out], axis=1)
result.head()
result.to_csv("result.csv")


