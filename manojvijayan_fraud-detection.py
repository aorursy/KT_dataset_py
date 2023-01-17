# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')

df.info()
df.describe().T
df.head(5)
fig, ax = plt.subplots(figsize=(10,5)) 

ax.set_yticks(np.arange(0, 28000, 2500))

ax.set_ylabel("Amount ->")

ax.set_xlabel("Index ->")

x1 = plt.scatter(x=df[(df.Class == 0)].index, y = df[(df.Class == 0)].Amount, alpha=0.4, c='g' )

x2 = plt.scatter(x=df[(df.Class == 1)].index, y = df[(df.Class == 1)].Amount, c='r' )

plt.legend([x1,x2],['G', 'F'])

plt.title('Transaction Amount')
fig, ax = plt.subplots(figsize=(10,5)) 

ax.set_ylabel("Amount ->")

ax.set_xlabel("Index ->")

x1 = plt.scatter(x=df[(df.Class == 0) & (df.Amount >= 2500)].index, y = df[(df.Class == 0) & (df.Amount >= 2500)].Amount, alpha=0.4, c='g' )

x2 = plt.scatter(x=df[(df.Class == 1) & (df.Amount >= 2500)].index, y = df[(df.Class == 1) & (df.Amount >= 2500)].Amount, c='r' )

plt.legend([x1,x2],['G', 'F'])

plt.title('Transaction Amount => 2,500')
fig, ax = plt.subplots(figsize=(10,5)) 

ax.set_yticks(np.arange(0, 2500, 100))

ax.set_ylabel("Amount ->")

ax.set_xlabel("Index ->")

x1 = plt.scatter(x=df[(df.Class == 0) & (df.Amount < 2500)].index, y = df[(df.Class == 0) & (df.Amount < 2500)].Amount, alpha=0.4, c='g' )

x2 = plt.scatter(x=df[df.Class == 1].index, y = df[df.Class == 1].Amount, c='r' )

plt.legend([x1,x2],['G', 'F'])

plt.title('Transaction Amount < 2,500')
df.drop('Time', axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df.Amount = sc.fit_transform(df.Amount.values.reshape(-1, 1))
df_nf = df[df.Class == 0]
y_nf = df_nf.Class
df_nf.drop('Class', axis=1, inplace=True)
df_f = df[df.Class == 1]
y_f = df_f.Class
df_f.drop('Class', axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_nf, y_train, y_nf = train_test_split(df_nf.iloc[0:50000], y_nf[0:50000], test_size=0.30, random_state=101)
X_test = pd.concat([X_nf,df_f])
y_test = pd.concat([y_nf,y_f])
y_test.sort_index(inplace=True)
X_test.sort_index(inplace=True)
from sklearn.ensemble import IsolationForest
isf = IsolationForest(contamination=0.0, bootstrap=True, n_estimators=200)
isf.fit(X_train)
pred = isf.predict(X_test)
pred_t =[]

count = 0

for each in pred:

    if each == -1:

        count = count + 1

        pred_t.append(0)

    else:

        pred_t.append(1)
set(pred_t)
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, pred_t)
print(classification_report(y_test, pred_t))
print("SKEW :-" , df.Class.skew())
df_nf = df[df.Class == 0].sample(frac=0.005, random_state=101)
df_nf.shape
y_nf = df_nf.Class
df_nf.drop('Class', axis=1, inplace=True)
df_new = pd.concat([df_nf, df_f])
y_new = pd.concat([y_nf, y_f])
print("SKEW :-" , y_new.skew())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_new, y_new, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=2)
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
pred_prob  = lr.predict_proba(X_test)
pred_prob_C = []

for each in pred_prob:

    if each[0] > .9831:

        pred_prob_C.append(0)

    else:

        pred_prob_C.append(1)
confusion_matrix(y_test, pred_prob_C)
print(classification_report(y_test, pred_prob_C))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
pred_prob_C = rf.predict_proba(X_test)
pred_prob_C = []

for each in pred_prob:

    if each[0] > .9831:

        pred_prob_C.append(0)

    else:

        pred_prob_C.append(1)
confusion_matrix(y_test, pred_prob_C)
print(classification_report(y_test, pred_prob_C))
from sklearn.svm import SVC
svc = SVC(C=3, probability=True)
svc.fit(X_train,y_train)
pred = svc.predict(X_test)
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
pred_prob_C = svc.predict_proba(X_test)
pred_prob_C = []

for each in pred_prob:

    if each[0] > .9831:

        pred_prob_C.append(0)

    else:

        pred_prob_C.append(1)
confusion_matrix(y_test, pred_prob_C)
print(classification_report(y_test, pred_prob_C))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
pred_prob_C = svc.predict_proba(X_test)

pred_prob_C = []

for each in pred_prob:

    if each[0] > .9831:

        pred_prob_C.append(0)

    else:

        pred_prob_C.append(1)
confusion_matrix(y_test, pred_prob_C)
print(classification_report(y_test, pred_prob_C))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
pred = nb.predict(X_test)
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
pred_prob_C = nb.predict_proba(X_test)

pred_prob_C = []

for each in pred_prob:

    if each[0] > .9831:

        pred_prob_C.append(0)

    else:

        pred_prob_C.append(1)
confusion_matrix(y_test, pred_prob_C)
print(classification_report(y_test, pred_prob_C))
y = df.Class
df.drop('Class', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=2)
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))