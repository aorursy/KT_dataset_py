# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
test.head()
train.head()
train.info()
test.info()
train['target'].describe()
sns.countplot(x = 'target', data = train)

one = train[train['target'] == 1]

zero = train[train['target'] == 0]

print("Target = 0:", zero['target'].count(), "\nTarget = 1:", one['target'].count())
x = train.drop(['target', 'ID_code'], axis = 1)

y = train['target'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0, stratify=y)

logreg = LogisticRegression(C=1)

logreg.fit(x, y)
trainn = train.drop(['ID_code','target'], axis = 1)

testt = test.drop(['ID_code'], axis = 1)
y_pred = logreg.predict(x_test)

print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test, y_pred)

cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
lr = LogisticRegression()

lr.fit(x_train,y_train)

acc = lr.score(x_test,y_test)*100

y_head_lr = lr.predict(x_test)

cm_lr = confusion_matrix(y_test,y_head_lr)

print("Logistic regression accuracy {:.2f}%".format(acc))
logreg_pred_test = logreg.predict_proba(testt)[:,1]

result = test[['ID_code']]

result['target'] = logreg_pred_test

result.head()
result.to_csv('log_reg_baseline.csv', index = False)
nb = GaussianNB()

nb.fit(x_train, y_train)

acc = nb.score(x_test,y_test)*100

y_head_nb = nb.predict(x_test)

cm_nb = confusion_matrix(y_test,y_head_nb)

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
nb_pred_test = nb.predict_proba(testt)[:,1]

result = test[['ID_code']]

result['target'] = nb_pred_test

result.head()
result.to_csv('NB_baseline.csv', index = False)
plt.figure(figsize=(8,2))

plt.subplot(2,3,1)

plt.title("Naive Bayes")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 16})

           

plt.subplot(2,3,2)

plt.title("Logistic Regression")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 16})