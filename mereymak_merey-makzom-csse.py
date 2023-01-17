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

from sklearn.utils.testing import ignore_warnings

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost import XGBClassifier

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

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import roc_auc_score



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
test.info()

train.info()
train.isnull().sum()
train['target'].value_counts(normalize=True)
plt.figure(figsize=(10, 7))

sns.countplot(train['target'])
train.shape
test.shape
train.describe()
X = train.drop(['target', 'ID_code'], axis = 1)

y = train['target'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4,random_state=15, stratify=y)

reg = LogisticRegression(C=1)

reg.fit(X, y)
y_pred = reg.predict(X_test)
print('accuracy is: ',accuracy_score(y_test, y_pred))

print((classification_report(y_test,y_pred)))
cm = confusion_matrix(y_test, y_pred)

cm
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
gn = GaussianNB()

gn.fit(X_train,y_train)
y_pred = gn.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
bn = BernoulliNB()

bn.fit(X_train,y_train)
trainn = train.drop(['ID_code','target'], axis = 1)

testt = test.drop(['ID_code'], axis = 1)
y_pred = bn.predict(X_test)
print('accuracy is   : ',accuracy_score(y_test, y_pred))

print('roc auc score : ',roc_auc_score(y_test, y_predm))

print(classification_report(y_test, y_pred))
xgb = XGBClassifier()
X_test_ch = test[test.columns[1:]]

y_test_ch = test[test.columns[:1]]
reg_pred_test = reg.predict_proba(testt)[:,1]

final = test[['ID_code']]

final['target'] = reg_pred_test

final.head()
final.to_csv('log_reg_baseline.csv', index = False)
bn_pred_test = bn.predict_proba(testt)[:,1]

final = test[['ID_code']]

final['target'] = bn_pred_test

final.head()
final.to_csv('log_reg_baseline.csv', index = False)