# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split

from sklearn import metrics 

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split

from sklearn import metrics 

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

sam = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
train.head()
train.dtypes
train.shape, test.shape, sam.shape
print(train.isnull().sum())
train_corr = train.iloc[2:10, 2:10].corr()
plt.figure(figsize=(10,10))

sns.heatmap(train_corr)
target = train['target']

train = train.drop(["ID_code", "target"], axis=1)

sns.set_style('darkgrid')

sns.countplot(target)
features = ["var_{}".format(i) for i in range(200)]

plt.figure(figsize=[16,9])

sns.heatmap(train[features].corr())
plt.figure(figsize=(15,5))

features = train.columns.values[2:202]

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(15,5))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=1),color="black", kde=True,bins=120, label='train')

sns.distplot(test[features].std(axis=1),color="red", kde=True,bins=120, label='test')

plt.legend();plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per column in the train and test set")

sns.distplot(train[features].std(axis=0),color="blue",kde=True,bins=120, label='train')

sns.distplot(test[features].std(axis=0),color="green", kde=True,bins=120, label='test')

plt.legend(); plt.show()
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

sam = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
X = train.drop(columns=['target','ID_code'])

y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()

logist_pred = logreg.predict_proba(test.drop(columns=['ID_code']))

len(logist_pred)

sample = pd.DataFrame({'ID_code':test['ID_code'], 'target': logist_pred[:, 1]})
logist_pred[:, 1]
sample = pd.DataFrame({'ID_code':test['ID_code'], 'target': logist_pred[:, 1]})
sample.to_csv('logreg_.csv', index = False)
clf = DecisionTreeClassifier()



clf = clf.fit(X_train,y_train)



y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print(classification_report(y_test, y_pred))
Acc = accuracy_score(y_test, y_pred)

print("Accuracy: " + str(Acc))
cm = confusion_matrix
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
logist_pred = clf.predict_proba(test.drop(columns=['ID_code']))

sample = pd.DataFrame({'ID_code':test['ID_code'], 'target': logist_pred[:, 1]})

sample.to_csv('decission_tree.csv', index = False)
model = GaussianNB()

model.fit(X_train, y_train)

predicted= model.predict(X_test)

print("NBGaussian Accuracy :", accuracy_score(y_test, predicted))
roc_auc_score(y_test, predicted)
proba = model.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_test, proba)

fpr, tpr, _  = roc_curve(y_test, proba)



plt.figure()

plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")

plt.plot([0, 1], [0, 1], color='m', linestyle='--')

plt.title("Results")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()
logist_pred = clf.predict_proba(test.drop(columns=['ID_code']))

sample = pd.DataFrame({'ID_code':test['ID_code'], 'target': logist_pred[:, 1]})

sample.to_csv('naive_bayes.csv', index = False)