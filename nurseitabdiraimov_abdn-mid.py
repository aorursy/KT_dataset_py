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

dataset = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

dataset1 = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
type(dataset)

dataset.head(5)
type(dataset1)
dataset1.head(5)
dataset.shape
dataset1.shape
dataset.info()
dataset1.info()
dataset.isna().sum()
dataset1.isna().sum()
dataset.nunique()
dataset1.nunique()
dataset.dropna(inplace=True)
dataset1.dropna(inplace=True)
X = dataset1.iloc[:,4:8].values

X
y = dataset1.iloc[:,1].values

y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1,stratify =y)

X_train.shape
import seaborn as sns

sns.heatmap(dataset1.isnull(), xticklabels = False)
sns.countplot(dataset1.target)
from sklearn.neighbors import KNeighborsClassifier

knn_cls = KNeighborsClassifier()

knn_cls.fit(X_train, y_train)
y_preds = knn_cls.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

accuracy_score(y_test, y_preds)
cm = confusion_matrix(y_test, y_preds)

cm
import matplotlib.pyplot as plt

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
print(classification_report(y_test, y_preds, target_names = ['target', 'var_0']))
for i in np.arange(1,95,3):

    knn_cls = KNeighborsClassifier(i)

    knn_cls.fit(X_train, y_train)

    print('k: ',i,' acc: ',accuracy_score(y_test,knn_cls.predict(X_test)))
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

print(X_train)
print(X_test)
X.mean()
X.std()
sns.boxplot(x='target', y = 'var_0', data = dataset1)
sns.distplot(dataset1[dataset1.target == 0]['var_0'], hist = False)

sns.distplot(dataset1[dataset1.target == 1]['var_1'], hist = False)
f, axes = plt.subplots(2, 2, figsize=(16, 12), sharex = True)

sns.distplot(dataset1[dataset1.target == 0]['var_0'], label = 'var0',hist = False, ax = axes[0,0])

sns.distplot(dataset1[dataset1.target == 1]['var_0'], label = 'var0',hist = False, ax = axes[0,0])



sns.distplot(dataset1[dataset1.target == 0]['var_1'], label = 'var1', hist = False, ax = axes[0,1])

sns.distplot(dataset1[dataset1.target == 1]['var_1'], label = 'var1', hist = False, ax = axes[0,1])



sns.distplot(dataset1[dataset1.target == 0]['var_2'], label = 'var2', hist = False, ax = axes[1,0])

sns.distplot(dataset1[dataset1.target == 1]['var_2'], label = 'var2', hist = False, ax = axes[1,0])



sns.distplot(dataset1[dataset1.target == 0]['var_3'], label = 'var3', hist = False, ax = axes[1,1])

sns.distplot(dataset1[dataset1.target == 1]['var_3'], label = 'var3', hist = False, ax = axes[1,1])
dataset.corr()
dataset1.corr()
from sklearn.naive_bayes import GaussianNB
g_nb_cls = GaussianNB()
g_nb_cls.fit(X_train, y_train)
y_preds = g_nb_cls.predict(X_test)
try:

    print(classification_report(y_test, y_preds))

except ZeroDivisionError:

    print(0)
from sklearn.utils.testing import ignore_warnings

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
st_sc = StandardScaler()
X_sc = st_sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 123, stratify = y)

print('X_train shape is :', X_train.shape, '\ny_train shape is ',y_train.shape)
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
y_preds_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_preds_xgb))
train_inp = dataset1.drop(columns = ['target', 'ID_code'])

test_inp = dataset.drop(columns = ['ID_code'])
X_train, X_test, y_train,  y_test = train_test_split(train_inp, dataset1.target,test_size=0.5, random_state=0)

X_train.shape

X_test.shape

y_train.shape

y_test.shape
from sklearn.linear_model import LogisticRegression

logist = LogisticRegression(class_weight='balanced')

logist.fit(X_train, y_train)
logist_pred = logist.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

def performance(Y_test, logist_pred):

    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]

    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)

    print('AUC:')

    print(auc(fpr, tpr))
performance(y_test, logist_pred)
logist_pred_test = logist.predict_proba(test_inp)[:,1]

submit = dataset[['ID_code']]

submit['target'] = logist_pred_test

submit.head()
submit.to_csv('abd_n.csv', index = False)