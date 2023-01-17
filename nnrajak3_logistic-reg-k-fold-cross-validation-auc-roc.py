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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv('/kaggle/input/wine-quality-by-uci/winequality-red.csv', delimiter=';')

dataset2 = pd.read_csv('/kaggle/input/wine-quality-by-uci/winequality-white.csv', delimiter=';')

dataset.head()
dataset2.head()
dataset = pd.concat([dataset, dataset2], ignore_index=True)
dataset.info()
dataset['quality'] = dataset['quality']>5
dataset['quality'] = dataset['quality'].astype(int)
dataset.describe()
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
ns_probs = [0 for i in range(len(y_test))]
lr_probs = classifier.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_auc_score, roc_curve
ns_auc = roc_auc_score(y_test, ns_probs)

lr_auc = roc_auc_score(y_test, lr_probs)

print('No skill AUC : ', ns_auc)

print('Logistic Regression AUC : ', lr_auc)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No skill AUC : {}'.format(ns_auc))

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic AUC : {0:.2f}'.format(lr_auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.plot()