# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline  
bank=pd.read_csv("../input/UniversalBank.csv")

bank.head()
del bank['ID']

del bank["ZIP Code"]

bank.head()
sns.countplot(data=bank,x="CreditCard")
sns.countplot(data=bank,x="Personal Loan")
set(bank["Education"])
set(bank["Family"])
bank.describe()
correlation=bank.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation,annot=True)
!apt-get remove swig

!apt-get install swig3.0

!ln -s /usr/bin/swig3.0 /usr/bin/swig

!curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

!pip install auto-sklearn
import autosklearn.classification
bank.tail()
y=bank["Personal Loan"]

x= bank.drop('Personal Loan', axis=1)

x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1176)

autocls=autosklearn.classification.AutoSklearnClassifier( time_left_for_this_task=100, per_run_time_limit=30)

autocls.fit(x_train,y_train)

ypred=autocls.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))

print(autocls.show_models())
ypred=autocls.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1176)

autocls=autosklearn.classification.AutoSklearnClassifier( time_left_for_this_task=400, per_run_time_limit=30)

autocls.fit(x_train,y_train)

ypred=autocls.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1176)

autocls=autosklearn.classification.AutoSklearnClassifier( time_left_for_this_task=700, per_run_time_limit=40)

autocls.fit(x_train,y_train)

ypred=autocls.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))
autocls.show_models()
utocls.show_models()