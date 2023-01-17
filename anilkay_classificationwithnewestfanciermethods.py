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
data=pd.read_csv("../input/heart-disease-uci/heart.csv")

data.head()
%matplotlib inline

import seaborn as sns
x=data.iloc[:,0:13]

y=data.iloc[:,13:]
from sklearn.model_selection import cross_val_score

from catboost import CatBoostClassifier

cat=CatBoostClassifier(iterations=5)

cross_val_score(cat,x,y,cv=3)
cat=CatBoostClassifier(iterations=50)

cross_val_score(cat,x,y,cv=3)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
cat=CatBoostClassifier(iterations=150,learning_rate=1)

cat.fit(X_train,y_train)
ypred=cat.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=20)

gb.fit(X_train,y_train)

ypred=gb.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=10)

gb.fit(X_train,y_train)

ypred=gb.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(X_train,y_train)

ypred=rfc.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.decomposition import PCA

pca=PCA(n_components=3)

x_pca=pca.fit_transform(x)
from sklearn.model_selection import train_test_split

xp_train, xp_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=42)
cat=CatBoostClassifier(iterations=150,learning_rate=1)

cat.fit(xp_train,y_train)

ypred=cat.predict(xp_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=10)

gb.fit(xp_train,y_train)

ypred=gb.predict(xp_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(xp_train,y_train)

ypred=rfc.predict(xp_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.decomposition import PCA

pca=PCA(n_components=10)

x_pca=pca.fit_transform(x)
from sklearn.model_selection import train_test_split

xp_train, xp_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=42)

cat=CatBoostClassifier(iterations=150,learning_rate=1)

cat.fit(xp_train,y_train)

ypred=cat.predict(xp_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=10)

gb.fit(xp_train,y_train)

ypred=gb.predict(xp_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(xp_train,y_train)

ypred=rfc.predict(xp_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
import category_encoders as ce

count_encode=ce.CountEncoder(cols=["sex","cp","slope","thal"])

x_count=count_encode.fit_transform(x)



from sklearn.model_selection import train_test_split

Xco_train, Xco_test, y_train, y_test = train_test_split(x_count, y, test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(Xco_train,y_train)

ypred=rfc.predict(Xco_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=10)

gb.fit(Xco_train,y_train)

ypred=gb.predict(Xco_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
from sklearn.model_selection import cross_val_score

from catboost import CatBoostClassifier

cat=CatBoostClassifier(iterations=5)

cross_val_score(cat,x_count,y,cv=3)
cat=CatBoostClassifier(verbose=0)

cat.fit(Xco_train,y_train)

ypred=cat.predict(Xco_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))