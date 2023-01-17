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
data=pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")

data.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(data=data,x="price_range")
sns.lmplot(data=data,x="clock_speed",y="ram",hue="price_range")
sns.countplot(data=data,x="wifi")
correlation=data.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
x=data[["ram","px_height","px_width"]]

y=data["price_range"]

x.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

rfc=RandomForestClassifier(max_depth=2)

#scores = cross_val_score(clf, iris.data, iris.target, cv=5)

cross_val_score(rfc,x,y,cv=10)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

cross_val_score(dtc,x,y,cv=10)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

selector=SelectKBest(chi2,k=5)

allx=data.loc[:, data.columns != 'price_range']

x_best=selector.fit_transform(allx,y)
x_best.shape
rfc=RandomForestClassifier(max_depth=2)

#scores = cross_val_score(clf, iris.data, iris.target, cv=5)

cross_val_score(rfc,x_best,y,cv=10)
dtc=DecisionTreeClassifier()

cross_val_score(dtc,x_best,y,cv=10)
rfc=RandomForestClassifier(max_depth=2)

#scores = cross_val_score(clf, iris.data, iris.target, cv=5)

cross_val_score(rfc,x_best,y,cv=5)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_best, y, test_size=0.25, random_state=1881)

dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)

ypred=dtc.predict(x_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
test=pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")

test.head()
test=data.loc[:, test.columns != 'id']

xt_best=selector.transform(test)
predictions=dtc.predict(xt_best)
set(predictions)
sifir=0

bir=0

for pred in predictions:

    if pred==0:

        sifir=sifir+1

    else:

        bir=bir+1

print("zero: "+str(sifir))

print("one:"+str(bir))
x_train, x_test, y_train, y_test = train_test_split(x_best, y, test_size=0.25, random_state=1881)

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))

print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))
predictions2=rfc.predict(xt_best)

print(set(predictions))
sifir=0

bir=0

for pred in predictions2:

    if pred==0:

        sifir=sifir+1

    else:

        bir=bir+1

print("zero: "+str(sifir))

print("one:"+str(bir))