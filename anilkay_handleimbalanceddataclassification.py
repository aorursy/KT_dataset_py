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
data=pd.read_csv("/kaggle/input/success-of-bank-telemarketing-data/Alpha_bank.csv")

data.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(data=data,x="Marital_Status")
plt.figure(figsize=(13,13))

sns.countplot(data=data,x="Education")
sns.countplot(data=data,x="Subscribed")
x=data.iloc[:,0:7]

y=data.iloc[:,7:]
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

X=enc.fit_transform(x)

X.shape
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

seletK=SelectKBest(chi2, k=20)

x_train_new = seletK.fit_transform(x_train, y_train)

x_test_new=seletK.transform(x_test)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train_new,y_train)

ypred=rfc.predict(x_test_new)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
majority=data[data["Subscribed"]=="no"]

minority=data[data["Subscribed"]=="yes"]
from sklearn.utils import resample

minority_upsampled = resample(minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=15000,    # to match majority class

                                 random_state=123) # reproducible results
data_upsampled=pd.concat([majority,minority_upsampled])
sns.countplot(data=data_upsampled,x="Subscribed")
x=data_upsampled.iloc[:,0:7]

y=data_upsampled.iloc[:,7:]

X=enc.fit_transform(x)

print(X.shape)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


seletK=SelectKBest(chi2, k=100)

x_train_new = seletK.fit_transform(x_train, y_train)

x_test_new=seletK.transform(x_test)
rfc=RandomForestClassifier()

rfc.fit(x_train_new,y_train)

ypred=rfc.predict(x_test_new)

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
majority=data[data["Subscribed"]=="no"][0:5000]

minority=data[data["Subscribed"]=="yes"]
data_downsampled=pd.concat([majority,minority])
sns.countplot(data=data_downsampled,x="Subscribed")
x=data_downsampled.iloc[:,0:7]

y=data_downsampled.iloc[:,7:]

X=enc.fit_transform(x)

print(X.shape)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


seletK=SelectKBest(chi2, k=20)

x_train_new = seletK.fit_transform(x_train, y_train)

x_test_new=seletK.transform(x_test)



rfc=RandomForestClassifier()

rfc.fit(x_train_new,y_train)

ypred=rfc.predict(x_test_new)

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
x=data.iloc[:,0:7]

y=data.iloc[:,7:]

enc = OneHotEncoder(handle_unknown='ignore')

X=enc.fit_transform(x)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X, y)
from sklearn.model_selection import cross_val_score

print(cross_val_score(RandomForestClassifier(), X_res, y_res, cv=3))
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

ypred=svm.predict(x_test)

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

maxaccu=0

bestk=0

for i in range(1,23,2):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    ypred=knn.predict(x_test)

    accuracy=metrik.accuracy_score(y_pred=ypred,y_true=y_test)

    print("K'nin Değeri: ",i,accuracy)

    if accuracy>maxaccu:

        bestk=i

        maxaccu=accuracy

print("Max Accuracy: ",maxaccu)

print("Best K: ",bestk)

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=2,max_time_mins=138)

tpot.fit(x_train.toarray(), y_train)

ypred=tpot.predict((x_test.toarray()))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
ypred=tpot.predict((x_test.toarray()))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))