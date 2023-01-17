# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")

data.head()
data_meaningful=data[["Country","Sex","Age","Category","Survived"]]

data_meaningful
from sklearn import preprocessing

le_country = preprocessing.LabelEncoder()

data_meaningful["Country"]=le_country.fit_transform(data_meaningful["Country"])
le_country = preprocessing.LabelEncoder()

data_meaningful["Sex"]=le_country.fit_transform(data_meaningful["Sex"])
le_country = preprocessing.LabelEncoder()

data_meaningful["Category"]=le_country.fit_transform(data_meaningful["Category"])
data_meaningful.isnull().sum()
data_meaningful["Survived"].value_counts()
x=data_meaningful.iloc[:,0:4]

y=data_meaningful.iloc[:,4:]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

import sklearn.metrics as metrik

metrik.confusion_matrix(y_pred=ypred,y_true=y_test)
print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from catboost import CatBoostClassifier





catb = CatBoostClassifier(iterations=1000,verbose=0)

catb.fit(x_train,y_train)



ypred=catb.predict(x_test)

import sklearn.metrics as metrik

metrik.confusion_matrix(y_pred=ypred,y_true=y_test)
print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

dtree.fit(x_train,y_train)

ypred=dtree.predict(x_test)

import sklearn.metrics as metrik

metrik.confusion_matrix(y_pred=ypred,y_true=y_test)
from imblearn.over_sampling import BorderlineSMOTE

bsm=BorderlineSMOTE()

x_train_bsm,y_train_bsm=bsm.fit_resample(x_train,y_train)
from catboost import CatBoostClassifier





catb = CatBoostClassifier(iterations=1000,verbose=0)

catb.fit(x_train_bsm,y_train_bsm)



ypred=catb.predict(x_test)

import sklearn.metrics as metrik

metrik.confusion_matrix(y_pred=ypred,y_true=y_test)
print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from imblearn.over_sampling import KMeansSMOTE

kme=KMeansSMOTE(cluster_balance_threshold=0.1)

x_train_kme,y_train_kme=kme.fit_resample(x_train,y_train)



from catboost import CatBoostClassifier





catb = CatBoostClassifier(iterations=1000,verbose=0)

catb.fit(x_train_kme,y_train_kme)



ypred=catb.predict(x_test)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from imblearn.over_sampling import SVMSMOTE

svms=SVMSMOTE()

x_train_svms,y_train_svms=kme.fit_resample(x_train,y_train)



from catboost import CatBoostClassifier





catb = CatBoostClassifier(iterations=1000,verbose=0)

catb.fit(x_train_svms,y_train_svms)



ypred=catb.predict(x_test)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from imblearn.combine import SMOTEENN

combine=SMOTEENN()



x_train_combine,y_train_combine=combine.fit_resample(x_train,y_train)



from catboost import CatBoostClassifier





catb = CatBoostClassifier(iterations=1000,verbose=0)

catb.fit(x_train_combine,y_train_combine)



ypred=catb.predict(x_test)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from imblearn.combine import SMOTEENN

combine=SMOTEENN()



x_train_combine,y_train_combine=combine.fit_resample(x_train,y_train)







rfc= RandomForestClassifier()

rfc.fit(x_train_combine,y_train_combine)



ypred=rfc.predict(x_test)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.classification_report(y_pred=ypred,y_true=y_test))
from imblearn.pipeline import Pipeline

model = Pipeline([

        ('sampling', SMOTEENN()),

        ('classification', RandomForestClassifier())

    ])
model.fit(x_train,y_train)
model.predict(x_test)
model.steps
import pickle

modelpickle=pickle.dump(model,open("smoteenn_randomforest.pkl", 'wb'))
