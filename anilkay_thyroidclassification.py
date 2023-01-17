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
data=pd.read_csv("/kaggle/input/hypothyroid/hypothyroid.csv")

data.head()
data=data.rename(columns={data.columns[0]:"target"})
data["target"].value_counts()
data["target"]=data["target"].map({"negative":0,"hypothyroid":1})
for column in data.columns:

    howmany=len(set(data[column]))

    print(column,": ",howmany)
data["pregnant"].value_counts()
data=data.replace({"t":1,"f":0})
data["T3_measured"].value_counts()
data=data.replace({"y":1,"n":0})
data
data["TBG"].value_counts()
del data["TBG"]
data
data=data.replace({"?":np.NAN})
data.isnull().sum()
data.dtypes
cols = data.columns[data.dtypes.eq('object')]

data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

data.dtypes
y=data.iloc[:,0:1]

x=data.iloc[:,1:]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
from  xgboost import XGBClassifier

import sklearn.metrics as metrik

xgbo = XGBClassifier(learning_rate=0.01)





xgbo.fit(x_train,y_train)

ypred=xgbo.predict(x_test)





print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
print(metrik.f1_score(y_pred=ypred,y_true=y_test))
from catboost import CatBoostClassifier

catboost1= CatBoostClassifier(max_depth=3,verbose=0)

catboost1.fit(x_train,y_train)

ypred=catboost1.predict(x_test)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.f1_score(y_pred=ypred,y_true=y_test))
from sklearn.model_selection import cross_val_score

xgbo = XGBClassifier(learning_rate=0.01)

cross_val_score(xgbo,x, y, cv=7, scoring='f1')
from sklearn.model_selection import cross_val_score

catboost1= CatBoostClassifier(max_depth=3,verbose=0)

cross_val_score(catboost1,x, y, cv=7, scoring='f1')