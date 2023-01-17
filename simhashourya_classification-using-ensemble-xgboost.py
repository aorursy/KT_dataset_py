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
data = pd.read_csv('/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv')

data.head()
data['TenYearCHD'].value_counts()
data.isna().sum()
data.dropna(axis=0,inplace=True)

data['TenYearCHD'].value_counts()
data.info()
import matplotlib.pyplot as plt

#Gender vs TenYearCHD

x=pd.crosstab(data.male,data.TenYearCHD,margins=True)/data.shape[0]

x
data.shape[0]
from sklearn.model_selection import train_test_split

y=data.pop('TenYearCHD')

X=data

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,shuffle=True,stratify=y)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train =ss.fit_transform(X_train)

X_test =ss.fit_transform(X_test)


import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier



lr=LogisticRegression(C=2,solver='liblinear')

lr.fit(X_train,y_train)

lr_y_pred=lr.predict(X_test)

print("The accuracy is %f" %accuracy_score(y_test,lr_y_pred))

confusion_matrix(y_test,lr_y_pred)
svc=SVC(C=1,kernel='linear')

svc.fit(X_train,y_train)

svc_y_pred=svc.predict(X_test)

print("The accuracy is %f" %accuracy_score(y_test,svc_y_pred))

confusion_matrix(y_test,svc_y_pred)
dc = DecisionTreeClassifier()

dc.fit(X_train,y_train)

y_pred=dc.predict(X_test)

print("The accuracy is %f" %accuracy_score(y_test,y_pred))

confusion_matrix(y_test,y_pred)
rf=RandomForestClassifier(max_depth=100,criterion='entropy',max_features='log2')

rf.fit(X_train,y_train)

rf_y_pred=rf.predict(X_test)

print("The accuracy is %f" %accuracy_score(y_test,rf_y_pred))

confusion_matrix(y_test,rf_y_pred)
xg=  XGBClassifier(n_estimators=50,booster='dart',learning_rate=0.07)

xg.fit(X_train,y_train)

xg_y_pred = xg.predict(X_test)

print("The accuracy is %f" %accuracy_score(y_test,xg_y_pred))

confusion_matrix(y_test,xg_y_pred)
pd.DataFrame(xg_y_pred).to_csv('submission')