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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/weatherAUS.csv')

df.head()
df = df.drop(columns=['Sunshine','WindDir9am','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date','WindGustDir',"WindGustSpeed","WindDir9am"],axis=1)

df.shape
df = df.dropna(how='any')

df.shape
from scipy import stats

z = np.abs(stats.zscore(df._get_numeric_data()))

print(z)

df= df[(z < 3).all(axis=1)]

print(df.shape)

df.head()
#Lets deal with the categorical cloumns now

# simply change yes/no to 1/0 for RainToday and RainTomorrow

df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
df.head()

df = df.drop(columns=['WindDir3pm'],axis=1)

df.shape
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(df)

df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

df.iloc[4:10]
from sklearn.feature_selection import SelectKBest, chi2

X = df.loc[:,df.columns!='RainTomorrow']

y = df[['RainTomorrow']]

selector = SelectKBest(chi2, k=4)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)])
df = df[['Humidity9am','Rainfall','RainToday','RainTomorrow', 'Humidity3pm']]

X = df[['Humidity3pm','Rainfall']] # let's use only one feature Humidity3pm

y = df[['RainTomorrow']]
#LogisticRegression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import time



t0=time.time()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf_logreg = LogisticRegression(random_state=0)

clf_logreg.fit(X_train,y_train)

y_pred = clf_logreg.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)

#Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



t0=time.time()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)

clf_rf.fit(X_train,y_train)

y_pred = clf_rf.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



t0=time.time()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf_dt = DecisionTreeClassifier(random_state=0)

clf_dt.fit(X_train,y_train)

y_pred = clf_dt.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)