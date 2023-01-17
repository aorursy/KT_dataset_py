# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,confusion_matrix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.shape
df.head()
df.info()
df.describe().T
plt.figure(figsize=(10,6))

df.corr()['salary'][:-1].sort_values().plot.bar()
sns.lmplot('mba_p', 'salary', data=df)
sns.lmplot('etest_p', 'salary', data=df)
plt.figure(figsize=(10,6))

sns.heatmap(df.corr(),annot=True,cmap='viridis')
sns.countplot(df['status'])
sns.distplot(df['salary'])
sns.countplot(df['gender'])
sns.countplot(df['gender'],hue=df['status'])
sns.countplot(df['gender'],hue=df['workex'])
df['ssc_p'].plot.hist(bins=50)
fig,ax=plt.subplots(2,2,figsize=(16,8))

_=sns.countplot(df['ssc_b'],ax=ax[0,0])

_=sns.countplot(df['ssc_b'],hue=df['gender'],ax=ax[0,1])

_=sns.countplot(df['ssc_b'],hue=df['workex'],ax=ax[1,0])

_=sns.countplot(df['ssc_b'],hue=df['status'],ax=ax[1,1])
plt.figure(figsize=(12,5))

sns.lineplot(df['ssc_p'],df['salary'],hue=df['ssc_b'])
df['hsc_p'].plot.hist(bins=60)
fig,ax=plt.subplots(2,2,figsize=(16,8))

_=sns.countplot(df['hsc_b'],ax=ax[0,0])

_=sns.countplot(df['hsc_b'],hue=df['gender'],ax=ax[0,1])

_=sns.countplot(df['hsc_b'],hue=df['workex'],ax=ax[1,0])

_=sns.countplot(df['hsc_b'],hue=df['status'],ax=ax[1,1])
fig,ax=plt.subplots(1,3,figsize=(16,5))

_=sns.countplot(df['hsc_s'],ax=ax[0])

_=sns.countplot(df['hsc_s'],hue=df['hsc_b'],ax=ax[1])

_=sns.countplot(df['hsc_s'],hue=df['status'],ax=ax[2])
plt.figure(figsize=(12,5))

sns.lineplot(df['hsc_p'],df['salary'],hue=df['hsc_b'])
df['degree_p'].plot.hist(bins=60)
fig,ax=plt.subplots(2,2,figsize=(16,8))

_=sns.countplot(df['degree_t'],ax=ax[0,0])

_=sns.countplot(df['degree_t'],hue=df['gender'],ax=ax[0,1])

_=sns.countplot(df['degree_t'],hue=df['workex'],ax=ax[1,0])

_=sns.countplot(df['degree_t'],hue=df['status'],ax=ax[1,1])
plt.figure(figsize=(12,5))

sns.lineplot(df['degree_p'],df['salary'],hue=df['degree_t'])
df['etest_p'].plot.hist(bins=30)
plt.figure(figsize=(12,5))

sns.lineplot(df['etest_p'],df['salary'],hue=df['workex'])
df['mba_p'].plot.hist(bins=60)
fig,ax=plt.subplots(2,2,figsize=(16,8))

_=sns.countplot(df['specialisation'],ax=ax[0,0])

_=sns.countplot(df['specialisation'],hue=df['gender'],ax=ax[0,1])

_=sns.countplot(df['specialisation'],hue=df['workex'],ax=ax[1,0])

_=sns.countplot(df['specialisation'],hue=df['status'],ax=ax[1,1])
plt.figure(figsize=(12,5))

sns.lineplot(df['mba_p'],df['salary'],hue=df['workex'])
df.drop(['sl_no','ssc_b','hsc_b','salary'],axis=1,inplace=True)
df['gender']=df['gender'].map({'M':1,'F':0})

df['hsc_s']=df['hsc_s'].map({'Commerce':1,

                            'Science':0,

                            'Arts':2})

df['degree_t']=df['degree_t'].map({'Sci&Tech':0,

                                  'Comm&Mgmt':1,

                                  'Others':2})

df['workex']=df['workex'].map({'Yes':1,

                              'No':0})

df['specialisation']=df['specialisation'].map({'Mkt&HR':0,

                                              'Mkt&Fin':1})

df['status']=df['status'].map({'Placed':1,

                              'Not Placed':0})
df=pd.get_dummies(df,columns=['hsc_s','degree_t'],

                   drop_first=True)
X=df.drop('status',axis=1)

y=df['status']
scaler=StandardScaler()

X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
regressor=LogisticRegression()

regressor.fit(X_train,y_train)

prediction=regressor.predict(X_test)

acc_lr=accuracy_score(y_test,prediction)

print('Accuracy score is {}'.format(acc_lr))

print(confusion_matrix(y_test,prediction))
classifier_knn=KNeighborsClassifier()

classifier_knn.fit(X_train,y_train)

prediction=classifier_knn.predict(X_test)

acc_knn=accuracy_score(y_test,prediction)

print('Accuracy score is {}'.format(acc_knn))

print(confusion_matrix(y_test,prediction))
classifier_dtr=DecisionTreeClassifier()

classifier_dtr.fit(X_train,y_train)

prediction=classifier_dtr.predict(X_test)

acc_dtr=accuracy_score(y_test,prediction)

print('Accuracy score is {}'.format(acc_dtr))

print(confusion_matrix(y_test,prediction))
classifier_rfc=RandomForestClassifier()

classifier_rfc.fit(X_train,y_train)

prediction=classifier_rfc.predict(X_test)

acc_rfc=accuracy_score(y_test,prediction)

print('Accuracy score is {}'.format(acc_rfc))

print(confusion_matrix(y_test,prediction))
classifier_xgbc=XGBClassifier()

classifier_xgbc.fit(X_train,y_train)

prediction=classifier_xgbc.predict(X_test)

acc_xgbc=accuracy_score(y_test,prediction)

print('Accuracy score is {}'.format(acc_xgbc))

print(confusion_matrix(y_test,prediction))
classifier_gnb=GaussianNB()

classifier_gnb.fit(X_train,y_train)

prediction=classifier_gnb.predict(X_test)

acc_gnb=accuracy_score(y_test,prediction)

print('Accuracy score is {}'.format(acc_gnb))

print(confusion_matrix(y_test,prediction))
classifier_svc=SVC(probability=True)

classifier_svc.fit(X_train,y_train)

prediction=classifier_svc.predict(X_test)

acc_svc=accuracy_score(y_test,prediction)

print('Accuracy score is {}'.format(acc_svc))

print(confusion_matrix(y_test,prediction))
model=pd.DataFrame({'Model':['LogisticRegression','KNeighborsClassifier',

                             'RandomForestClassifier','DecisionTreeClassifier'

                            ,'SVM','GaussianNB','XGBClassifier'],

                  'Score':[acc_lr,acc_knn,acc_rfc,acc_dtr,acc_svc,acc_gnb,acc_xgbc]})

model.sort_values('Score',ascending=False)