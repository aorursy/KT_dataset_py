# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



%matplotlib notebook

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data.drop('sl_no',axis=1,inplace=True)

data.head()
data.shape
data.describe()
data.isnull().sum()
data.status.value_counts()
data.gender.value_counts()
plt.figure()

sns.countplot(data.gender,hue=data.status)

plt.show()
data.ssc_b.value_counts()
sns.countplot('ssc_b',hue='status',data=data)



plt.show()
sns.kdeplot(data.ssc_p[data.status=='Placed'])

sns.kdeplot(data.ssc_p[data.status=='Not Placed'])

plt.legend(['Placed','Not Placed'])

plt.xlabel('Secondary Education Percentage')

plt.show()
sns.kdeplot(data.hsc_p[data.status=='Placed'])

sns.kdeplot(data.hsc_p[data.status=='Not Placed'])

plt.legend(['Placed','Not Placed'])

plt.xlabel('Higher Education Percentage')

plt.show()
data.hsc_b.value_counts()
sns.countplot('hsc_b',hue='status',data=data)

plt.show()
data.hsc_s.value_counts()
sns.countplot('hsc_s',hue='status',data=data)

plt.show()
sns.kdeplot(data.degree_p[data.status=='Placed'])

sns.kdeplot(data.degree_p[data.status=='Not Placed'])

plt.legend(['Placed','Not Placed'])

plt.xlabel('Degree Percentage')

plt.show()
data.degree_t.value_counts()
sns.countplot('degree_t',hue='status',data=data)

plt.show()
data.workex.value_counts()
sns.countplot('workex',hue='status',data=data)

plt.show()
sns.kdeplot(data.etest_p[data.status=='Placed'])

sns.kdeplot(data.etest_p[data.status=='Not Placed'])

plt.legend(['Placed','Not Placed'])

plt.xlabel('Employability Test Percentage')

plt.show()
data.specialisation.value_counts()
sns.countplot('specialisation',hue='status',data=data)

plt.show()
sns.kdeplot(data.mba_p[data.status=='Placed'])

sns.kdeplot(data.mba_p[data.status=='Not Placed'])

plt.legend(['Placed','Not Placed'])

plt.xlabel('MBA Percentage')

plt.show()
data['gender']=data.gender.map({'M':0,'F':1})

data['hsc_s']=data.hsc_s.map({'Commerce':0,'Science':1,'Arts':2})

data['degree_t']=data.degree_t.map({'Comm&Mgmt':0,'Sci&Tech':1,'Others':2})

data['workex']=data.workex.map({'Yes':0,'No':1})

data['specialisation']=data.specialisation.map({'Mkt&HR':0,'Mkt&Fin':1})

data['status']=data.status.map({'Placed':1,'Not Placed':0})

data['ssc_b']=data.ssc_b.map({'Central':0,'Others':1})

data['hsc_b']=data.hsc_b.map({'Central':0,'Others':1})
data.head()
cor=data.corr()

plt.figure(figsize=(14,8))

sns.heatmap(cor,annot=True)
features=['gender','ssc_b','ssc_p','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','mba_p']

X=data[features]

y=data['status']

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)

scaler=StandardScaler()

X_train_sc=scaler.fit_transform(X_train)

X_test_sc=scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()

clf.fit(X_train_sc,y_train)

predictions=clf.predict(X_test_sc)

score=accuracy_score(y_test,predictions)

score
from sklearn.svm import SVC

clf=SVC()

clf.fit(X_train_sc,y_train)

predictions=clf.predict(X_test_sc)

score=accuracy_score(y_test,predictions)

score
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(random_state=0)

tree.fit(X_train_sc,y_train)

predictions=tree.predict(X_test_sc)

score=accuracy_score(y_test,predictions)

score
from sklearn.ensemble import RandomForestClassifier

rfclf=RandomForestClassifier(random_state=0,n_estimators=100)

rfclf.fit(X_train_sc,y_train)

predictions=rfclf.predict(X_test_sc)

score=accuracy_score(y_test,predictions)

score
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()

model.fit(X_train_sc,y_train)

predictions=model.predict(X_test_sc)

score=accuracy_score(y_test,predictions)

score
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()

model.fit(X_train_sc,y_train)

predictions=model.predict(X_test_sc)

score=accuracy_score(y_test,predictions)

score