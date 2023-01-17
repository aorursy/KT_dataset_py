# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data1=data

data.head()
data.info()
sns.heatmap(data1.isnull(),cmap='Blues')
data1.head()
a=pd.get_dummies(data1['gender'],drop_first=True)

data1=pd.concat([data1,a],axis=1)

data1=data1.rename(columns={'M':'Male or Female'})

data1
b=pd.get_dummies(data['ssc_b'],drop_first=True)

data1=pd.concat([data1,b],axis=1)

data1=data1.rename(columns={'Others':'Secondary Board'})

data1.head(1)
c=pd.get_dummies(data['hsc_b'],drop_first=True)

data1=pd.concat([data1,c],axis=1)

data1=data1.rename(columns={'Others':'Higher Secondary Board'})

data1.head(1)
d=pd.get_dummies(data['hsc_s'],drop_first=True)

data1=pd.concat([data1,d],axis=1)

e=pd.get_dummies(data['degree_t'],drop_first=True)

data1=pd.concat([data1,e],axis=1)

data1=data1.rename(columns={'Others':'Degree other'})

data1.head(1)
f=pd.get_dummies(data['workex'],drop_first=True)

g=pd.get_dummies(data['specialisation'],drop_first=True)

h=pd.get_dummies(data['status'],drop_first=True)

data1=pd.concat([data1,f,g,h],axis=1)

data1.head(1)
data1.info()
data1.drop(['gender','workex','ssc_b','hsc_b','degree_t','status','specialisation','hsc_s','sl_no','salary'],axis=1,inplace=True)
data1.head()
data1.info()
data1.dropna(inplace=True)

sns.heatmap(data1.isnull(),cmap='Blues')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

from sklearn import metrics
data1.dropna(inplace=True)

x=data1.drop('Placed',axis=1)

y=data1['Placed']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101,shuffle=True)
model=LogisticRegression()
model.fit(x_train,y_train)
result=model.predict(x_test)
print(confusion_matrix(y_test,result))
print(classification_report(y_test,result))
metrics.accuracy_score(result,y_test)
cor=data1.corr()
plt.figure(figsize=(14,6))

sns.heatmap(cor,annot=True)
data1.drop(['Mkt&HR','Degree other'],axis=1,inplace=True)
x1=data1.drop('Placed',axis=1)

y1=data1['Placed']

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3,random_state=101,shuffle=True)
updated=LogisticRegression()
updated.fit(x1_test,y1_test)
updated_result=updated.predict(x1_test)
print(classification_report(y1_test,updated_result))
print(confusion_matrix(y1_test,updated_result))
metrics.accuracy_score(updated_result,y1_test)