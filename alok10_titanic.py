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
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
data=pd.read_csv('../input/titanic/train.csv')
data.head()
data.isnull().sum()
sns.heatmap(data.isnull())
sns.countplot(x='Survived',data=data)
sns.countplot(x='Survived',data=data,hue='Sex')
sns.countplot(x='Survived',data=data,hue='Pclass')
data.info()
data.drop('Cabin',axis=1,inplace=True)
data.dropna(inplace=True)
sns.heatmap(data.isnull())
data.isnull().sum()
sex=pd.get_dummies(data['Sex'],drop_first=True)

sex.head()
embarked=pd.get_dummies(data['Embarked'],drop_first=True)

embarked.head()
pclass=pd.get_dummies(data['Pclass'],drop_first=True)

pclass.head()
data=pd.concat([data,sex,embarked,pclass],axis=1)
data.head()
data.drop('Name',axis=1,inplace=True)
data.drop(['PassengerId','Pclass','Embarked','Sex','Ticket'],inplace=True,axis=1)
data.head()
x_train=data.drop("Survived",axis=1)

y_train=data["Survived"]
## Creating the Model
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
tdata=pd.read_csv('../input/titanic/test.csv')
tdata.head()
tdata.isnull().sum()
tdata.drop('Cabin',axis=1,inplace=True)
tdata['Age']=tdata['Age'].fillna(tdata['Age'].mean())
tdata['Fare']=tdata['Fare'].fillna(tdata['Fare'].mean())
tdata.isnull().sum()
sex=pd.get_dummies(tdata['Sex'],drop_first=True)

sex.head()
embarked=pd.get_dummies(tdata['Embarked'],drop_first=True)

embarked.head()
pclass=pd.get_dummies(tdata['Pclass'],drop_first=True)

pclass.head()
tdata.head()
tdata=pd.concat([tdata,sex,embarked,pclass],axis=1)
tdata.head()
tdata.drop('Name',axis=1,inplace=True)

tdata.drop(['PassengerId','Pclass','Embarked','Sex','Ticket'],inplace=True,axis=1)
tdata.head()
y_pred=logmodel.predict(tdata)
y_test=pd.read_csv('../input/titanic/gender_submission.csv')
y_test.isnull().sum()
len(y_test['Survived'].values)
from sklearn.metrics import confusion_matrix,accuracy_score
cf=confusion_matrix(y_test['Survived'].values,y_pred)
sns.heatmap(cf,annot=True,fmt='g')
accuracy_score(y_test['Survived'].values,y_pred)*100
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train2=pd.DataFrame(sc_X.fit_transform(x_train))

X_test2=pd.DataFrame(sc_X.fit_transform(tdata))
X_train2.columns=x_train.columns.values

X_test2.columns=tdata.columns.values

X_train2.index=x_train.index.values

X_test2.index=tdata.index.values
X_train=X_train2.sort_index()

X_test=X_test2.sort_index()
X_train
X_test
logmodel.fit(X_train,y_train)
pred=logmodel.predict(X_test)
accuracy_score(y_test['Survived'].values,pred)*100
temp=pd.DataFrame(y_test['PassengerId'],columns=['PassengerId','Survived'])

temp['Survived']=pred

csv_data = temp.to_csv('pred.csv',index=False) 