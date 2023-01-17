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
import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.isnull().sum()
import seaborn as sns
sns.countplot(train['Sex'])
df1=train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
df1.head()
df1['Embarked'].fillna('S',inplace=True)
df1.isnull().any()
df1['Age'].interpolate(inplace=True)
df1.isnull().any()
train['Sex'].replace('male', 1, inplace = True)
train['Sex'].replace('female', 0, inplace = True)
train['Embarked'].replace('S', 2, inplace = True)
train['Embarked'].replace('C', 1, inplace = True)
train['Embarked'].replace('Q', 0, inplace = True)
train.head()
df1 = pd.get_dummies(df1, columns=["Sex","Embarked","Person","Family"])
df1.head()
from sklearn.preprocessing import MinMaxScaler
a=MinMaxScaler()
scaled=a.fit_transform(df1[['Age','Fare']])
df1[['Age','Fare']]=pd.DataFrame(scaled)
df1.head()
df1.corr()
from sklearn.model_selection import train_test_split
X=df1.drop('Survived',axis=1)
y=df1['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc_model = rfc.fit(X_train, y_train)
pred8 = rfc_model.predict(X_test)
print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(y_test, pred8) * 100))
df2=test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
df2.head()
df2 = pd.get_dummies(df2, columns=["Sex","Embarked","Person","Family"])
df2.head()
from sklearn.preprocessing import MinMaxScaler
a=MinMaxScaler()
scaled=a.fit_transform(df2[['Age','Fare']])
df2[['Age','Fare']]=pd.DataFrame(scaled)
df2.head()
pred4 = train_model2.predict(df2)
pred4
pred=pd.DataFrame(pred4)
df = pd.read_csv("../input/titanic/gender_submission.csv")
data=pd.concat([df['PassengerId'],pred],axis=1)
data.columns=['PassengerId','Survived']
data.to_csv('sample_submission.csv',index=False)
