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
import seaborn as sns
import pandas as pd

df=pd.read_csv("../input/titanic/train.csv")
df.shape
df.head()
df.isnull().sum()

test=pd.read_csv("../input/titanic/test.csv")

test.head()
df=df.drop(columns=['Cabin'])
df.groupby(['Embarked','Survived'], as_index=False)[['PassengerId']].count()
df['Embarked'].fillna('S', inplace=True)



sns.countplot(x="Embarked", hue="Survived", data=df)

df['Age'].fillna(df['Age'].median(), inplace=True)

df=df.dropna()



df.shape
df.corr()
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df.head()
df["Age"]=pd.cut(df["Age"],5)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df['Alone'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

df['Sex'] = LE.fit_transform(df['Sex'])

df['Embarked'] = LE.fit_transform(df['Embarked'])

df['Age'] = LE.fit_transform(df['Age'])

df['Title'] = LE.fit_transform(df['Title'])





features=df[['Pclass','Sex','Age','SibSp','Parch','Embarked','Title','Alone','FamilySize']]
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(features)

features=scaler.transform(features)

features
labeles=df["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labeles, test_size=0.2,random_state=5 )

from sklearn import tree 

clf=tree.DecisionTreeClassifier(min_samples_split=70)

clf.fit(X_train,y_train)

pred=clf.predict(X_test)

from sklearn.metrics import confusion_matrix

s=confusion_matrix(y_test, pred)

s
from sklearn.metrics import accuracy_score

accur=accuracy_score(y_test,pred)

print(accur)
test_data=pd.read_csv("../input/titanic/test.csv")

test_data.head()
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_data.head()
test=test_data.drop(columns=['PassengerId','Name'])
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

test['Alone'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)

test.isnull().sum()
test["Age"].fillna(test['Age'].median(), inplace=True)

test["Age"]=pd.cut(test["Age"],5)

test=test[['Pclass','Sex','Age','SibSp','Parch','Embarked','Title','Alone','FamilySize']]
test.head()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

test['Sex'] = LE.fit_transform(test['Sex'])

test['Embarked'] = LE.fit_transform(test['Embarked'])

test['Age'] = LE.fit_transform(test['Age'])

test['Title'] = LE.fit_transform(test['Title'])

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(test)

test=scaler.transform(test)

test
prediction=clf.predict(test)
test_data["Survived"]=prediction
submission=test_data[["PassengerId","Survived"]]
test_data.head()
submission = pd.DataFrame({ 'PassengerId': test_data.PassengerId.values, 'Survived':test_data.Survived.values  })

submission.to_csv("my_submission_1.csv", index=False)
real_data=pd.read_csv("../input/titanic/gender_submission.csv")
real_labeles=real_data["Survived"]
from sklearn.metrics import accuracy_score

accur=accuracy_score(real_labeles,prediction)

print(accur)