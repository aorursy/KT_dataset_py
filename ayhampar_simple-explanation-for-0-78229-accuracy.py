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
# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns


train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head(10)
train_data.tail()
train_data.shape
train_data.columns.values

train_data.info()

missın_values = train_data.isnull().sum().sort_values(ascending=False)
missın_values
percent_missing_cabin = train_data["Cabin"].isnull().sum()/train_data.shape[0]*100
percent_missing_cabin
import re

train_data['Cabin'] = train_data['Cabin'].fillna("M101")
train_data['Deck'] = train_data['Cabin'].map(lambda x: re.compile("([azA-Z]+)").search(x).group())
sns.countplot(x="Deck",hue="Survived",data=train_data)
train_data[train_data['Deck']=='T'].replace(train_data,"M")
train_data["Cabin"].isnull().sum()


percent_missing_cabin = train_data["Age"].isnull().sum()/train_data.shape[0]*100
percent_missing_cabin
train_data['Age'].mean()
train_data['Age'].median()
train_data['Age'].plot.hist(grid=False, bins=30, rwidth=0.9,color='#FDCDA4')

std = train_data["Age"].std()
std
mean = train_data["Age"].mean()
std = train_data["Age"].std()
is_null = train_data["Age"].isnull().sum()
random_values = np.random.randint(mean - std, mean + std, size =is_null)
age_slice = train_data["Age"].copy()
age_slice[np.isnan(age_slice)] = random_values
train_data["Age"] = age_slice
train_data["Age"] = train_data["Age"].astype(int)
train_data['Age'].plot.hist(grid=False, bins=30, rwidth=0.9,color='#FDCDA4')


train_data['Embarked'].describe()
top = 'S'
train_data['Embarked'] = train_data['Embarked'].fillna(top)
train_data.isnull().sum()
train_data.info()
train_data.drop(columns=['PassengerId'],axis=0,inplace=True)
train_data['Fare'] = train_data['Fare'].astype(int)
train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

train_data.groupby('Title')['Title'].count()

# replace titles with a more common title or as Rare
train_data['Title'] = train_data['Title'].replace(['Lady','Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir','Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

sns.countplot(x="Title",hue="Survived",data=train_data)
train_data.groupby('Title')['Title'].count()
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data['Alone'] = train_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train_data.head()
train_data=train_data[['Title','Deck','Embarked','Parch','SibSp','Age','Sex','Pclass','Alone','FamilySize','Survived']]
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
train_data['Sex'] = LE.fit_transform(train_data['Sex'])
train_data['Embarked'] = LE.fit_transform(train_data['Embarked'])
train_data['Age'] = LE.fit_transform(train_data['Age'])
train_data['Title'] = LE.fit_transform(train_data['Title'])
train_data['Deck'] = LE.fit_transform(train_data['Deck'])
train_data.head()
labeles=train_data['Survived']
features=train_data.drop(columns=['Survived'])
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
test_data.isnull().sum()
import re

test_data['Cabin'] = test_data['Cabin'].fillna("M101")
test_data['Deck'] = test_data['Cabin'].map(lambda x: re.compile("([azA-Z]+)").search(x).group())
test_data.groupby('Deck')['Deck'].count()
mean1 = test_data["Age"].mean()
std1 = test_data["Age"].std()
is_null1 = test_data["Age"].isnull().sum()
random_values1 = np.random.randint(mean1 - std1, mean1 + std1, size =is_null1)
age_slice1 = test_data["Age"].copy()
age_slice1[np.isnan(age_slice1)] = random_values1
test_data["Age"] = age_slice1
test_data["Age"] = test_data["Age"].astype(int)
m=test_data['Fare'].mean()
test_data['Fare'] = test_data['Fare'].fillna(m)
test_data.isnull().sum()
test_data['Fare'] = test_data['Fare'].astype(int)
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
test_data.groupby('Title')['Title'].count()
# replace titles with a more common title or as Rare
test_data['Title'] = test_data['Title'].replace(['Lady','Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir','Jonkheer', 'Dona'], 'Rare')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data.groupby('Title')['Title'].count()
test_data["Age"]=pd.cut(test_data["Age"],5)

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['Alone'] = test_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test_data.head()
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
test_data['Sex'] = LE.fit_transform(test_data['Sex'])
test_data['Embarked'] = LE.fit_transform(test_data['Embarked'])
test_data['Age'] = LE.fit_transform(test_data['Age'])
test_data['Title'] = LE.fit_transform(test_data['Title'])
test_data['Deck'] = LE.fit_transform(test_data['Deck'])
features_test=test_data[['Title','Deck','Embarked','Parch','SibSp','Age','Sex','Pclass','Alone','FamilySize']]
test_data["Survived"]=clf.predict(features_test)

test_data.head()
submission=test_data[["PassengerId","Survived"]]
submission = pd.DataFrame({ 'PassengerId': test_data.PassengerId.values, 'Survived':test_data.Survived.values  })
submission.to_csv("my_submission.csv", index=False)
