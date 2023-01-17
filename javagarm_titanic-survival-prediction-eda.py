# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head(10)
train_data.tail(9)
train_data.shape
train_data.isnull()
sns.heatmap(train_data.isnull(),yticklabels=False)
train_data.isnull().sum()

train_data.info()

train_data.Pclass.unique()

plt.figure(figsize=(8,6))
sns.countplot(x='Survived',hue='Pclass',data=train_data,palette='rainbow')
sns.boxplot(x = train_data["Survived"],y = train_data["Age"],hue = train_data["Survived"],palette = 'dark')
sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=train_data, aspect=1,height=8);
train_data.Sex.unique()


plt.figure(figsize=(8,6))
sns.countplot(x='Survived',hue='Sex',data=train_data)
train_data.Age.unique()

sns.distplot(train_data['Age'].dropna(),kde=False,color='darkred',bins=40)
train_data.Embarked.unique()

train_data.Name.unique()
train_data['Title']=0
train_data['Title']=train_data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
train_data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
train_data['Name_length'] = train_data['Name'].apply(len)
train_data.head(10)

train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Mr'),'Age']= train_data.Age[train_data.Title=="Mr"].mean()
train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Mrs'),'Age']= train_data.Age[train_data.Title=="Mrs"].mean()
train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Master'),'Age']= train_data.Age[train_data.Title=="Master"].mean()
train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Miss'),'Age']= train_data.Age[train_data.Title=="Miss"].mean()
train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Other'),'Age']= train_data.Age[train_data.Title=="Other"].mean()
train_data.isnull().sum()
train_data.Age.unique()
train_data.shape
train_data=train_data.dropna(subset=['Age','Embarked'])
train_data.isnull().sum()
train_data.shape
xtrain_data = train_data.drop(['Cabin','Ticket','Name'], axis =1 )
xtrain_data.Parch.unique()
Ticketclass = {3:"lower_class",2:'middle_class',1:'upper_class'}
xtrain_data['Pclass'] = xtrain_data['Pclass'].map(Ticketclass)
xtrain_data.Pclass.unique()
train_titan = xtrain_data
train_titan.head()
train_dummy = pd.get_dummies(train_titan[['Pclass','Sex','Embarked']],)
train_titan=pd.concat([train_titan.drop(['Pclass','Sex','Embarked'],axis=1),train_dummy],axis=1)
train_titan.head(12)
train_titan.isnull().sum()
train_titan.head()
train_titan.info()

train_titan.Age.value_counts()
train_titan.Age.unique()
train_titan.Age = train_titan.Age.astype(int)
train_titan.Age.unique()
train_titan['Age'].replace({0:1},inplace = True)
train_titan['Fare'] = np.round(train_titan.Fare,decimals=3)
train_titan.head()
train_titan = train_titan.drop('Title',axis = 1)
sns.heatmap(train_titan.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train_titan.head()
test_data.head()
test_data['Title']=0
test_data['Title']=test_data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
test_data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

test_data['Name_length'] = test_data['Name'].apply(len)
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Mr'),'Age']= test_data.Age[test_data.Title=="Mr"].mean()
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Mrs'),'Age']= test_data.Age[test_data.Title=="Mrs"].mean()
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Master'),'Age']= test_data.Age[test_data.Title=="Master"].mean()
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Miss'),'Age']= test_data.Age[test_data.Title=="Miss"].mean()
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Other'),'Age']= test_data.Age[test_data.Title=="Other"].mean()
test_data.head(8)
test_data.shape

test_data.isnull().sum()

xtest_data = test_data.drop({'Name','Cabin','Title'},axis = 1)

xtest_data.info()
Ticketclass = {3:"lower_class",2:'middle_class',1:'upper_class'}
xtest_data['Pclass'] = xtest_data['Pclass'].map(Ticketclass)
test_dummies = pd.get_dummies(xtest_data[['Pclass','Sex','Embarked']])
test_dummies.head()

test_titan = pd.concat([xtest_data.drop(['Pclass','Sex','Embarked'],axis =1),test_dummies],axis = 1)
test_titan.head()

test_titan.shape

test_titan.Fare= np.round(test_titan.Fare,decimals=3)
test_titan.isnull().sum()
test_titan['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_titan.head()
test_titan.Age = test_titan.Age.astype(int)


test_titan.Age.unique()
test_titan.replace({0:1},inplace=True)
test_titan.Age.unique()
test_titan = test_titan.drop(['Ticket'],axis =1)
test_titan.shape
test_titan.head()
features = ['Age','SibSp','Parch','Fare','Pclass_lower_class','Pclass_middle_class','Pclass_upper_class','Name_length','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']
selected_train_titan = train_titan[features]
labels= train_titan.Survived
x_train, x_test, y_train, y_test = train_test_split(selected_train_titan,labels,train_size=0.75,test_size=0.25,random_state=42)

model =LogisticRegression(max_iter = 10000)
model.fit(x_train, y_train)

print(model.score(x_train, y_train))
print(model.score(x_test,y_test))
selected_test_titan = test_titan[features]
model =LogisticRegression(max_iter=10000)
model.fit(selected_train_titan,labels)

submission = pd.DataFrame({'PassengerID':test_titan.PassengerId,'Survived':model.predict(selected_test_titan)})
submission.to_csv("Submissions.csv")
