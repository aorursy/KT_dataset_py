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

import seaborn as sns

%matplotlib inline
# Import Data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
test_data.head()
train_data.info()
train_data.describe()
# Explore Training Data
sns.countplot(x='Survived',data=train_data,hue='Sex')
sns.countplot(x='Survived',data=train_data,hue='Pclass')
train_data['Age'].hist(bins=35)
sns.countplot(x='SibSp',data = train_data)
sns.countplot(x='Parch',data=train_data)
train_data['Fare'].hist(bins=40)
sns.heatmap(train_data.corr(),cmap='coolwarm')
sns.countplot(x='Survived',data=train_data,hue='Embarked')
plt.figure(figsize=(12,5))

sns.heatmap(train_data.isnull(),cmap = 'viridis',cbar=False,yticklabels=False)
plt.figure(figsize=(12,5))

sns.heatmap(test_data.isnull(),cmap = 'viridis',cbar=False,yticklabels=False)
# Data Cleaning
train_data.drop('Cabin',axis=1,inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)
plt.figure(figsize=(12,5))

sns.heatmap(train_data.isnull(),cmap = 'viridis',cbar=False,yticklabels=False)
plt.figure(figsize=(12,5))

sns.heatmap(test_data.isnull(),cmap = 'viridis',cbar=False,yticklabels=False)
# Feature Engineering
# Only training data should used. Test data should not be introduced

sns.boxplot(x='Pclass',y='Age',data=train_data)
# Function to replace missing age values based of the mean age

def add_avg_age_for_null(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
# Fill the missing age values based of the mean age in the training data

train_data['Age'] = train_data[['Age','Pclass']].apply(add_avg_age_for_null,axis=1)

test_data['Age'] = test_data[['Age','Pclass']].apply(add_avg_age_for_null,axis=1)
train_data.dropna(inplace=True)
test_data['Fare'].fillna(test_data.groupby('Pclass')['Fare'].transform('median'),inplace=True)
plt.figure(figsize=(12,5))

sns.heatmap(train_data.isnull(),cmap = 'viridis',cbar=False,yticklabels=False)
plt.figure(figsize=(12,5))

sns.heatmap(test_data.isnull(),cmap = 'viridis',cbar=False,yticklabels=False)
# Convert categorical features into dummy variables
dummies_train = pd.get_dummies(train_data[['Sex','Embarked']],drop_first=True)

dummies_test = pd.get_dummies(test_data[['Sex','Embarked']],drop_first=True)
train_data = pd.concat([train_data.drop(['Sex','Embarked'],axis=1),dummies_train],axis=1)

test_data = pd.concat([test_data.drop(['Sex','Embarked'],axis=1),dummies_test],axis=1)
train_data.head()
test_data.head()
train_data['Name Title'] = train_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

test_data['Name Title'] = test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
train_data['Name Title'].value_counts()
# Function to map name title to Mr,Mrs,Miss or Others

def name_title_mapping(title):    

    

    if title == 'Mr':

        return 0

    elif title == 'Mrs':

        return 1

    elif title == 'Miss':

        return 2

    else:

        return 3
train_data['Name Title'] = train_data['Name Title'].apply(name_title_mapping)

test_data['Name Title'] = test_data['Name Title'].apply(name_title_mapping)
train_data.drop('Name',axis=1,inplace=True)

test_data.drop('Name',axis=1,inplace=True)
train_data.head()
test_data.head()
train_data['Numeric Ticket'] = train_data['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)

test_data['Numeric Ticket'] = test_data['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
train_data.drop('Ticket',axis=1,inplace=True)

test_data.drop('Ticket',axis=1,inplace=True)
train_data.head()
test_data.head()
X_train = train_data.drop('Survived',axis = 1)

y_train = train_data['Survived']

X_test = test_data
#scale the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Model Selection
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
kfold = KFold(n_splits=10,shuffle=True,random_state=0)
knn = KNeighborsClassifier(n_neighbors=15)

score = cross_val_score(knn,X_train,y_train,cv=kfold,scoring='accuracy')

print(score)

print(score.mean())
dtc = DecisionTreeClassifier()

score = cross_val_score(dtc,X_train,y_train,cv=kfold,scoring='accuracy')

print(score)

print(score.mean())
rfc = RandomForestClassifier(n_estimators=15)

score = cross_val_score(rfc,X_train,y_train,cv=kfold,scoring='accuracy')

print(score)

print(score.mean())
gnb = GaussianNB()

score = cross_val_score(gnb,X_train,y_train,cv=kfold,scoring='accuracy')

print(score)

print(score.mean())
svc = SVC()

score = cross_val_score(svc,X_train,y_train,cv=kfold,scoring='accuracy')

print(score)

print(score.mean())
svc_model = SVC()

svc_model.fit(X_train, y_train)
predictions = svc_model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('SVC_model_.csv', index=False)

print("Your submission was successfully saved!")
RandForesClass_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

RandForesClass_model.fit(X_train, y_train)
predictions = RandForesClass_model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('RandForesClass_model_02.csv', index=False)

print("Your submission was successfully saved!")