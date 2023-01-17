import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import csv

%matplotlib inline
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.info()
len(train)
train.count()
train.head()
train.isnull()
train.isnull().sum()
train.Name.unique()
train.Age.unique()
def title(name_col):

    if 'Mrs' in name_col:

        return "Mrs"

    elif 'Mr' in name_col:

        return "Mr"

    elif 'Miss' in name_col:

        return "Miss"

    elif 'Master' in name_col:

        return "Master"

    else:

        return 'unknown'

train['Name'] = train['Name'].apply(title)

train.head(10)
train.Name.unique()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='copper')
sns.set_style('darkgrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='cubehelix_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='gist_rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='black',bins=40)
train['Age'].hist(bins=30,color='blue',alpha=0.3)

sns.countplot(x='SibSp',data=train,palette='twilight_shifted_r')
train['Fare'].hist(color='darkblue',bins=40,figsize=(8,6))
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='GnBu')
train.Embarked.value_counts()

train.Cabin.value_counts()

train[train.Embarked.isna()]

train[train['Pclass'] == 1]

train[train['Cabin'] == "B28"]

train[(train['Sex'] == 'female') & (train['Age'] >=38) & (train['Parch'] == 0) & (train['Pclass'] == 1) & (train['Fare'] >=80)]

train.loc[train.Embarked.isna(), 'Embarked',] = 'C'

train[(train['Age']>=38) & (train['Parch']==0) & (train['Sex'] == 'female') & (train['Fare'] >=80)]

train.info()

train.loc[train['Sex']=='female','Sex',] =1

train.loc[train['Sex']=='male','Sex',] =0

train
def impute_age(cols):

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
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train.isnull().sum()

train.drop('Cabin',axis=1,inplace=True)

train.head()

train.isnull().sum()

train.dropna(inplace=True)

train.values[:1]

train.values.shape

pd.get_dummies(train['Embarked'],drop_first=True).head()

sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.head()

train = pd.concat([train,sex,embark],axis=1)

train
train.drop('Survived',axis=1).head()

train['Survived'].head(100)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.10, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)

accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)

accuracy

predictions
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

gender_submission.to_csv('submission.csv', index=False)