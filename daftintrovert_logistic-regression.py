import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import os

print(os.listdir("../input"))
file = '../input/train.csv'

train = pd.read_csv(file)
train.head()
train.isnull().head()
plt.figure(figsize = (7,7))

sns.heatmap(train.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
sns.countplot(x = 'Survived',hue= 'Sex',data = train,palette = 'RdBu_r')
sns.countplot(x = 'Survived',hue= 'Pclass',data = train)
sns.distplot(train['Age'].dropna(),kde = False,bins = 30)
train.info()
plt.figure(figsize = (5,5))

sns.countplot(x = 'SibSp',data = train)
plt.figure(figsize = (10,5))

train['Fare'].hist(bins = 40)
plt.figure(figsize = (10,7))

sns.boxplot(x = 'Pclass', y = 'Age',data = train)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass ==1:

            return 37

        elif Pclass ==2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis = 1)
plt.figure(figsize = (7,7))

sns.heatmap(train.isnull(), yticklabels = False,cbar = False,cmap = 'viridis')
train.drop('Cabin',axis = 1,inplace = True)
train.head()
train.dropna(inplace = True)
sex = pd.get_dummies(train['Sex'],drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
embark.head()
train = pd.concat([train,sex,embark],axis = 1)
train.head(2)
train.drop(['Name','Sex','Embarked','Ticket'],axis = 1,inplace = True)
train.head()
train.drop('PassengerId',axis = 1,inplace = True)
train.head()
X = train.drop(['Survived'],axis = 1)

y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel =  LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
file2 = '../input/gender_submission.csv'

sub = pd.read_csv(file2)
a = sub.head(267)

a
my_submission = pd.DataFrame({'PassengerId': a['PassengerId'], 'Survived': predictions})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)