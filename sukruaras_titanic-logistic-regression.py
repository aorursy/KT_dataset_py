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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

combine = [train, test]
train.shape, test.shape
# CREATING NEW VARIABLE AS TITLE



train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train.head(2)
train.shape, test.shape
train.head(2)
# CATEGORIZING THE TITLE VARIABLE



train['Title'] = train['Title'].replace([ 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Countess', 'Lady', 'Sir'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace([ 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Countess', 'Lady', 'Sir'], 'Rare')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')
train.head(2)
train.shape, test.shape
train[["Title","Age"]].groupby("Title").mean()
# FILLING AGE OF NULL VALUES



for i in train["Title"]:

    if i=="Master":

        train["Age"]=train["Age"].fillna(4)

    elif i=="Miss":

        train["Age"]=train["Age"].fillna(22) 

    elif i=="Mr":

        train["Age"]=train["Age"].fillna(32)

    elif i=="Mrs":

        train["Age"]= train["Age"].fillna(36)

    elif i=="Major":

        train["Age"]= train["Age"].fillna(46)

    else:

        train["Age"]=train["Age"].fillna(41)
train.isnull().sum()
test[["Title","Age"]].groupby("Title").mean()
for i in train["Title"]:

    if i=="Master":

        test["Age"]=test["Age"].fillna(7)

    elif i=="Miss":

        test["Age"]=test["Age"].fillna(22) 

    elif i=="Mr":

        test["Age"]=test["Age"].fillna(32)

    elif i=="Mrs":

        test["Age"]= test["Age"].fillna(38)

    elif i=="Major":

        test["Age"]= test["Age"].fillna(44)

    else:

        test["Age"]=test["Age"].fillna(41)
test.isnull().sum()
test[["Pclass","Fare"]].groupby("Pclass").mean()
test["Fare"] = test["Fare"].fillna(12)
test.isnull().sum()
train.isnull().sum()
train.shape, test.shape
train["Embarked"].value_counts()
# FILLING embarked OF NULL VALUES



train["Embarked"]=train["Embarked"].fillna("S")
from sklearn import preprocessing



# WRONG WAY

#  lbe=preprocessing.LabelEncoder()

#  train["Embarked"]=lbe.fit_transform(train["Embarked"])

#  test["Embarked"]=lbe.fit_transform(test["Embarked"])
# YOU SHOULD MAKE GET DUMMIES??????





#  title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

#  for dataset in combine:

#    dataset['Title'] = dataset['Title'].map(title_mapping)

        # dataset['Title'] = dataset['Title'].fillna(0)



test.head(1)
dumytitle=pd.get_dummies(train['Title'])
# OR train=pd.merge(dumytitle, left_index= True, right_index= True, axis=1)



train=pd.concat([train,dumytitle], axis=1)

train.head()
dumytitle2=pd.get_dummies(test['Title'])
test=pd.concat([test,dumytitle2], axis=1)

test.head(1)
train.head(1)
train.shape, test.shape
# MANIPULATING NAME VARIABLE



train_name=train["Name"]

for i in train['Name']:

    train['Name']= train['Name'].replace(i,len(i))
test_name=test["Name"]

for i in test['Name']:

    test['Name']= test['Name'].replace(i,len(i))
train.head(1)
bins = [0,25,40, np.inf]

mylabels = ['s_name', 'm_name', 'l_name',]

train["Name_len"] = pd.cut(train["Name"], bins, labels = mylabels)

test["Name_len"] = pd.cut(test["Name"], bins, labels = mylabels)
train.head(1)
train["Name_len"].value_counts()
train.shape, test.shape
Name_mapping = {'s_name': 1, 'm_name': 2 , 'l_name': 3}

train['Name_len'] = train['Name_len'].map(Name_mapping)

test['Name_len'] = test['Name_len'].map(Name_mapping)
train.head(2)
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)



# to display all rowss:

pd.set_option('display.max_rows', None)



from sklearn.model_selection import train_test_split, GridSearchCV
sns.distplot(train["Age"], kde = False);
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)
train[["AgeGroup","Survived"]].groupby("AgeGroup").mean()
train.shape, test.shape
# Map each Age value to a numerical value:

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult':5 , 'Adult': 6, 'Senior':7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head(1)
test.head(1)
train['FareBand'] = pd.qcut(train['Fare'], 8, labels = [1, 2, 3, 4,5,6,7,8])

test['FareBand'] = pd.qcut(test['Fare'], 8, labels = [1, 2, 3, 4,5,6,7,8])
train.head(1)
train.shape, test.shape
train["FamilySize"] =train["SibSp"]+train["Parch"]+1

train["FamilySize"].mean()
test["FamilySize"] =test["SibSp"]+test["Parch"]+1

test["FamilySize"].mean()
sns.distplot(train["FamilySize"], kde = False);
train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)

train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
train.head(1)
test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)

test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
#  GET DUMMIES  

#  train = pd.get_dummies(train, columns = ["Title"])



train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")
# test = pd.get_dummies(test, columns = ["Title"])



test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
train["Pclass"] = train["Pclass"].astype("category")

train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")
test["Pclass"] = test["Pclass"].astype("category")

test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")
train.shape, test.shape
train.head(1)
test.head(1)
test.info()
# TICKET VARIABLE, selecting first character end coding
train['Ticket1']=train['Ticket'].str[:1]
test['Ticket1']=test['Ticket'].str[:1]
train.head(1)
train['Ticket1']=train['Ticket1'].replace(['1', '2', '3', '4', '5', '6', '7', '8', '9'], 'N')
train['Ticket1']=train['Ticket1'].replace('N', 'NumberTicket')
train['Ticket1']=train['Ticket1'].replace(['A','W', 'F', 'L'], 'OtherTicket')
train['Ticket1']=train['Ticket1'].replace(['S','P', 'C'], ['S_Ticket', 'P_Ticket', 'C_Ticket'])
train["Ticket1"].value_counts()
test['Ticket1']=test['Ticket1'].replace(['1', '2', '3', '4', '5', '6', '7', '8', '9'], 'N')
test['Ticket1']=test['Ticket1'].replace(['S','P', 'C', 'N'], ['S_Ticket', 'P_Ticket', 'C_Ticket','NumberTicket'])
test['Ticket1']=test['Ticket1'].replace(['A','W', 'F', 'L'], 'OtherTicket')
test["Ticket1"].value_counts()
train.head(2)
train.shape, test.shape
dumytitletr1=pd.get_dummies(train['Ticket1'])

train=pd.concat([train,dumytitletr1], axis=1)
dumytitletes1=pd.get_dummies(test['Ticket1'])

test=pd.concat([test,dumytitletes1], axis=1)
train.head(2)
test.head(2)
train.shape, test.shape
Sex_mapping={"male":0,"female":1}

train["Sex"]=train["Sex"].map(Sex_mapping)

test["Sex"]=test["Sex"].map(Sex_mapping)
test=test.drop(['Ticket', 'Ticket1', 'Cabin', 'Title'], axis=1)
train=train.drop(['Ticket', 'Ticket1', 'Cabin', 'Title'], axis=1)
train.shape, test.shape
# GETTING TRAIN DATASET FOR MODEL SETTING



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X = train.drop(['Survived', 'PassengerId'], axis=1)

y = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 21)
x_train.shape, x_test.shape
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_logreg)
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression(random_state= 21, solver='lbfgs', max_iter=1000)

logmodel.fit(x_train, y_train)
logmodel.score(x_train, y_train)
logmodel.coef_
logmodel.intercept_
y_pred = logmodel.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
logmodel.score(x_train, y_train)
X_test = test.drop(['PassengerId'], axis=1)
y_test_pred = logmodel.predict(X_test)
#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': y_test_pred })

output.to_csv('submission.csv', index=False)
output