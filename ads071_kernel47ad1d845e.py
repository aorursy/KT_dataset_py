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
test_df=pd.read_csv("../input/titanic/test.csv")

train_df=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
train_df.head()

train_df.describe()

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style
total = train_df.isnull().sum().sort_values(ascending=False)

percent_1=train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train_df.columns.values
survived='survived'

not_survived='not survived'

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(20,5))

men=train_df[train_df['Sex']=='male']

women=train_df[train_df['Sex']=='female']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax =sns.distplot(men[men['Survived']==1].Age.dropna(),bins=18,label=survived,ax=axes[1],kde=False)

ax =sns.distplot(men[men['Survived']==1].Age.dropna(),bins=18,label=not_survived,ax=axes[1],kde=False)

ax.legend()

ax.set_title('male')
train_df=train_df.drop(['PassengerId'],axis=1)
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 10))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

ax.set_title('Male')
train_df.head()

#test_df.head()

print(train_df)
genders={"male":0 ,"female":1}

data=[train_df,test_df]

for dataset in data:

    dataset["Sex"]=dataset["Sex"].map(genders)

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



data = [train_df, test_df]



for dataset in data:

    mean= train_df["Age"].mean()

    std= test_df["Age"].std()

    is_null= dataset["Age"].isnull().sum()

    rand_age= np.random.randint(mean - std, mean + std, size = is_null)

    age_slice= dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)

train_df["Age"].isnull().sum()
data = [train_df, test_df]

titles={"Mr":0 , "Miss":1 ,"Mrs" :2,"Master": 3,"Royal": 4, "Rare": 5}



for dataset in data:



    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

    dataset['Title'] = dataset['Title'].replace(['Lady','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



    dataset['Title'] = dataset['Title'].map(titles)



    dataset['Title'] = dataset['Title'].fillna(0)

    

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

train_df.head()

ports={"S": 0,"C": 1,"Q": 2}

data=[train_df,test_df]

for dataset in data:

    dataset["Embarked"]=dataset["Embarked"].map(ports)

    dataset["Embarked"]=dataset["Embarked"].fillna(0)

#train_df=train_df.drop(['Name'],axis=1)

#test_df=test_df.drop(['Name'],axis=1)
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]



for dataset in data:

    dataset["Cabin"] = dataset["Cabin"].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset["Deck"] = dataset["Deck"].map(deck)

    dataset["Deck"] = dataset["Deck"].fillna(0)

    dataset["Deck"] = dataset["Deck"].astype(int)

    

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



# let's see how it's distributed 

train_df['Age'].value_counts()
test_df.head()
data = [train_df, test_df]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

train_df['Age_Class'].head()
data=[train_df,test_df]

for dataset in data:

    dataset['Deck_Fare']=dataset['Deck']* dataset['Fare'].astype(float)

    

#train_df['Deck_Fare'].isnull().sum()

test_df.fillna(test_df['Deck_Fare'].mean(),inplace=True)
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)

train_df = train_df.drop(['Fare'], axis=1)

test_df = test_df.drop(['Fare'], axis=1)



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()







from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)









submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':Y_pred})

submission.to_csv('submission.csv', index=False)
Y_train
Y_pred