# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re # use for regex



# Input data files are available in the "../input/" directory.
# import dataset train and test

input_path = "../input/titanic/"



df_train = pd.read_csv(input_path + 'train.csv')

df_test = pd.read_csv(input_path + 'test.csv')

sub = pd.read_csv(input_path + "gender_submission.csv")
# Check start line dataset

df_train.head()
# Check end line dataset

df_train.tail()
# Verification of data typing for df_train

df_train.info()
# # Verification of data typing for df_test

df_test.info()
# Verification of data completeness

df_train.isna().sum()
# check if there are any valuations that are not zero

df_test.isna().sum()
# Printing df_train basic descriptive statistics

df_train.describe()
# # Printing df_test basic descriptive statistics

df_test.describe()
# Let's calculate the average age with df_train

average_age = df_train['Age'].loc[~df_train['Age'].isna()].mean()

average_age
# Replace Age None Attribuate Values by Average age

df_train['Age'] = df_train['Age'].fillna(average_age)

df_test['Age'] = df_test['Age'].fillna(average_age)

df_train
# count peaple boarding

df_train['Embarked'].value_counts()
df_train['Embarked'].fillna('S', inplace = True)

df_test['Embarked'].fillna('S', inplace = True)
# create columns who said who in (0) or not in cabin (1)

df_train['Cabin'] = np.where(df_train['Cabin'].isnull() , 0, 1)

df_test['Cabin'] = np.where(df_test['Cabin'].isnull() , 0, 1)

df_train
# # check if there are any valuations that are not zero on test dataframe

df_test.isna().sum()
df_test.loc[df_test['Fare'].isnull()]
# calculate the Fare average on test dataframe

average_fare = df_test['Fare'].loc[~df_test['Fare'].isna()].mean()

average_fare
# give him the Fare average

df_test['Fare'] = df_train['Fare'].fillna(average_fare)
df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df_test['Sex'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



df_train['Sex'] = df_train['Sex'].astype('category').cat.codes

df_train['Embarked'] = df_train['Embarked'].astype('category').cat.codes



df_train
sns.countplot(x="Sex", data=df_train)
sns.catplot(x="Sex", y="Survived", data=df_train, kind="bar")
sns.catplot(x="Pclass", y="Survived", data=df_train, kind="bar")
sns.countplot(x="Pclass", data=df_train)
sns.catplot(x="SibSp", y="Survived", data=df_train, kind="bar")
sns.countplot(x="SibSp", data=df_train)
sns.catplot(x="Parch", y="Survived", data=df_train, kind="bar")
sns.countplot(x="Parch", data=df_train)
sns.catplot(x="Embarked", y="Survived", data=df_train, kind="bar")
sns.countplot(x="Embarked", data=df_train)
sns.catplot(x="Cabin", y="Survived", data=df_train, kind="bar")
sns.boxplot(x='Survived', y="Fare", data=df_train.loc[df_train['Fare'] <500]);
df_train['FamilyCount'] = df_train['SibSp'] + df_train['Parch']

df_test['FamilyCount'] = df_test['SibSp'] + df_test['Parch']
df_train
def ageGroup(df):

    if df['Age'] <= 0 :

        group='Unknown'

    elif df['Age'] <= 14 :

        group='Child'

    elif df['Age'] <=24 :

        group='Teenager'

    elif df['Age'] <=64 :

        group='Adult'

    else :

        group='Senior'

    return group



df_train['Age'] = df_train.apply(ageGroup, axis=1)

df_test['Age'] = df_test.apply(ageGroup, axis=1)



df_train['Age'] = df_train['Age'].map( {'Unknown': 0, 'Child': 1, 'Teenager': 2, 'Adult': 3,'Senior': 4 } ).astype(int)

df_test['Age'] = df_test['Age'].map( {'Unknown': 0, 'Child': 1, 'Teenager': 2, 'Adult': 3,'Senior': 4 } ).astype(int)

df_train
sns.catplot(x="Age", y="Survived", data=df_train, kind="bar")
def civility(df):

    if re.search('Mme.',df['Name']) != None:

        civility = 'Mrs'

    elif re.search('Ms.',df['Name']) != None:

        civility = 'Mrs'

    elif re.search('Major.',df['Name']) != None:

        civility = 'Major'

    elif re.search('Capt.',df['Name']) != None:

        civility = 'Captain'

    elif re.search('Jonkheer.',df['Name']) != None:

        civility = 'Jonkheer'

    elif re.search('Mlle.',df['Name']) != None:

        civility = 'Miss'

    elif re.search('the Countess.',df['Name']) != None:

        civility = 'the Countess'

    elif re.search('Mlle.',df['Name']) != None:

        civility = 'colonel'

    elif re.search('Col.',df['Name']) != None:

        civility = 'Colonel'

    elif re.search('Don.',df['Name']) != None:

        civility = 'Don'

    elif re.search('Dr.',df['Name']) != None:

        civility = 'Doctor'

    elif re.search('Master.',df['Name']) != None:

        civility = 'Master'

    elif re.search('Mrs.',df['Name']) != None:

        civility = 'Mrs'

    elif re.search('Miss.',df['Name']) != None:

        civility = 'Miss'

    elif re.search('Rev.',df['Name']) != None:

        civility = 'Reverand'

    elif re.search('Mr.',df['Name']) != None:

        civility = 'Mr'

    else :

        civility = 'Unknown'

    return civility



df_train['Civility'] = df_train.apply(civility, axis=1)

df_test['Civility'] = df_test.apply(civility, axis=1)



df_train['Civility'] = df_train['Civility'].map( {'Unknown': 0, 'Mr': 1, 'Mrs': 2, 'Miss': 3,'Reverand': 4, 'Master': 5, 'Doctor': 6, 'Don': 7, 'Colonel': 8, 'the Countess': 9, 'Jonkheer': 10, 'Captain': 11, 'Major': 12 } ).astype(int)

df_test['Civility'] = df_test['Civility'].map( {'Unknown': 0, 'Mr': 1, 'Mrs': 2, 'Miss': 3,'Reverand': 4, 'Master': 5, 'Doctor': 6, 'Don': 7, 'Colonel': 8, 'the Countess': 9, 'Jonkheer': 10, 'Captain': 11, 'Major': 12 } ).astype(int)



df_train

# For find the rest of civility to map

df_train.query("Civility == '0'")
sns.catplot(x="Civility", y="Survived", data=df_train, kind="bar")
df_train
def fareSection(df):

    df['Fare']

    if df['Fare'] < 10:

        fare = 'less expensive'

    elif df['Fare'] < 30:

        fare = 'less expensive than 30'

    elif df['Fare'] < 70:

        fare = 'less expensive than 70'

    elif df['Fare'] < 100:

        fare = 'less expensive than 100'

    else : 

        fare = 'expensive price'

    return fare



df_train['Fare'] = df_train.apply(fareSection, axis=1)

df_test['Fare'] = df_test.apply(fareSection, axis=1)



df_train['Fare'] = df_train['Fare'].map( { 'less expensive': 0, 'less expensive than 30': 1,'less expensive than 70': 2, 'less expensive than 100': 3, 'expensive price': 4  } ).astype(int)

df_test['Fare'] = df_test['Fare'].map( { 'less expensive': 0, 'less expensive than 30': 1,'less expensive than 70': 2, 'less expensive than 100': 3, 'expensive price': 4   } ).astype(int)



df_train
my_cols = ['Age', 'Sex', 'Pclass', 'FamilyCount', 'Fare', 'SibSp', 'Parch', 'Cabin', 'Civility']
y_train = df_train['Survived']
X_train = df_train.loc[:,my_cols]
X_test = df_test.loc[:, my_cols]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

rf = RandomForestClassifier(n_estimators=100)
def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))

    test_acc = accuracy_score(y_test, model.predict(X_test))

    return {

        'train accuracy': train_acc,

        'test accuracy': test_acc

    }



print(train_model(rf, X_train, y_train))
rf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cell to store the result of the model by calling its rf model and data set

# generate a dataframe with PassengerId and survived



submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'survived': rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)
