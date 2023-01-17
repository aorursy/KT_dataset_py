import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



import warnings

warnings.simplefilter('ignore')
path = '../input/'

df = pd.read_csv(path + 'train.csv')

train = pd.read_csv(path + 'train.csv')

target = train.Survived.astype('category', ordered=False)

train.drop('Survived', axis=1)



test = pd.read_csv(path + 'test.csv')

PassengerId = test.PassengerId



def get_Titles(df):

    df.Name = df.Name.apply(lambda name: re.findall("\s\S+[.]\s", name)[0].strip())

    df = df.rename(columns = {'Name': 'Title'})

    df.Title.replace({'Ms.': 'Miss.', 'Mlle.': 'Miss.', 'Dr.': 'Rare', 'Mme.': 'Mr.', 'Major.': 'Rare', 'Lady.': 'Rare', 'Sir.': 'Rare', 'Col.': 'Rare', 'Capt.': 'Rare', 'Countess.': 'Rare', 'Jonkheer.': 'Rare', 'Dona.': 'Rare', 'Don.': 'Rare', 'Rev.': 'Rare'}, inplace=True)

    return df



def fill_Age(df):

    df.Age = df.Age.fillna(df.groupby("Title").Age.transform("median"))

    return df



def get_Group_size(df):

    Ticket_counts = df.Ticket.value_counts()

    df['Ticket_counts'] = df.Ticket.apply(lambda x: Ticket_counts[x])

    df['Family_size'] = df['SibSp'] + df['Parch'] + 1

    df['Group_size'] = df[['Family_size', 'Ticket_counts']].max(axis=1)

    return df



def process_features(df):

    df.Sex = df.Sex.astype('category', ordered=False).cat.codes

    features_to_keep = ['Age', 'Fare', 'Group_size', 'Pclass', 'Sex']

    df = df[features_to_keep]

    return df



def process_data(df):

    df = df.copy()

    df = get_Titles(df)

    df = fill_Age(df)

    df = get_Group_size(df)

    df = process_features(df)

    medianFare = df['Fare'].median()

    df['Fare'] = df['Fare'].fillna(medianFare)

    return df



X_train, X_test = process_data(train), process_data(test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, target, random_state=0)
X_train.head()
y_train.head()
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_test})

submission.to_csv('submission.csv', index=False)