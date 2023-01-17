%matplotlib inline

import pandas as pd

import os

pd.options.display.max_columns = 100

import warnings

import matplotlib

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

import seaborn as sns

matplotlib.style.use('ggplot')

from matplotlib import pyplot as plt

import numpy as np

pd.options.display.max_rows = 100

fname = '../input/train.csv'

data = pd.read_csv(fname)

fname2 = '../input/test.csv'

test = pd.read_csv(fname2)

len(data)

data.head()
data.count()
data['Age'].min(), data['Age'].max()
data['Survived'].value_counts()
data['Survived'].value_counts() * 100 / len(data)
data['Sex'].value_counts()
data['Pclass'].value_counts()
%matplotlib inline



alpha_color = 0.5



data['Survived'].value_counts().plot(kind='bar')
data['Sex'].value_counts().plot(kind="bar", color=['b', 'r'], alpha=alpha_color)
data['Pclass'].value_counts().sort_index().plot(kind='bar', alpha=alpha_color)
data.plot(kind='scatter', x='Survived', y='Age')
data[data['Survived'] == 1]['Age'].value_counts().sort_index().plot(kind='bar')
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]



data['AgeBin'] = pd.cut(data['Age'], bins)
data[data['Survived'] == 1]['AgeBin'].value_counts().sort_index().plot(kind='bar')
data[data['Survived'] == 0]['AgeBin'].value_counts().sort_index().plot(kind='bar')
data['AgeBin'].value_counts().sort_index().plot(kind='bar')
data[data['Pclass'] == 1]['Survived'].value_counts().plot(kind='bar')
data[data['Pclass'] == 3]['Survived'].value_counts().plot(kind='bar')
data[data['Sex'] == 'male']['Survived'].value_counts().plot(kind='bar')
data[data['Sex'] == 'female']['Survived'].value_counts().plot(kind='bar')
data[(data['Sex'] == 'male') & (data['Pclass'] == 1)]['Survived'].value_counts().plot(kind="bar") 
data[(data['Sex'] == 'male') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind="bar") 
data[(data['Sex'] == 'female') & (data['Pclass'] == 1)]['Survived'].value_counts().plot(kind="bar") 
data[(data['Sex'] == 'female') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind="bar") 
data.count()
data['Age'].fillna(data['Age'].median(), inplace=True)
data.describe()
survived_sex = data[data['Survived']==1]['Sex'].value_counts()

dead_sex = data[data['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex, dead_sex])

df.index = ['Survived', 'Dead']

df.plot(kind="bar", stacked=True, figsize=(15, 8))
data[data.Age.isnull()]
data[(data.Age < 11) & (data.Sex=="female")].Survived.value_counts().plot(kind="bar")
data[(data.Age < 11)].Survived.value_counts().plot(kind="bar")
sns.barplot(x="Pclass", y="Survived", hue="Sex", data = data, palette={"male" : "blue", "female" : "pink"});
def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df
def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df
def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df
def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = drop_features(df)

    return df
train_df2 = transform_features(data)

test_df2 = transform_features(test)
train_df2.head()
sns.barplot(x="Age", y="Survived", hue="Sex", data=train_df2, palette={"male" : "blue", "female" : "pink"});
sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train_df2, palette={"male": "blue", "female": "pink"});
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_df2, palette={'male': 'blue', 'female': 'pink'});
def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test



train_df2, test_df2 = encode_features(train_df2, test_df2)

train_df2.head()
train_df2.info()
X_train = train_df2.drop(['Survived', 'PassengerId'], axis=1)

Y_train = train_df2["Survived"]

X_test = test_df2.drop('PassengerId', axis=1).copy()



X_train.shape, Y_train.shape, X_test.shape
X_train.head()
Y_train.head()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": Y_pred

})
submission.describe()
os.getcwd()
submission.to_csv('/Desktop/submission.csv', index=False)