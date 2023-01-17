import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

data_train.sample(3)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

data_train = simplify_ages(data_train)
data_test = simplify_ages(data_test)
data_train.head()
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

data_train = simplify_cabins(data_train)
data_test = simplify_cabins(data_test)
data_train.head()
data_train.Fare.describe()
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', 'Low', 'Medium', 'High', 'Expensive']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

data_train = simplify_fares(data_train)
data_test = simplify_fares(data_test)
data_train.head()
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

data_train = drop_features(data_train)
data_test = drop_features(data_test)
data_train.head()
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
    group_names = ['Unknown', 'Low', 'Medium', 'High', 'Expensive']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = drop_features(df)
    return df

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()
sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);
sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train);
sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train);
from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
data_train.head()
from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']
num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)