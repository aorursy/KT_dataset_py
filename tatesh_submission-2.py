import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
%matplotlib inline
#Read
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.sample(3)
sns.pointplot(x = "Pclass", y = "Survived", hue = "Sex", data = train_data)
sns.barplot(x = "Sex", y = "Survived", hue = "Sex", data = train_data)
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

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = format_name(df)
    df = drop_features(df)
    return df

train_data = transform_features(train_data)
test_data = transform_features(test_data)
train_data.head()
sns.barplot(x = "Age", y = "Survived", hue = "Sex", data = train_data)
sns.barplot(x = "Cabin", y = "Survived", hue = "Sex", data = train_data)
sns.barplot(x = "SibSp", y = "Survived", hue = "Sex", data = train_data)
from sklearn import preprocessing

def encode_features(df_train, df_test):
    features = ['Pclass', 'Sex', 'Age', 'Cabin', 'SibSp']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for features in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[features])
        df_train[features] = le.transform(df_train[features])
        df_test[features] = le.transform(df_test[features])
    return df_train, df_test

train_data, test_data = encode_features(train_data, test_data)
train_data.head()
from sklearn.model_selection import train_test_split

train_predictors = ['Pclass', 'Sex', 'Age', 'Cabin', 'SibSp']

train_X = train_data[train_predictors]
train_y = train_data.Survived

train_X, test_X, train_y, test_y = train_test_split(train_data[train_predictors],
                                                    train_data['Survived'])

forest_model = RandomForestClassifier(n_estimators=100,
                                     criterion='gini',
                                     max_depth=5,
                                     min_samples_split=10,
                                     min_samples_leaf=5,
                                     random_state=0)

forest_model.fit(train_X, train_y)

print("Random Forest score: {0:.2}".format(forest_model.score(test_X, test_y)))
survived_prediction = forest_model.predict(test_data[train_predictors])

my_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': survived_prediction})
my_submission.to_csv('titanic_submission2.csv', index=False)