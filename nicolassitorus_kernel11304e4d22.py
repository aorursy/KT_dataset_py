import numpy as py

import pandas as pd



data_train = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')
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



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    



def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



data_train = transform_features(data_train)

data_test = transform_features(data_test)

data_train.head()
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

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



x_all = data_train.drop(['Survived', 'PassengerId'], axis=1)

y_all = data_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=23)
from sklearn.ensemble import RandomForestClassifier as RandomForest



data_model = RandomForest(n_estimators=1000,bootstrap=True)

data_model.fit(x_train, y_train)
passanger_id = data_test['PassengerId']

predictions = data_model.predict(data_test.drop('PassengerId', axis=1))



result = pd.DataFrame({ 'PassengerId' : passanger_id, 'Survived': predictions })



result.to_csv('titanic_kaggle.csv', index = False)



result.head()