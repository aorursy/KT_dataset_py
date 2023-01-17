# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv('../input/train.csv' )

data_test = pd.read_csv('../input/test.csv' )
def set_cabin(df):

    # 设置Cabin如果空值为No，有值为Yes

    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'

    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'

    return df





def set_title(df):

    # 设置称谓为5种，Mr,Mrs,Miss,noble,other

    import re

    df['Title'] = df.Name.map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

    title_dict = {}

    title_dict.update(dict.fromkeys(['Mr'], 'mr'))

    title_dict.update(dict.fromkeys(['Mrs', 'Ms', 'Mme'], 'mrs'))

    title_dict.update(dict.fromkeys(['Miss', 'Mlle'], 'miss'))

    title_dict.update(dict.fromkeys(['Master', 'Sir', 'the Countess', 'Lady', 'Major', 'Mme', 'Col', 'Don', 'Dona'], 'noble'))

    title_dict.update(dict.fromkeys(['Rev', 'Dr', 'Jonkheer', 'Capt'], 'other'))

    df['Title'] = df.Title.map(title_dict)

    return df





def set_age_level(df):

    # 年龄等级划分

    df.loc[(df.Age <= 15.), 'Agelevel'] = 'child'

    df.loc[(df.Age > 15.) & (df.Age <= 35.), 'Agelevel'] = 'young'

    df.loc[(df.Age > 35.) & (df.Age <= 60.), 'Agelevel'] = 'midlife'

    df.loc[(df.Age > 60.), 'Agelevel'] = 'old'

    return df





def get_missing_embarked(df):

    # 缺失的不多使用众数填充

    df['Embarked'].fillna(df.Embarked.mode().iloc[0], inplace=True)

    return df





def set_missing_age(df_train, df_test):

    # 随机森林算法填补缺失年龄

    target = ['Age', 'Fare', 'Pclass', 'Name_lenth', 'SibSp', 'Parch', 'Fare', 'Cabin_No', 'Cabin_Yes', 'Embarked_C',

              'Embarked_Q', 'Embarked_S', 'Sex_female', 'Sex_male', 'Title_miss', 'Title_mr', 'Title_mrs', 'Title_noble', 'Title_other']

    age_df = df_train[target]

    age_df_test = df_test[target]

    known_age = age_df[age_df.Age.notnull()].values

    unknown_age = age_df[age_df.Age.isnull()].values



    y = known_age[:, 0]

    X = known_age[:, 1:]

    rf_reg = RandomForestRegressor(n_estimators=1200, max_leaf_nodes=20, random_state=500, n_jobs=-1, oob_score=True)

    rf_reg.fit(X, y)

    predict_train_ages = rf_reg.predict(unknown_age[:, 1:])

    predict_test_ages = rf_reg.predict(age_df_test[age_df_test.Age.isnull()].values[:, 1:])

    df_train.loc[(df_train.Age.isnull()), 'Age'] = predict_train_ages

    df_test.loc[(df_test.Age.isnull()), 'Age'] = predict_test_ages

    return df_train, df_test





def set_data_standard(df_train, df_test):

    # 舱号，年龄，票价数据标准化

    standardScaler = StandardScaler()

    # standardScaler.fit(df_train[['Name_lenth', 'SibSp', 'Parch', 'Pclass', 'Age', 'Fare']].values)

    standardScaler.fit(df_train[['Name_lenth', 'Age', 'Fare']].values)

    df_train_scaled = standardScaler.transform(df_train[['Name_lenth', 'Age', 'Fare']].values)

    df_test_scaled = standardScaler.transform(df_test[['Name_lenth', 'Age', 'Fare']].values)

    df_train_scaled = pd.DataFrame(df_train_scaled, columns=['Name_lenth_scaled', 'Age_scaled', 'Fare_scaled'])

    df_train = pd.concat([df_train, df_train_scaled], axis=1)

    df_test_scaled = pd.DataFrame(df_test_scaled, columns=['Name_lenth_scaled', 'Age_scaled', 'Fare_scaled'])

    df_test = pd.concat([df_test, df_test_scaled], axis=1)

    return df_train, df_test





def data_process():

    # 填补缺失数据，数据标准化，数据转换

    # 测试数据缺少一个Fare数据，使用仓位平均值求得

    global data_train

    data_train = set_cabin(data_train)

    data_train = set_title(data_train)

    data_train = get_missing_embarked(data_train)

    data_train['Name_lenth'] = data_train.Name.apply(len)

    dummies_cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

    dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

    dummies_sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

    dummies_title = pd.get_dummies(data_train['Title'], prefix='Title')

    data_train = pd.concat([data_train, dummies_cabin, dummies_embarked, dummies_sex, dummies_title], axis=1)

    data_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1, inplace=True)



    global data_test

    data_test = set_cabin(data_test)

    data_test = set_title(data_test)

    data_test['Name_lenth'] = data_test.Name.apply(len)

    # print(data_train.Fare.groupby(by=data_train['Pclass']).mean().get([1, 2]))

    # print(data_test[data_test['Fare'].isnull()]['Pclass'])

    # print(data_test.loc[(data_test['Fare'].isnull()), 'Fare'])

    data_test['Fare'].fillna(data_train.Fare.groupby(by=data_train['Pclass']).mean().get([3]).values[0], inplace=True)

    dummies_test_cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')

    dummies_test_embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')

    dummies_test_sex = pd.get_dummies(data_test['Sex'], prefix='Sex')

    dummies_test_title = pd.get_dummies(data_test['Title'], prefix='Title')

    data_test = pd.concat([data_test, dummies_test_cabin, dummies_test_embarked, dummies_test_sex, dummies_test_title], axis=1)

    data_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1, inplace=True)



    data_train, data_test = set_missing_age(data_train, data_test)

    data_train = set_age_level(data_train)

    data_test = set_age_level(data_test)

    dummies_agelevel = pd.get_dummies(data_train['Agelevel'], prefix='Agelevel')

    data_train = pd.concat([data_train, dummies_agelevel], axis=1)

    dummies_test_agelevel = pd.get_dummies(data_test['Agelevel'], prefix='Agelevel')

    data_test = pd.concat([data_test, dummies_test_agelevel], axis=1)



    data_train, data_test = set_data_standard(data_train, data_test)



    return data_train, data_test

data_train, data_test = data_process()

train_df = data_train.filter(regex='Survived|Name_lenth_.*|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass|Title_.*|Agelevel_.*')

train_np = train_df.values

y = train_np[:, 0]

X = train_np[:, 1:]



test_df = data_test.filter(regex='Name_lenth_.*|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass|Title_.*|Agelevel_.*')

X_test = test_df.values
def get_training_goals(X, y, X_test):

    # 集成学习

    voting_clf = VotingClassifier(estimators=[

        ('log_clf', LogisticRegression(penalty='l1', tol=1e-4, C=10)),

        ('svm_clf', SVC(C=0.5, kernel='rbf', gamma=0.1, tol=1e-3, probability=True, random_state=0)),

        ('xgb_clf', XGBClassifier(learning_rate=0.01, max_depth=4, n_estimators=100, random_state=0)),

        ('rf_clf', RandomForestClassifier(n_estimators=500, max_leaf_nodes=25, random_state=0))

    ], voting='soft')

    voting_clf.fit(X, y)

    predict_y = voting_clf.predict(X_test)

    return predict_y
predict = get_training_goals(X, y, X_test)

result = pd.DataFrame({'PassengerId': data_test['PassengerId'].values, 'Survived': predict.astype(np.int64)})

result.to_csv('predict_xgb.csv', index=False)