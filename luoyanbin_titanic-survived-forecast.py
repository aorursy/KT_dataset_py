# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def train_preprocess():
    # 读取train.csv为pandas.DataFrame
    train = pd.read_csv('../input/train.csv')

    # Pclass
    # One-hot编码
    train['P1'] = np.array(train['Pclass'] == 1).astype(np.int32)
    train['P2'] = np.array(train['Pclass'] == 2).astype(np.int32)
    train['P3'] = np.array(train['Pclass'] == 3).astype(np.int32)

    # Sex
    # 把male/female转换成1/0
    train['Sex'] = [1 if i == 'male' else 0 for i in train.Sex]

    # SibSp and Parch
    # 'FamilySize'：家庭成员人数
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    # 'IsAlone'：是否只身一人
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

    # Embarked
    train['Embarked'] = train['Embarked'].fillna('S')
    # One-hot编码
    train['E1'] = np.array(train['Embarked'] == 'S').astype(np.int32)
    train['E2'] = np.array(train['Embarked'] == 'C').astype(np.int32)
    train['E3'] = np.array(train['Embarked'] == 'Q').astype(np.int32)

    # Fare
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    train['CategoricalFare'].cat.categories = [1, 2, 3, 4]
    # one-hot编码
    train['F1'] = np.array(train['CategoricalFare'] == 1).astype(np.int32)
    train['F2'] = np.array(train['CategoricalFare'] == 2).astype(np.int32)
    train['F3'] = np.array(train['CategoricalFare'] == 3).astype(np.int32)
    train['F4'] = np.array(train['CategoricalFare'] == 4).astype(np.int32)

    # Age
    age_avg = train['Age'].mean()
    age_std = train['Age'].std()
    age_null_count = train['Age'].isnull().sum()
    age_null_random_list = np.random.randint(
        age_avg - age_std, age_avg + age_std, size=age_null_count)
    train['Age'][np.isnan(train['Age'])] = age_null_random_list
    train['Age'] = train['Age'].astype(int)
    train['CategoricalAge'] = pd.qcut(train['Age'], 5)
    train['CategoricalAge'].cat.categories = [1, 2, 3, 4, 5]
    train['A1'] = np.array(train['CategoricalAge'] == 1).astype(np.int32)
    train['A2'] = np.array(train['CategoricalAge'] == 2).astype(np.int32)
    train['A3'] = np.array(train['CategoricalAge'] == 3).astype(np.int32)
    train['A4'] = np.array(train['CategoricalAge'] == 4).astype(np.int32)
    train['A5'] = np.array(train['CategoricalAge'] == 5).astype(np.int32)

    # Name
    train['Title'] = train['Name'].apply(get_title)
    train['Title'] = train['Title'].replace([
        'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
        'Jonkheer', 'Dona'
    ], 'Rare')
    train['Title'] = train['Title'].replace('Mlle', 'Miss')
    train['Title'] = train['Title'].replace('Ms', 'Miss')
    train['Title'] = train['Title'].replace('Mme', 'Mrs')
    train['T1'] = np.array(train['Title'] == 'Master').astype(np.int32)
    train['T2'] = np.array(train['Title'] == 'Miss').astype(np.int32)
    train['T3'] = np.array(train['Title'] == 'Mr').astype(np.int32)
    train['T4'] = np.array(train['Title'] == 'Mrs').astype(np.int32)
    train['T5'] = np.array(train['Title'] == 'Rare').astype(np.int32)

    # 数据清洗
    train_x = train[[
        'P1', 'P2', 'P3', 'Sex', 'IsAlone', 'E1', 'E2', 'E3', 'F1', 'F2', 'F3',
        'F4', 'A1', 'A2', 'A3', 'A4', 'A5', 'T1', 'T2', 'T3', 'T4', 'T5'
    ]]
    train_y_ = train['Survived'].values.reshape(len(train), 1)

    return train_x, train_y_


def test_preproces():
    # 读取test.csv为pandas.DataFrame
    test = pd.read_csv('../input/test.csv')

    # Pclass
    test['P1'] = np.array(test['Pclass'] == 1).astype(np.int32)
    test['P2'] = np.array(test['Pclass'] == 2).astype(np.int32)
    test['P3'] = np.array(test['Pclass'] == 3).astype(np.int32)

    # Sex
    # 把male/female转换成1/0
    test['Sex'] = [1 if i == 'male' else 0 for i in test.Sex]

    # SibSp and Parch
    # 'FamilySize'：家庭成员人数
    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    # 'IsAlone'：是否只身一人
    test['IsAlone'] = 0
    test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

    # Embarked
    test['Embarked'] = test['Embarked'].fillna('S')
    # One-hot编码
    test['E1'] = np.array(test['Embarked'] == 'S').astype(np.int32)
    test['E2'] = np.array(test['Embarked'] == 'C').astype(np.int32)
    test['E3'] = np.array(test['Embarked'] == 'Q').astype(np.int32)

    # Fare
    test['CategoricalFare'] = pd.qcut(test['Fare'], 4)
    test['CategoricalFare'].cat.categories = [1, 2, 3, 4]
    # one-hot编码
    test['F1'] = np.array(test['CategoricalFare'] == 1).astype(np.int32)
    test['F2'] = np.array(test['CategoricalFare'] == 2).astype(np.int32)
    test['F3'] = np.array(test['CategoricalFare'] == 3).astype(np.int32)
    test['F4'] = np.array(test['CategoricalFare'] == 4).astype(np.int32)

    # Age
    age_avg = test['Age'].mean()
    age_std = test['Age'].std()
    age_null_count = test['Age'].isnull().sum()
    age_null_random_list = np.random.randint(
        age_avg - age_std, age_avg + age_std, size=age_null_count)
    test['Age'][np.isnan(test['Age'])] = age_null_random_list
    test['Age'] = test['Age'].astype(int)
    test['CategoricalAge'] = pd.qcut(test['Age'], 5)
    test['CategoricalAge'].cat.categories = [1, 2, 3, 4, 5]
    test['A1'] = np.array(test['CategoricalAge'] == 1).astype(np.int32)
    test['A2'] = np.array(test['CategoricalAge'] == 2).astype(np.int32)
    test['A3'] = np.array(test['CategoricalAge'] == 3).astype(np.int32)
    test['A4'] = np.array(test['CategoricalAge'] == 4).astype(np.int32)
    test['A5'] = np.array(test['CategoricalAge'] == 5).astype(np.int32)

    # Name
    test['Title'] = test['Name'].apply(get_title)
    test['Title'] = test['Title'].replace([
        'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
        'Jonkheer', 'Dona'
    ], 'Rare')
    test['Title'] = test['Title'].replace('Mlle', 'Miss')
    test['Title'] = test['Title'].replace('Ms', 'Miss')
    test['Title'] = test['Title'].replace('Mme', 'Mrs')
    test['T1'] = np.array(test['Title'] == 'Master').astype(np.int32)
    test['T2'] = np.array(test['Title'] == 'Miss').astype(np.int32)
    test['T3'] = np.array(test['Title'] == 'Mr').astype(np.int32)
    test['T4'] = np.array(test['Title'] == 'Mrs').astype(np.int32)
    test['T5'] = np.array(test['Title'] == 'Rare').astype(np.int32)

    # 数据清洗
    test_x = test[[
        'P1', 'P2', 'P3', 'Sex', 'IsAlone', 'E1', 'E2', 'E3', 'F1', 'F2', 'F3',
        'F4', 'A1', 'A2', 'A3', 'A4', 'A5', 'T1', 'T2', 'T3', 'T4', 'T5'
    ]]

    test_y = test[['PassengerId']]

    return test_x, test_y


# 读取数据集
def read_data_sets():
    dataset = {}
    train_x, train_y_ = train_preprocess()
    test_x, test_y = test_preproces()
    dataset['train_x'] = train_x
    dataset['train_y_'] = train_y_
    dataset['test_x'] = test_x
    dataset['test_y'] = test_y
    return dataset

# read dataset
dataset = read_data_sets()
X_train = dataset['train_x']
Y_train = dataset['train_y_']
X_test = dataset['test_x']
Submission = dataset['test_y']
# # Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict_praba(X_test)
# logreg.score(X_train, Y_train)
# # AdaBoost
# abt = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME",
#                          n_estimators=200, learning_rate=0.8)

# abt.fit(X_train, Y_train)
# Y_pred = abt.predict(X_test)
# abt.score(X_train, Y_train)
# GradientBoosting
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, Y_train)
Y_pred = gbdt.predict(X_test)


Submission['Survived'] = Y_pred
Submission.to_csv('gbdt_20180712_1412.csv', index=False)