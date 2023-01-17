import pandas as pd

import numpy as np

import xgboost as xgb

import math

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
def build(df, is_train):

    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    median_ages = np.zeros((2,3))

    for i in range(0,2):

        for j in range(0,3):

            median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']

    for i in range(0, 2):

        for j in range(0, 3):

            df.loc[ (df['Age'].isnull()) & (df['Gender'] == i) & (df['Pclass'] == j+1),'AgeFill'] = median_ages[i,j]

    df['AgeIsNull'] = pd.isnull(df['Age']).astype(int)

    df['FamilySize'] = df['SibSp'] + df['Parch']

    df['Age*Class'] = df['AgeFill'] * df['Pclass']

    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 

    if is_train:

        df = df.dropna()

        y = df["Survived"].values

        del df["Survived"]

        del df["PassengerId"]

        X = df.values

        return X, y

    else:

        df.fillna(0, inplace=True)

        del df["PassengerId"]

        X = df.values

        return X



#交差検証

def cross_val(clf, X, y, K, rs):

    kf = KFold(n_splits = K, shuffle = True, random_state=rs)

#    kf = KFold(len(y), K, shuffle=True, random_state=rs)

#    KFold(n_splits=3, shuffle=False, random_state=None)[source]

    scores = []

    clfs = []

    for train_index, valid_index in kf.split(X):

        X_train, X_valid = X[train_index], X[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_valid)

        score = accuracy_score(y_valid, y_pred)

        scores.append(score)

        clfs.append(clf)

    return clfs, scores
train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)

X_train, y_train = build(train_df, is_train=True)

X_test = build(test_df, is_train=False)

#モデル1作成

clfs_1, scores_1 = cross_val(RandomForestClassifier(n_estimators = 100), X_train, y_train, 3, 0)

clfs_2, scores_2 = cross_val(xgb.XGBClassifier(), X_train, y_train, 3, 0)

clfs_3, scores_3 = cross_val(LogisticRegression(), X_train, y_train, 3, 0)



clfs = clfs_1 + clfs_2 + clfs_3
#モデル2作成：probabirityをfeatureとしてモデル作成

y_pred_list = np.zeros((len(X_train), len(clfs)))



for i, clf in enumerate(clfs):

    y_pred_list[:, i] = clf.predict_proba(X_train)[:, 1]

#予測値をフィーチャーに

clfs2, scores2 = cross_val(RandomForestClassifier(n_estimators = 100), y_pred_list, y_train, 3, 0)
#モデル2用inputテストデータ作成

y_pred_list2 = np.zeros((len(X_test), len(clfs)))



for i, clf in enumerate(clfs):

    y_pred_list2[:, i] = clf.predict_proba(X_test)[:,1]

    
#モデル3用inputテストデータ作成

y_pred_list3 = np.zeros((len(X_test), len(clfs)))



for i, clf in enumerate(clfs2):

    y_pred_list3[:, i] = clf.predict_proba(y_pred_list2)[:,1]
y_pred_mean = np.mean(y_pred_list3, axis=1)
submit = np.asarray(y_pred_mean)

test_df["Survived"] = submit

test_df[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)

#submit.to_csv("submission.csv", index=False)