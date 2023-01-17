# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fancyimpute import KNN



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def create_dummies(df, column_name, pref=False):

    dummies = pd.get_dummies(df[column_name], prefix=column_name if pref else None)

    df = df.join(dummies)

    return df





def prepare_data(train, test):

    # replace numerical categorical features with indicators

    train = create_dummies(train, "Pclass", pref=True)

    test = create_dummies(test, "Pclass", pref=True)

    train = create_dummies(train, "Sex")

    test = create_dummies(test, "Sex")



    # deduce missing ports from fares

    train.loc[train.Embarked != train.Embarked, "Embarked"] = "C"

    test.loc[test.Embarked != test.Embarked, "Embarked"] = "C"

    train = create_dummies(train, "Embarked", pref=True)

    test = create_dummies(test, "Embarked", pref=True)



    # deduce missing fare by port

    full = pd.concat([train, test], sort=False)

    imp_fare = full.loc[(full.Embarked == 'S') & (full.Pclass == 3), "Fare"].mean()

    test.loc[test.Fare != test.Fare, "Fare"] = round(imp_fare, 2)



    # create features by aggregation

    train_fsize = train.Parch + train.SibSp

    train = train.join(train_fsize.rename('Fsize'))

    test_fsize = test.Parch + test.SibSp

    test = test.join(test_fsize.rename('Fsize'))



    # engineer features from existing with regexps

    train_titles = train.Name.str.replace('(.*, )|(\\..*)', '').rename('Title')

    test_titles = test.Name.str.replace('(.*, )|(\\..*)', '').rename('Title')

    train = train.join(train_titles)

    test = test.join(test_titles)

    rare_title = ["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess"]

    train.loc[train.Title.isin(rare_title), "Title"] = "Rare"

    test.loc[test.Title.isin(rare_title), "Title"] = "Rare"

    train.loc[train.Title.isin(["Mlle", "Ms"]), "Title"] = "Miss"

    test.loc[test.Title.isin(["Mlle", "Ms"]), "Title"] = "Miss"

    train.loc[train.Title == "Mme", "Title"] = "Mrs"

    test.loc[test.Title == "Mme", "Title"] = "Mrs"

    train = create_dummies(train, 'Title')

    test = create_dummies(test, 'Title')

    train.drop(['Title'], axis=1, inplace=True)

    test.drop(['Title'], axis=1, inplace=True)



    # drop useless columns

    train.drop(['Sex', 'Name', 'Pclass', 'Embarked', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

    test.drop(['Sex', 'Name', 'Pclass', 'Embarked', 'Ticket', 'Cabin'], axis=1, inplace=True)



    # impute Age using knn

    k_train = int(np.sqrt(train.shape[0]))

    k_test = int(np.sqrt(test.shape[0]))

    train_features = train.drop(['Survived'], axis=1).select_dtypes(include=[np.float, np.int])

    test_features = test.select_dtypes(include=[np.float, np.int])

    filled_ages_train = pd.DataFrame(KNN(k=k_train).fit_transform(train_features)).loc[:, 0]

    filled_ages_test = pd.DataFrame(KNN(k=k_test).fit_transform(test_features)).loc[:, 0]

    train.Age = round(filled_ages_train, 1)

    test.Age = round(filled_ages_test, 1)



    # add two age categories

    cut_points = [0, 16, 100]

    label_names = ["Child", 'Adult']

    test["AgeGroup"] = pd.cut(test["Age"], cut_points, labels=label_names)

    train["AgeGroup"] = pd.cut(train["Age"], cut_points, labels=label_names)

    test = create_dummies(test, "AgeGroup")

    train = create_dummies(train, "AgeGroup")

    test.drop(['AgeGroup'], axis=1, inplace=True)

    train.drop(['AgeGroup'], axis=1, inplace=True)



    train_x = train[train.columns[1:]]

    train_y = train['Survived']

    test_x = test



    return train_x, train_y, test_x
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train_x, train_y, test_x = prepare_data(train, test)
random_forest = RandomForestClassifier(n_estimators=200)

scores = cross_val_score(random_forest, train_x, train_y, cv=10)

print(f"random forest scrore: {np.mean(scores):.5f}±{np.std(scores):.5f}")
model = RandomForestClassifier(n_estimators=200)

model.fit(train_x, train_y)

test_predictions = model.predict(test_x[train_x.columns])

test_ids = test["PassengerId"]

submission_df = {"PassengerId": test_ids, "Survived": test_predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv('titanic_submission.csv', index=False)