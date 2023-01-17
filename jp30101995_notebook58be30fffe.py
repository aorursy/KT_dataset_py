# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)

train_df.info()

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



# make bins for ages and name them for ease:

def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



#keep only the first letter (similar effect as making bins/clusters):

def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



# make bins for fare prices and name them:

def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



# createa all in transform_features function to be called later:

def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = drop_features(df)

    return df



# create new dataframe with different name:

train_df2 = transform_features(train_df)

test_df2 = transform_features(test_df)



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

    

train_df2, test_df2 = encode_features(train_df2, test_df2)

train_df2.head()

train_df2.info()



X_train = train_df2.drop(["Survived","PassengerId"], axis = 1)

Y_train = train_df2["Survived"]

X_test = test_df2.drop("PassengerId", axis = 1).copy()

X_train.info()



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log



from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree



from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



acc_random_forest



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.describe()

os.getcwd()

submission.to_csv('../kaggle/working/submission.csv', index=False)

#sns.barplot(x="Age", y="Survived", hue="Sex", data=train_df2, palette={"male": "blue", "female": "pink"});





#sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train_df2, palette={"male": "blue", "female": "pink"});

#train_df[train_df.Sex == "female"].Survived.value_counts().plot(kind = "bar", title = "Survival among female")



# Any results you write to the current directory are saved as output.
