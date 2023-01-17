import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



titanic_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



titanic_df = titanic_df.drop(['PassengerId'], axis = 1)
sns.factorplot("Sex", data = titanic_df, kind = 'count',hue = "Survived")
sns.factorplot("SibSp", data = titanic_df, kind = 'count',hue = "Survived")
sns.factorplot("Parch", data = titanic_df, kind = 'count',hue = "Survived")
def df_cleanup(df):

    df = df.drop(['Cabin', 'Name', 'Ticket'], axis = 1)

    

    for passenger in df[(df['Age'].isnull())].index:

        df.loc[passenger, 'Age'] = np.average(df[(df['Age'].notnull())]['Age'])



    for passenger in df[(df['Fare'].isnull())].index:

        df.loc[passenger, 'Fare'] = np.average(df[(df['Fare'].notnull())]['Fare'])



    df = sex_category(df)

    df = embark_category(df)

    df = family_category(df)

    df = child_category(df)

    

    df[['Sex','Embarked']] = df[['Sex','Embarked']].apply(pd.to_numeric)

    return df
def sex_category(df):

    df.loc[(df['Sex'] == 'male'), 'Sex'] = 0

    df.loc[(df['Sex'] == 'female'), 'Sex'] = 1

    df.loc[(df['Sex'].isnull()), 'Sex'] = 2

    return df
def embark_category(df):

    df.loc[(df['Embarked'] == 'S'), 'Embarked'] = 0

    df.loc[(df['Embarked'] == 'C'), 'Embarked'] = 1

    df.loc[(df['Embarked'] == 'Q'), 'Embarked'] = 2

    df.loc[(df['Embarked'].isnull()), 'Embarked'] = 3

    return df
def family_category(df):

    df["FamilySize"] = df["SibSp"] + df["Parch"]

    return df
def child_category(df):

    df['Children'] = df['Age'].map(lambda x: 1 if x < 6.0 else 0)

    return df
titanic_df = df_cleanup(titanic_df)

test_df = df_cleanup(test_df)
features_list = list(titanic_df.columns.values)



X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, chi2

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn import tree



classifier = "Rforest"



if classifier == "Ada":

    pipe = Pipeline([

           ('k_best', SelectKBest()),

           ('classify', AdaBoostClassifier())

        ])

        

    param_grid = ([

            {

                'k_best__k': [2,3,4,5,6,7,8],

                'classify__n_estimators':[10, 15],

                'classify__algorithm': ['SAMME', 'SAMME.R'],

                'classify__learning_rate': [0.2, 0.5, 1.0, 1.5, 2.0]

            }

        ])

        

elif classifier == "DTree":

    pipe = Pipeline([

           ('k_best', SelectKBest()),

           ('classify', tree.DecisionTreeClassifier())

        ])

        

    param_grid = ([

            {

                'k_best__k': ['all'],

                'classify__max_features': [0.5, 1.0, 'sqrt', 'auto'],

                'classify__max_depth': [4, 6, 8, 10, 12, None]

            }

        ])

        

elif classifier == "Rforest":

    pipe = Pipeline([

           ('k_best', SelectKBest()),

           ('classify', RandomForestClassifier())

        ])

        

    param_grid = ([

            {

                'k_best__k': [2,3,4,5,6,7,8],

                'classify__criterion':['gini', 'entropy'],

                'classify__max_features': [0.5, 1.0, 'sqrt', 'auto'],

                'classify__max_depth': [4, 6, 8, 10, 12, None]

            }

        ])





sss = StratifiedShuffleSplit()

clf = GridSearchCV(pipe, param_grid = param_grid, cv = sss, scoring='roc_auc')

clf.fit(X_train, Y_train)



print("(clf.best_estimator_.steps): ", (clf.best_estimator_.steps))

print( "(clf.best_score_): ", (clf.best_score_))

print( "(clf.best_params_): ", (clf.best_params_))

print( "(clf.scorer_): ", (clf.scorer_))



chosen_features = clf.best_estimator_.named_steps['k_best'].get_support(indices=True)

finalFeatureList = [features_list[i+1] for i in chosen_features]

print(finalFeatureList)
Y_pred = clf.predict(X_test)



print(clf.score(X_train, Y_train))



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)