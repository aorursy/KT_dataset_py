# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('../input/train.csv')

holdout = pd.read_csv('../input/test.csv')
train.head()
def preprocess(df):

# Missing data

    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())

    df["Embarked"] = df["Embarked"].fillna("S")

    

# Age

    df["Age"] = df["Age"].fillna(-0.5)

    cut_points = [-1,0,5,12,18,35,60,100]

    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    

# Fare

    cut_points = [-1,12,50,100,1000]

    label_names = ["0-12","12-50","50-100","100+"]

    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)

    

# Cabin

    df["Cabin_type"] = df["Cabin"].str[0]

    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")

    df = df.drop('Cabin',axis=1)

    

    return df



def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df
def preprocess2(df):

    df = preprocess(df)



    for col in ["Age_categories","Fare_categories",

                "Cabin_type","Sex"]:

        df = create_dummies(df,col)

    

    return df



train = preprocess2(train)

holdout = preprocess2(holdout)
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV

# select numerical columns and remove columns with null value

train = train.select_dtypes([np.number]).dropna(axis = 1)

all_X = train.drop(["PassengerId","Survived"], axis = 1)

all_y = train["Survived"]



clf = RandomForestClassifier(random_state = 1)

selector = RFECV(clf, cv = 10)

selector.fit(all_X, all_y)



best_columns = list(all_X.columns[selector.support_])

print(best_columns)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



all_X = train[best_columns]

all_y = train["Survived"]



models = [

        {

            "name": "LogisticRegression",

            "estimator": LogisticRegression(),

            "hyperparameters":

                {

                    "solver": ["newton-cg", "lbfgs", "liblinear"]

                }

        },

        {

            "name": "KNeighborsClassifier",

            "estimator": KNeighborsClassifier(),

            "hyperparameters":

                {

                    "n_neighbors": range(1,20,2),

                    "weights": ["distance", "uniform"],

                    "algorithm": ["ball_tree", "kd_tree", "brute"],

                    "p": [1,2]

                }

        },

        {

            "name": "RandomForestClassifier",

            "estimator": RandomForestClassifier(random_state=1),

            "hyperparameters":

                {

                    "n_estimators": [4, 6, 9],

                    "criterion": ["entropy", "gini"],

                    "max_depth": [2, 5, 10],

                    "max_features": ["log2", "sqrt"],

                    "min_samples_leaf": [1, 5, 8],

                    "min_samples_split": [2, 3, 5]



                }

        }

    ]



for model in models:

    print(model["name"])

    print('-'*len(model["name"]))

    

    grid = GridSearchCV(model["estimator"],

                        param_grid=model["hyperparameters"],

                        cv=10)

    grid.fit(all_X, all_y)

    model["best_params"] = grid.best_params_

    model["best_score"] = grid.best_score_

    model["best_model"] = grid.best_estimator_

    

    print("Best Score: {}".format(model["best_score"]))

    print("Best Params: {}".format(model["best_params"]))
best_model = models[2]["best_model"]

holdout_data = holdout[best_columns]

predictions = best_model.predict(holdout_data)



holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv('submission.csv', index=False)
