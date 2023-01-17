import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('../input/train.csv')

holdout = pd.read_csv('../input/test.csv')
def process_missing(df):

    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())

    df["Embarked"] = df["Embarked"].fillna("S")

    return df



def process_age(df):

    df["Age"] = df["Age"].fillna(-0.5)

    cut_points = [-1,0,5,12,18,35,60,100]

    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df



def process_fare(df):

    cut_points = [-1,12,50,100,1000]

    label_names = ["0-12","12-50","50-100","100+"]

    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)

    return df



def process_cabin(df):

    df["Cabin_type"] = df["Cabin"].str[0]

    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")

    df = df.drop('Cabin',axis=1)

    return df



def process_titles(df):

    titles = {

        "Mr" :         "Mr",

        "Mme":         "Mrs",

        "Ms":          "Mrs",

        "Mrs" :        "Mrs",

        "Master" :     "Master",

        "Mlle":        "Miss",

        "Miss" :       "Miss",

        "Capt":        "Officer",

        "Col":         "Officer",

        "Major":       "Officer",

        "Dr":          "Officer",

        "Rev":         "Officer",

        "Jonkheer":    "Royalty",

        "Don":         "Royalty",

        "Sir" :        "Royalty",

        "Countess":    "Royalty",

        "Dona":        "Royalty",

        "Lady" :       "Royalty"

    }

    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    df["Title"] = extracted_titles.map(titles)

    return df



def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



def add_feature(df):

    df['isalone'] = df['SibSp']==0

    return df



def clean_df(df):

    df = process_missing(df)

    df = process_age(df)

    df = process_fare(df)

    df = process_titles(df)

    df = process_cabin(df)

    for col in ['Age_categories', 'Fare_categories', 'Title', 'Cabin_type', 'Sex']:    

        df = create_dummies(df, col)

    df = add_feature(df)

    return df
train = clean_df(train)

holdout = clean_df(holdout)
from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier



def select_features(df):

    df = df.select_dtypes(include=['number'])

    df = df.dropna()

    all_X = df.drop(['PassengerId', 'Survived'], axis=1)

    all_y = df['Survived']

    rf = RandomForestClassifier()

    selector = RFECV(rf, cv=10)

    selector.fit(all_X, all_y)

    optimized_columns = all_X.columns[selector.support_]

    print(optimized_columns)

    return optimized_columns



best_features = select_features(train)
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



def select_model(df, features):

    all_X = df[features]

    all_y = df['Survived']

    

    models = []

    models.append( 

    {

        'name': 'LogisticRegression',

        'estimator': LogisticRegression(),

        'hyper_parameters':

        {

            'solver': ['newton-cg', 'lbfgs', 'liblinear']

        }

    })

    models.append(

    {

        'name': 'KNeighborsClassifier',

        'estimator': KNeighborsClassifier(),

        'hyper_parameters':

        {

            'n_neighbors': range(1, 20, 2),

            'weights': ['distance', 'uniform'],

            'algorithm': ["ball_tree", "kd_tree", "brute"],

            'p': [1,2]

        }

    })

    models.append(

    {

        'name': 'RandomForestClassifier',

        'estimator': RandomForestClassifier(),

        'hyper_parameters':

        {

            "n_estimators": [4, 6, 9],

            "criterion": ["entropy", "gini"],

            "max_depth": [2, 5, 10],

            "max_features": ["log2", "sqrt"],

            "min_samples_leaf": [1, 5, 8],

            "min_samples_split": [2, 3, 5]

        }

    })

    

    for model in models:

        print(model['name'])

        grid = GridSearchCV(model['estimator'], param_grid=model['hyper_parameters'], cv=10)

        grid.fit(all_X, all_y)

        model['best_estimator'] = grid.best_estimator_

        print(grid.best_params_)

        print(grid.best_score_)

    best_rf = grid.best_estimator_

    return models



best_models = select_model(train, best_features)
from sklearn.model_selection import cross_val_score



all_X = train[best_features]

all_y = train['Survived']

accuracies = {}

for model in best_models:

    model['best_estimator'].fit(all_X, all_y)

    scores = cross_val_score(model['best_estimator'], all_X, all_y, cv=10)

    accuracy = scores.mean()

    accuracies[model['name']] = accuracy



print(accuracies)
def save_submission_file(models, columns, filename='untitled.csv'):

    for model in models:

        model['best_estimator'].fit(all_X, all_y)

        holdout_predictions = model['best_estimator'].predict(holdout[columns])

        submission = pd.DataFrame(

            {

                'PassengerID': holdout['PassengerId'],

                'Survived': holdout_predictions

            })

        submission.to_csv('submission_'+model['name']+'.csv', index=False)

save_submission_file(best_models, best_features, filename='abc.csv')