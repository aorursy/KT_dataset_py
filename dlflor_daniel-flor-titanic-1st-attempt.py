import pandas as pd

import matplotlib.pyplot as plt

import plotly

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import time

from sklearn.feature_selection import SelectKBest

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

import numpy as np
holdout = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.columns
holdout.columns
# Separate Survived label since it only exists in the train dataset

survived = train["Survived"]

train = train.drop("Survived",axis=1)
holdout.shape
train.shape
## concatenate all data to guarantee that holdout and train datasets have the same number of

## features after one hote encoding

all_data = pd.concat([train,holdout],axis=0)
all_data.describe()
def process_ticket(df):

    # see https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

    # Get first digit of tickets, which groups people by their ticket

    Ticket = []

    for i in list(df.Ticket):

        if not i.isdigit():

            #Take prefix

            Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) 

        else:

            Ticket.append("X")

    df["Ticket"] = Ticket

    return df



def process_missing(df):



    # process missing values in the features that need it

    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

    df["Embarked"] = df["Embarked"].fillna("S")

    return df



def process_age(df):



    # group people by age



    df["Age"] = df["Age"].fillna(-0.5)

    cut_points = [-1,0,5,12,18,35,60,100]

    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    

    df = df.drop("Age",axis=1)

    

    return df



def process_fare(df):



    # group fares by price



    cut_points = [-1,12,50,100,1000]

    label_names = ["0-12","12-50","50-100","100+"]

    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)

    

    df = df.drop("Fare",axis=1)

    

    return df



def process_cabin(df):



    # Group people by the firs letter of cabin, whihc indicates in which deck 

    # was their cabin



    df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'Unknown' for i in df["Cabin"]])



    # aux  = df["Cabin"]

    # df["Cabin"] = aux.str[0]

    # df["Cabin"] = df["Cabin"].fillna("Unknown")



    return df



def process_family_size(df):



   # Uses information in SibSp and Parch to compute the family size of an someone,

   # including themselves



    Fsize = df["SibSp"] + df["Parch"] + 1

    

    cut_points = [-1,1,2,4,20]

    label_names = ["Single","Small","Medium","Big"]

    df["Fsize"] = pd.cut(Fsize,cut_points,labels=label_names)



  # dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

  # dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

  # dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

  # dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

    return df



def process_titles(df):



    # Extract peoples titles from their names. More influent passengers might have

    # priority in escaping



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



    # Enconde variables with one hot encoding



    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df
# Complete preprocessing function



def pre_process(df):

    df = process_ticket(df)

    df = process_missing(df)

    df = process_age(df)

    df = process_fare(df)

    df = process_titles(df)

    df = process_cabin(df)

    df = process_family_size(df)



    for col in ["Age_categories","Fare_categories",

                "Title","Cabin","Sex","Ticket","Pclass","Fsize","Embarked"]:

        df = create_dummies(df,col)

    

    df = df.drop(columns=["Age_categories","Fare_categories",

                "Title","Cabin","Sex","Ticket","Pclass","Fsize",'Name','Embarked'],axis=1)

    

    return df



# preprocess all data

all_data_processed = pre_process(all_data)



# separate training and holdout data, and add back "Survived" label to training data

processed_data = all_data_processed.iloc[:891]

processed_data = pd.concat([processed_data,survived],axis=1)

holdout = all_data_processed.iloc[891:]
from sklearn.model_selection import train_test_split



seed = 42

X_train, X_test, y_train, y_test = train_test_split(processed_data.drop(columns=["Survived",'PassengerId'],axis=1),

                                                    processed_data["Survived"],

                                                    test_size=0.20,

                                                    random_state=seed,

                                                    shuffle=True,

                                                    stratify=processed_data["Survived"])
pipe = Pipeline(steps = [

                          # ("fs",SelectKBest()),

                         ("clf",XGBClassifier())])



# During testing, it was found out that using all features worked best



search_space = [

                {"clf":[RandomForestClassifier()],

                 "clf__n_estimators": [250,100,150,200],

                 "clf__criterion": ["entropy","gini"],

                 "clf__max_depth": [4,6,8],

                 "clf__max_leaf_nodes": [16,32,8,64],

                 "clf__random_state": [seed],

                 },

                {"clf":[XGBClassifier()],

                 "clf__n_estimators": [100,150,200],

                 "clf__max_depth": [6,8],

                 "clf__learning_rate": [0.001,0.1],

                 "clf__random_state": [seed],

                 "clf__subsample": [1.0],

                 }

                ]

scoring = {'Accuracy': make_scorer(accuracy_score)}

num_folds = 10



# create grid search

kfold = StratifiedKFold(n_splits=num_folds,random_state=seed)



# return_train_score=True

grid = GridSearchCV(estimator=pipe, 

                    param_grid=search_space,

                    cv=kfold,

                    scoring=scoring,

                    return_train_score=True,

                    n_jobs=-1,

                    refit="Accuracy")



tmp = time.time()



# fit grid search

best_model = grid.fit(X_train,y_train)



print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
print("Best: %f using %s" % (best_model.best_score_,best_model.best_params_))
result = pd.DataFrame(best_model.cv_results_)

result.head()
result_acc = result[['mean_train_Accuracy', 'std_train_Accuracy','mean_test_Accuracy', 'std_test_Accuracy','rank_test_Accuracy']].copy()

result_acc["std_ratio"] = result_acc.std_test_Accuracy/result_acc.std_train_Accuracy

result_acc.sort_values(by="rank_test_Accuracy",ascending=True)
# best model

predict_first = best_model.best_estimator_.predict(X_test)

print(accuracy_score(y_test, predict_first))
# Drop the PassengerId before predicting!

predict_final = best_model.best_estimator_.predict(holdout.drop(columns=["PassengerId"],axis=1))
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": predict_final}

submission = pd.DataFrame(submission_df)



submission.to_csv("submission.csv",index=False)
files.download('submission.csv') 