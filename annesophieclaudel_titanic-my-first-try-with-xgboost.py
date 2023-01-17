import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier

import xgboost

from sklearn.model_selection import RandomizedSearchCV   

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import time

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')





train_df.head()
def feature_eng(df):

    df['TicketLetter'] = df['Ticket'].apply(lambda x : str(x)[0]) 

    df['TicketLetter'] = df['TicketLetter'].apply(lambda x : re.sub('[0-9]','N',x))

    df['Title'] = df.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

    normalized_titles = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Mme":        "Mrs",

        "Mlle":       "Miss",

        "Ms":         "Mrs",

        "Mr" :        "Mr",

        "Mrs" :       "Mrs",

        "Miss" :      "Miss",

        "Master" :    "Master",

        "Lady" :      "Royalty"

    }



    df['Title'] = df['Title'].map(normalized_titles)

    df["FamilySize"] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = 1

    df['IsAlone'].loc[df['FamilySize'] > 1] = 0

    df["Embarked"] = df["Embarked"].fillna("S")

    df["Age"] = df.groupby(['Sex','Pclass','Title'])["Age"].transform(lambda x: x.fillna(x.median()))

    df["Cabin"] = df["Cabin"].str[0:1]

    df["Cabin"] = df["Cabin"].fillna('T')

    df['Fare'] = df['Fare'].fillna(-1)



    columnshoice = ['Pclass', 'Sex', 'Age', 'SibSp',

           'Parch', 'Fare', 'Embarked', 'TicketLetter', 'Title',

           'FamilySize', 'IsAlone']

    return pd.get_dummies(df[columnshoice])



y = train_df["Survived"]

X = feature_eng(train_df)

test = feature_eng(test_df)

# split between X , y and test

print(X.columns)
X.head()
######################################################

# xgboost with a grid

######################################################

grid = {

    'n_estimators': range(4, 20),

    'max_depth': range(12, 20),

    'learning_rate': [.22,.23,.24,.25,.3],

    'colsample_bytree': [.6, .7, .8, .9, 1]

}

xgb_cla = xgboost.XGBClassifier(random_state=42)

xgb_random = RandomizedSearchCV(param_distributions=grid, 

                                    estimator = xgb_cla, scoring = "accuracy", 

                                    verbose = 1, n_iter = 200, cv = 4)





# Fit randomized_mse to the data

xgb_random.fit(X, y)



# Print the best parameters and lowest RMSE

print("Best parameters found: ", xgb_random.best_params_)

print("Best accuracy found: ", xgb_random.best_score_)

y_pred = xgb_random.predict(X)

conf_mx = confusion_matrix(y,y_pred)

print(conf_mx)

print ("------------------------------------------")

print (" Accuracy    : ", accuracy_score(y,y_pred))

print (" Precision   : ", precision_score(y,y_pred))

print (" Sensitivity : ", recall_score(y,y_pred))

print ("------------------------------------------")

######################################################

# Submission

######################################################



y_sub= xgb_random.predict(test)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_sub

    })

submission.to_csv('submission.csv', index=False)