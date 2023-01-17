# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load data

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
# Feature extraction

for df in (train_df, test_df):

    df["relatives"] = df["SibSp"]+df["Parch"]

    df["deck"] = df["Cabin"].str[0:1]
# Fill NAs

mean_age = train_df["Age"].mean()

embarked_mode = train_df["Embarked"].mode()

for df in (train_df, test_df):

    df["age_filled"] = df["Age"].fillna(mean_age)

    df["embarked_filled"] = df["Embarked"].fillna(embarked_mode)

    df["fare_filled"] = df["Fare"].fillna(0)
# Transform to categoricals

for df in (train_df, test_df):

    df["Sex"] = df["Sex"].astype("category")

    df["embarked_cat"] = df["embarked_filled"].astype("category")

    df["embarked_cat"].cat.categories = ["Cherbourg", "Queenstown", "Southampton"]

    df["deck_cat"] = df["deck"].astype("category")

train_df["embarked_cat"].cat.codes
# Prepare data

def prep_data(df):

    X = df[ ["Pclass", "Sex", "SibSp", "Parch", "fare_filled", "age_filled", "embarked_cat", "deck_cat"] ].copy()

    X["Sex"] = X["Sex"].cat.codes

    X["embarked_cat"] = X["embarked_cat"].cat.codes

    X["deck_cat"] = X["deck_cat"].cat.codes

    return X



X_train = prep_data(train_df)

y_train = train_df["Survived"]

X_test = prep_data(test_df)

X_train.shape, y_train.shape, X_test.shape

#X_test.isna().sum()
# Run model

#from sklearn.linear_model import LogisticRegression

#clf = LogisticRegression(max_iter=500)

#clf.fit(X=X_train, y=y_train)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

clf.fit(X=X_train, y=y_train)



y_pred = clf.predict(X_test)

y_pred[:5]
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)