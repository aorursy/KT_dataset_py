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
# load data

df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")
# drop some catagorical columns from both data sets

df_train.drop(columns = ["Name" ,"Ticket" , "Cabin"], inplace=True)

df_test.drop(columns = ["Name" ,"Ticket" , "Cabin"], inplace=True)
df_train.columns, df_test.columns
# missing values

df_train.Age.fillna(df_train.Age.dropna().median() , inplace= True)

df_test.Age.fillna(df_train.Age.dropna().median() , inplace = True)
df_train.loc[df_train["Sex"] == "male", "Sex"] = 0

df_train.loc[df_train["Sex"] == "female", "Sex"] = 1

df_test.loc[df_test["Sex"] == "male", "Sex"] = 0

df_test.loc[df_test["Sex"] == "female", "Sex"] = 1
df_train["Embarked"] = df_train["Embarked"].fillna("S")

df_train.loc[df_train["Embarked"] == "S", "Embarked"] = 0

df_train.loc[df_train["Embarked"] == "C", "Embarked"] = 1

df_train.loc[df_train["Embarked"] == "Q", "Embarked"] = 2

df_test["Embarked"] = df_test["Embarked"].fillna("S")

df_test.loc[df_test["Embarked"] == "S", "Embarked"] = 0

df_test.loc[df_test["Embarked"] == "C", "Embarked"] = 1

df_test.loc[df_test["Embarked"] == "Q", "Embarked"] = 2
df_test.Fare.fillna(df_train.Fare.dropna().median() , inplace = True)
# Feature Selection

y = df_train["Survived"].values

x = df_train.drop(columns = ["Survived"]).values
# data splitting

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=0)
# Classifier ML algorithm

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state = 1)

fitting = classifier.fit(x,y)

preds_val = classifier.predict(df_test)
# export CSV file

test_out = pd.DataFrame({

    'PassengerId': df_test.PassengerId, 

    'Survived': preds_val

})

test_out.to_csv('submission.csv', index=False)