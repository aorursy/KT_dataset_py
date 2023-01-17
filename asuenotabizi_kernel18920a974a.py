# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

sns.set()

train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

submit_df = pd.read_csv("../input/titanic/gender_submission.csv")
train_df.describe()
test_df.describe()
test_shape = test_df.shape

train_shape = train_df.shape

 

print(test_shape)

print(train_shape)
def kesson_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum() / len(df)

        kesson_table = pd.concat([null_val, percent], axis=1)

        kesson_table_ren_columns = kesson_table.rename(columns = {0 : '欠損数', 1 : '%'})

        return kesson_table_ren_columns

 

kesson_table(train_df)
kesson_table(test_df)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())

train_df["Embarked"] = train_df["Embarked"].fillna("S")

 

kesson_table(train_df)
train_df["Sex"][train_df["Sex"] == "male"] = 0

train_df["Sex"][train_df["Sex"] == "female"] = 1

train_df["Embarked"][train_df["Embarked"] == "S" ] = 0

train_df["Embarked"][train_df["Embarked"] == "C" ] = 1

train_df["Embarked"][train_df["Embarked"] == "Q"] = 2

 

train_df.head(10)
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())

test_df["Sex"][test_df["Sex"] == "male"] = 0

test_df["Sex"][test_df["Sex"] == "female"] = 1

test_df["Embarked"][test_df["Embarked"] == "S"] = 0

test_df["Embarked"][test_df["Embarked"] == "C"] = 1

test_df["Embarked"][test_df["Embarked"] == "Q"] = 2

test_df.Fare[152] = test_df.Fare.median()

 

test_df.head(10)
from sklearn import tree



target = train_df["Survived"].values

features_one = train_df[["Pclass", "Sex", "Age", "Fare"]].values

 

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

 

test_features = test_df[["Pclass", "Sex", "Age", "Fare"]].values

 

my_prediction = my_tree_one.predict(test_features)



my_prediction.shape
print(my_prediction)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": my_prediction

    })



submission.to_csv('submission.csv', index=False)