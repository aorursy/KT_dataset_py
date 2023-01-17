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
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.head()
test.head()
train.shape
test.shape
train.describe()
test.describe()
def kesson_table(df):

    null_val = df.isnull().sum()

    percent = 100 * df.isnull().sum()/len(df)

    kesson_table = pd.concat([null_val, percent], axis=1)

    kesson_table_ren_columns = kesson_table.rename(columns = {0 : '欠損数', 1 : '%'})

    return kesson_table_ren_columns
kesson_table(train)
kesson_table(test)
train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Embarked"] = train["Embarked"].fillna("S")
kesson_table(train)
train = train.replace({'male': 0, 'female': 1,'S': 0, 'C': 1, 'Q': 2})
train.head()
test["Age"] = test["Age"].fillna(test["Age"].mean())
test = test.replace({'male': 0, 'female': 1,'S': 0, 'C': 1, 'Q': 2})
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
test.head()
from sklearn import tree
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
my_prediction = my_tree_one.predict(test_features)
my_prediction.shape
my_prediction
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])