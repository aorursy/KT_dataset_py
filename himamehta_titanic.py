# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

from sklearn import tree

import statsmodels.api as sm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
Train_data = pd.read_csv("../input/train.csv")

print(Train_data.head(1))

print (Train_data.describe())

print(Train_data["Embarked"].unique())
Train_data["Age"] = Train_data["Age"].fillna(Train_data["Age"].median())

Train_data["Sex"][Train_data["Sex"]=="male"] = 0

Train_data["Sex"][Train_data["Sex"] == "female"] = 1

Train_data["Embarked"] = Train_data["Embarked"].fillna('S')

Train_data["Embarked"][Train_data["Embarked"] == 'S'] = 0

Train_data["Embarked"][Train_data["Embarked"] == 'C'] = 1

Train_data["Embarked"][Train_data["Embarked"] == 'Q'] = 2
#Target
my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features,Target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features,Target))
Test_data = pd.read_csv("../input/test.csv")

Test_data.describe()
Test_data["Age"] = Test_data["Age"].fillna(Train_data["Age"].median())

Test_data["Fare"] = Test_data["Fare"].fillna(Train_data["Fare"].median())

Test_data["Embarked"][Test_data["Embarked"] == 'S'] = 0

Test_data["Embarked"][Test_data["Embarked"] == 'C'] = 1

Test_data["Embarked"][Test_data["Embarked"] == 'Q'] = 2

Test_data["Sex"][Test_data["Sex"] == "male"] = 0

Test_data["Sex"][Test_data["Sex"] == "female"] = 1

test_features = Test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values

Test_data["Sex"].unique()
predictions = my_tree_one.predict(test_features)

PassengerId =np.array(Test_data["PassengerId"]).astype(int)

Sol = pd.DataFrame(predictions,PassengerId,columns = ["Survived"])

print(Sol)

Sol.to_csv("Titanic_Solution.csv", index_label = ["PassengerId"])
Test_data = pd.read_csv("../input/test.csv")

Test_data["Age"] = Test_data["Age"].fillna(Train_data["Age"].median())

Test_data["Fare"] = Test_data["Fare"].fillna(Train_data["Fare"].median())

Test_data["Embarked"][Test_data["Embarked"] == 'S'] = 0

Test_data["Embarked"][Test_data["Embarked"] == 'C'] = 1

Test_data["Embarked"][Test_data["Embarked"] == 'Q'] = 2

Test_data["Sex"][Test_data["Sex"] == "male"] = 0

Test_data["Sex"][Test_data["Sex"] == "female"] = 1

test_features = Test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values

Test_data["Sex"].unique()

predictions = my_tree_one.predict(test_features)

PassengerId =np.array(Test_data["PassengerId"]).astype(int)

Sol = pd.DataFrame(predictions,PassengerId,columns = ["Survived"])

print(Sol)

Sol.to_csv("Titanic_Solution.csv", index_label = ["PassengerId"])
my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features,Target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features,Target))
features = Train_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values

Target = Train_data["Survived"].values