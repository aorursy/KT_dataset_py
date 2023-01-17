import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
test_shape = test.shape
train_shape = train.shape

print(test_shape)
print(train_shape)
test.describe()
train.describe()
train.info()
def void_table(df):
    null_val = df.isnull().sum()
    per = 100 * null_val / len(df)
    void_table = pd.concat(
        [null_val, per]
        , axis = 1
    )
    void_table_ren_columns = void_table.rename(
        columns = {0: '欠損数', 1: '%'}
    )
    return void_table_ren_columns

void_table(train)
void_table(test)
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

void_table(train)
test["Age"] = test["Age"].fillna(test["Age"].median())
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()

test.head(10)
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

train.head(10)
from sklearn import tree
target = train["Survived"].values
features_one = train[
    ["Pclass", "Sex", "Age", "Fare"]
].values

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

test_features = test[
    ["Pclass", "Sex", "Age", "Fare"]
].values

my_prediction = my_tree_one.predict(test_features)
my_prediction.shape
print(my_prediction)
Passenger_id = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, Passenger_id, columns = ["Survived"])

my_solution.to_csv("sample_tree_one.csv", index_label = ["PassengerId"])