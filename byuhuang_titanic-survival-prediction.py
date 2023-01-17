import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
import matplotlib.pyplot as plt

sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot
sex_pivot.plot.bar()
plt.show()
pclass_pivot = train.pivot_table(index="Pclass",values="Survived")
pclass_pivot
pclass_pivot.plot.bar()
plt.show()
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha = 0.5, color = 'red', bins = 50)
died["Age"].plot.hist(alpha = 0.5, color='green', bins = 50)
plt.legend(['Survived','Died'])
plt.show()
embark_pivot = train.pivot_table(index="Embarked",values="Survived")
embark_pivot
embark_pivot.plot.bar()
plt.show()
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Fare"].plot.hist(alpha = 0.5, color = 'red', bins = 20)
died["Fare"].plot.hist(alpha = 0.5, color='green', bins = 20)
plt.legend(['Survived','Died'])
plt.show()
# Create a variable called Family to represent if the passenger had any family member aboard or not,
# and see whether having any family member will increase chances of survival

train['Family'] =  train["Parch"] + train["SibSp"]

train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]

test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

family_pivot = train.pivot_table(index="Family",values="Survived")
family_pivot
family_pivot.plot.bar()
plt.show()
import numpy as np

# replace missing value
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
# test.Fare[152] = test.Fare.median()
# convert features in train
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
X_train = train[['Sex', 'Pclass', 'Age', 'Embarked', 'Fare', 'Family']].values
X_test = test[['Sex', 'Pclass', 'Age', 'Embarked', 'Fare', 'Family']].values

Y_train = train["Survived"].values
from sklearn import tree

# fit the decision tree
class_tree = tree.DecisionTreeClassifier(max_leaf_nodes = 10, min_samples_leaf = 2, max_depth= 5, min_samples_split = 4, min_impurity_decrease = 0.01)
class_tree = class_tree.fit(X_train, Y_train)

# observe the importance and score of the features
print(class_tree.feature_importances_)
print(class_tree.score(X_train, Y_train))
# make prediction on test set
Y_test = class_tree.predict(X_test)
# create DataFrame
PassengerId = np.array(test["PassengerId"]).astype(int)
result = pd.DataFrame(Y_test, PassengerId, columns = ["Survived"])
print(result)
# write results to a csv file
result.to_csv("Beverly_submission.csv", index_label = ["PassengerId"])
import graphviz 
tree_data = tree.export_graphviz(class_tree, out_file = None) 
graph = graphviz.Source(tree_data) 
graph