import csv
import numpy as np
import pandas as pd
import sklearn.tree
import sklearn.model_selection
def tree(depth, train_data, train_target, valid_data, valid_target):
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(train_data, train_target)
    valid_predicted = clf.predict(valid_data)
    return clf, sum(valid_predicted == valid_target.values) / len(valid_target.values)
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
# Name,Ticket,Cabin

df_train_tmp = df_train.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
df_train_tmp = df_train_tmp.replace("male", 0).replace("female", 1)
df_train_tmp = df_train_tmp.replace("C", 0).replace("Q", 1).replace("S", 2)
df_train_tmp["Age"].fillna(df_train_tmp["Age"].median(), inplace=True)
df_train_tmp["Fare"].fillna(df_train_tmp["Fare"].median(), inplace=True)
df_train_tmp["Embarked"].fillna(df_train_tmp["Embarked"].median(), inplace=True)

train, valid = sklearn.model_selection.train_test_split(df_train_tmp, test_size=0.3)
train_data = train.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
train_target = train.loc[:, "Survived"]
valid_data = valid.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
valid_target = valid.loc[:, "Survived"]

df_test_tmp = df_test.replace("male", 0).replace("female", 1)
df_test_tmp = df_test_tmp.replace("C", 0).replace("Q", 1).replace("S", 2)
df_test_tmp["Age"].fillna(df_test_tmp["Age"].median(), inplace=True)
df_test_tmp["Fare"].fillna(df_test_tmp["Fare"].median(), inplace=True)
df_test_tmp["Embarked"].fillna(df_test_tmp["Embarked"].median(), inplace=True)
test_data = df_test_tmp.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
max_acc = 0
max_depth = 0
max_model = None
for i in range(1, 100):
    ret = tree(i, train_data, train_target, valid_data, valid_target)
    if max_acc < ret[1]:
        max_acc = ret[1]
        max_depth = i
        max_model = ret[0]
print(max_depth, ",", max_acc)
predicted = max_model.predict(test_data)
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(df_test["PassengerId"], predicted):
        writer.writerow([pid, survived])
