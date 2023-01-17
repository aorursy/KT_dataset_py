import csv
import numpy as np
import pandas as pd
import sklearn.tree
import sklearn.model_selection
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
# Name,Ticket,Cabin

df_train_tmp = df_train.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
df_train_tmp = df_train_tmp.replace("male", 0).replace("female", 1)
df_train_tmp = df_train_tmp.replace("C", 0).replace("Q", 1).replace("S", 2)
df_train_tmp["Age"].fillna(df_train_tmp["Age"].median(), inplace=True)
df_train_tmp["Fare"].fillna(df_train_tmp["Fare"].median(), inplace=True)
df_train_tmp["Embarked"].fillna(df_train_tmp["Embarked"].median(), inplace=True)

train_data = df_train_tmp.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
train_target = df_train_tmp.loc[:, "Survived"]

df_test_tmp = df_test.replace("male", 0).replace("female", 1)
df_test_tmp = df_test_tmp.replace("C", 0).replace("Q", 1).replace("S", 2)
df_test_tmp["Age"].fillna(df_test_tmp["Age"].median(), inplace=True)
df_test_tmp["Fare"].fillna(df_test_tmp["Fare"].median(), inplace=True)
df_test_tmp["Embarked"].fillna(df_test_tmp["Embarked"].median(), inplace=True)
test_data = df_test_tmp.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
param_grid = {
    "criterion": ["entropy", "gini"],
    "splitter": ["random", "best"],
    "max_depth": range(1, 20),
    "min_samples_leaf": range(1, 20),
}

grid_search = sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(train_data, train_target)
print('Best parameters: {}'.format(grid_search.best_params_))
print('Best cross-validation: {}'.format(grid_search.best_score_))
predicted = grid_search.predict(test_data)
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(df_test["PassengerId"], predicted):
        writer.writerow([pid, survived])
