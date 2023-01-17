import csv
import numpy as np
import pandas as pd
import sklearn.tree
import sklearn.model_selection
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
def preprocessing(df):
    tmp = df
    tmp["Sex"] = tmp["Sex"].replace("male", 0).replace("female", 1)
    tmp["Embarked"] = tmp["Embarked"].replace("C", 0).replace("Q", 1).replace("S", 2)
    tmp["Age"].fillna(tmp["Age"].median(), inplace=True)
    tmp["Fare"].fillna(tmp["Fare"].median(), inplace=True)
    tmp["Embarked"].fillna(tmp["Embarked"].median(), inplace=True)
    tmp["FamilySize"] = 1 + tmp["SibSp"] + tmp["Parch"]
    tmp["CabinHead"] = tmp["Cabin"].str[0]
    tmp["CabinHead"] = tmp["CabinHead"].replace("A", 1).replace("B", 2).replace("C", 3).replace("D", 4).replace("E", 5).replace("F", 6).replace("G", 7).replace("T", 8).replace("U", 9)
    tmp["CabinHead"].fillna(0, inplace=True)
    
    names = ["Mr.", "Miss.", "Mrs.", "William", "John", "Master.", "Henry", "James", "Charles", "George", "Thomas", "Mary", "Edward", "Anna", "Joseph", "Frederick", "Elizabeth", "Johan", "Samuel", "Richard", "Arthur", "Margaret", "Alfred", "Maria", "Jr", "Alexander"]
    names_c = [name.replace(".", "") for name in names]
    for name, name_c in zip(names, names_c):
        tmp[name_c] = tmp["Name"].str.contains(name).astype(int)
    
    return tmp, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "CabinHead"] + names_c
df_train_pp, columns = preprocessing(df_train)
train_data = df_train_pp.loc[:, columns]
train_target = df_train_pp.loc[:, "Survived"]

df_test_pp, columns = preprocessing(df_test)
test_data = df_test_pp.loc[:, columns]
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
