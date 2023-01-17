import csv
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.linear_model
import xgboost as xgb
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
    
    #names = ["Mr.", "Miss.", "Mrs.", "William", "John", "Master.", "Henry", "James", "Charles", "George", "Thomas", "Mary", "Edward", "Anna", "Joseph", "Frederick", "Elizabeth", "Johan", "Samuel", "Richard", "Arthur", "Margaret", "Alfred", "Maria", "Jr", "Alexander"]
    #names_c = [name.replace(".", "") for name in names]
    #for name, name_c in zip(names, names_c):
    #    tmp[name_c] = tmp["Name"].str.contains(name).astype(int)
    
    return tmp, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "CabinHead"]# + names_c
df_train_pp, columns = preprocessing(df_train)
train_data = df_train_pp.loc[:, columns]
train_target = df_train_pp.loc[:, "Survived"]

df_test_pp, columns = preprocessing(df_test)
test_data = df_test_pp.loc[:, columns]
xg = xgb.XGBClassifier()
rf = sklearn.ensemble.RandomForestClassifier()
lr = sklearn.linear_model.LogisticRegression()

xg_param = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 6, 9],
    "colsample_bytree": [0.5, 0.9, 1.0],
    "learning_rate": [0.5, 0.9, 1.0]
}
rf_param = {
    "n_estimators": [5, 10, 50, 100, 300],
    "max_features": range(1, 9),
    "min_samples_split": [3, 5, 10, 20],
    "max_depth": [3, 5, 10, 20]
}
lr_param = {
    "C": list(np.logspace(0, 4, 10))
}

params = {}
params.update({"xg__" + k: v for k, v in xg_param.items()})
params.update({"rf__" + k: v for k, v in rf_param.items()})
params.update({"lr__" + k: v for k, v in lr_param.items()})
eclf = sklearn.ensemble.VotingClassifier(estimators=[("xg", xg), ("rf", rf), ("lr", lr)], voting="soft")
clf = sklearn.model_selection.RandomizedSearchCV(eclf, param_distributions=params, cv=5, n_iter=1000, n_jobs=1,verbose=2)
clf.fit(train_data, train_target)
predicted = clf.predict(test_data)
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(df_test["PassengerId"], predicted):
        writer.writerow([pid, survived])
