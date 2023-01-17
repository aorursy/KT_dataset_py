import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv("../input/titanic/train.csv")

holdout = pd.read_csv("../input/titanic/test.csv")



train["origin"] = "train"

holdout["origin"] = "holdout"



cols_without_target = list(train.columns)

cols_without_target.remove("Survived")



df = pd.concat([train[cols_without_target], holdout])



df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Fare"] = df["Fare"].fillna(df["Fare"].mean())



df["Cabin"] = df["Cabin"].str.extract(r"^.*([A-Z])[0-9]+$")[0]



df = df.drop("Cabin", axis=1)



df["Embarked"] = df["Embarked"].fillna('S')



df["PassengerId"] = df["PassengerId"].astype("uint16")



df["Pclass"] = df["Pclass"].astype("category")



df = df.drop("Name", axis=1)



df["Sex"] = df["Sex"].astype("category")



df["Age"] = df["Age"].astype("uint8")



df["SibSp"] = df["SibSp"].astype("uint8")

df["Parch"] = df["Parch"].astype("uint8")



df["Ticket"] = df["Ticket"].str.extract(r"([0-9]+)$").astype('float')



df = df.dropna()



df["Fare"] = df["Fare"].astype("uint16")



df["Embarked"] = df["Embarked"].astype("category")



pclass_dummies = pd.get_dummies(df["Pclass"], prefix="Pclass", drop_first=True)



df = pd.concat([df, pclass_dummies], axis=1)



sex_dummies = pd.get_dummies(df["Sex"], prefix="Sex", drop_first=True)



df = pd.concat([df, sex_dummies], axis=1)



embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)



df = pd.concat([df, embarked_dummies], axis=1)



train = pd.concat([df.loc[df["origin"] == "train"], train["Survived"]], axis=1)

holdout = df.loc[df["origin"] == "holdout"]



train = train.dropna()



train = train.drop("origin", axis=1)

holdout = holdout.drop("origin", axis=1)



train["Age_Group"] = pd.cut(train["Age"], [0,1,5,13,18,35,65,100], 

       labels=["infant", "toddler", "child", "teenager", "young adult", "middle aged", "seniors"]).astype("category")

holdout["Age_Group"] = pd.cut(holdout["Age"], [0,1,5,13,18,35,65,100], 

       labels=["infant", "toddler", "child", "teenager", "young adult", "middle aged", "seniors"]).astype("category")



ageGroup_dummies = pd.get_dummies(train["Age_Group"], prefix="Age_Group", drop_first=True)

train = pd.concat([train, ageGroup_dummies], axis=1)



ageGroup_dummies = pd.get_dummies(holdout["Age_Group"], prefix="Age_Group", drop_first=True)

holdout = pd.concat([holdout, ageGroup_dummies], axis=1)



train = train.drop(["Pclass", "Sex", "Age",  "Embarked", "Age_Group"], axis=1)

holdout = holdout.drop(["Pclass", "Sex", "Age", "Embarked", "Age_Group"], axis=1)
feature_cols = ["SibSp", "Parch", "Ticket", "Fare", "Pclass_2", "Pclass_3", 

                "Sex_male", "Embarked_Q", "Embarked_S", "Age_Group_toddler", "Age_Group_child", "Age_Group_teenager",

               "Age_Group_young adult", "Age_Group_middle aged", "Age_Group_seniors"]

target_col = "Survived"



from sklearn.preprocessing import MinMaxScaler



mm_scaler = MinMaxScaler()

train[feature_cols] = mm_scaler.fit_transform(train[feature_cols])

holdout[feature_cols] = mm_scaler.fit_transform(holdout[feature_cols])



X = train[feature_cols]

y = train[target_col]
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

selector = RFECV(rf, step=1, cv=10, n_jobs=-1)

selector = selector.fit(X, y)

features_mask = selector.support_
best_features = [col for col, select in zip(feature_cols, features_mask) if select]

best_features
rf = RandomForestClassifier(random_state=42)

scores = cross_val_score(rf, X[best_features], y, cv=10, n_jobs=-1)

print(np.mean(scores))
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier(n_jobs=-1, random_state=42)



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Wighting of classification classes

class_weight=[None, "balanced", "balanced_subsample"]





params = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap,

                'class_weight': class_weight}
clf = RandomizedSearchCV(rf, params, n_iter=10, cv=3, n_jobs=-1, verbose=5)

clf.fit(X[best_features], y)

best_model = clf.best_estimator_

best_score = clf.best_score_
(best_model, best_score)
best_model.fit(X_train[best_features], y_train)

y_pred = best_model.predict(X_train[best_features])

print(f"training accuracy: {roc_auc_score(y_train, y_pred)}")
best_model.fit(X_train[best_features], y_train)

y_pred = best_model.predict(X_test[best_features])

print(f"testing accuracy: {roc_auc_score(y_test, y_pred)}")
scores = cross_val_score(best_model, X[best_features], y, cv=10, n_jobs=-1)

print(f"cross_val_score: {np.mean(scores)}")
clf.best_params_
!pwd
best_model.fit(X[best_features], y)

y_pred = best_model.predict(holdout[best_features])

submission = pd.DataFrame({"PassengerId": holdout["PassengerId"], "Survived": y_pred})

submission.to_csv("submission.csv", index=False)