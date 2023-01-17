import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.info()
df.describe().transpose()
plt.figure(figsize=(10,6))

sns.countplot(x="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="sex", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.distplot(df["age"])
plt.figure(figsize=(24,8))

sns.countplot(x="age", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="cp", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="fbs", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="restecg", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="exang", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="slope", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="ca", hue="target", data=df)
plt.figure(figsize=(10,6))

sns.countplot(x="thal", hue="target", data=df)
f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix', fontsize=25)



sns.heatmap(df.corr(), linewidths=0.25, vmax=0.7, square=True, cmap="BuGn",

            linecolor='w', annot=True, annot_kws={"size":8}, cbar_kws={"shrink": .9});
cp_dummies = pd.get_dummies(df["cp"], drop_first=True, prefix="cp")

restecg_dummies = pd.get_dummies(df["restecg"], drop_first=True, prefix="restecg")

slope_dummies = pd.get_dummies(df["slope"], drop_first=True, prefix="slope")

ca_dummies = pd.get_dummies(df["ca"], drop_first=True, prefix="ca")

thal_dummies = pd.get_dummies(df["thal"], drop_first=True, prefix="thal")

                            
concat_df = [df, cp_dummies, restecg_dummies, slope_dummies, ca_dummies, thal_dummies]

df = pd.concat(concat_df, axis=1)

df = df.drop(["cp", "restecg", "slope", "ca", "thal"], axis=1)
df.head()
X = df.drop("target", axis=1).values

y = df["target"].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
X_train.shape
X_test.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
classifiers = [

    ("Logistic Regression", LogisticRegression(random_state=0)),

    ("KNN", KNeighborsClassifier()),

    ("Support Vector Machine", SVC(kernel = 'rbf',random_state=0)),

    ("Naive Bayes", GaussianNB()),

    ("Random Forest", RandomForestClassifier(random_state=0)),

    ("Ada Boost", AdaBoostClassifier(random_state=0)),

    ("Gradient Boosting", GradientBoostingClassifier(random_state=0)),

    ("XGBoost", XGBClassifier(random_state=0)),

    ("LDA", LinearDiscriminantAnalysis())

    

]
from sklearn.model_selection import cross_val_score
columns = ["Classifier", "Validation Score", "+/-"]

model_comparison = pd.DataFrame(columns=columns)

row_index = 0



for name, clf in classifiers:

    scores = cross_val_score(clf, X_train, y_train, cv=5)

    model_comparison.loc[row_index, "Classifier"] = name

    model_comparison.loc[row_index, "Validation Score"] = round(scores.mean(), 4)

    model_comparison.loc[row_index, "+/-"] = round(scores.std()*2, 4)

    row_index += 1

    

model_comparison.sort_values(by=["Validation Score"], ascending=False, inplace=True)

model_comparison.reset_index(drop=True, inplace=True)

model_comparison
classifiers = [

    ("Logistic Regression", LogisticRegression(random_state=0)),

    ("KNN", KNeighborsClassifier()),

    ("Support Vector Machine", SVC(random_state=0)),

    ("Naive Bayes", GaussianNB()),

    ("Random Forest", RandomForestClassifier(random_state=0)),

    ("Ada Boost", AdaBoostClassifier(random_state=0)),

    ("Gradient Boosting", GradientBoostingClassifier(random_state=0)),

    ("XGBoost", XGBClassifier(random_state=0)),

    ("LDA", LinearDiscriminantAnalysis())

    

]
parameters = [

    [{

        #LR

        "solver": ["lbfgs", "liblinear"],

        "penalty": ["l2"],

    }],

    [{

        #KNN

        "n_neighbors": [3 , 4, 5, 6,  7, 8, 9, 10, 12, 15],

        "weights": ["uniform", "distance"],

        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]

    }],

    [{

        #SVM

        "kernel": ["linear", "rbf"],

        "gamma": ["scale", "auto"],

        "C": [1,2,3],

        "decision_function_shape": ["ovo", "ovr"]

    }],

    [{

        #NB

    }],

    [{

        #RF

        #"bootstrap": [True, False],  

        #"min_samples_leaf": [1, 2, 4, 6],

        #"min_samples_split": [2, 4, 6, 10],

        #"n_estimators": [50, 100, 200, 300],

        #"criterion": ["gini", "entropy"],

        #"max_depth": [20, 30, 40, None],

        #"max_features": ["auto", "log2"]

        "bootstrap": [True],

        "min_samples_leaf": [1],

        "min_samples_split": [4],

        "n_estimators": [50],

        "criterion": ["entropy"],

        "max_depth": [20],

        "max_features": ["auto"]

    }],

    [{

        #ADA

        "n_estimators": [25, 50, 75, 100],

        "learning_rate": [0.25, 0.5, 1],

        

    }],

    [{

        #GB

        #"learning_rate": [0.05, 0.1, 0.2, 0.25, 0.3],

        #"n_estimators": [50, 100, 150, 200],

        #"criterion": ["friedman_mse", "mse", "mae"],

        #"max_depth": [2, 3, 4, 5]

        "learning_rate": [0.25],

        "n_estimators": [150],

        "criterion": ["friedman_mse"],

        "max_depth": [3]

        

        

    }],

    [{

        #XGB

        "learning_rate": [0.025, 0.05, 0.1, 0.2, 0.3, 0.4],

        "max_depth": [2,4,6,8,10]

    }],

    [{

        # LDA

        "solver": ["svd", "lsqr"]

    }]

]
from sklearn.model_selection import GridSearchCV

import time
columns = ["Classifier", "Grid Search Score"]

model_comparison = pd.DataFrame(columns=columns)

row_index = 0



for (name, clf), param in zip(classifiers, parameters):

    start = time.perf_counter()

    grid_search = GridSearchCV(estimator=clf, param_grid=param, cv=5)

    grid_search.fit(X_train, y_train)

    stop = time.perf_counter() - start

    best_params = grid_search.best_params_

    best_score = round(grid_search.best_score_, 4)

    model_comparison.loc[row_index, "Classifier"] = name

    model_comparison.loc[row_index, "Grid Search Score"] = best_score

    row_index += 1

    print("Best parameters for {} are: {}.".format(name, best_params))

    print("Best score of {} is: {}.".format(name, best_score))

    print("Run time of {} is {:.2f} second".format(name, stop))
model_comparison.sort_values(by=["Grid Search Score"], ascending=False, inplace=True)

model_comparison.reset_index(drop=True, inplace=True)

model_comparison
from sklearn.ensemble import VotingClassifier
voting_classifiers = [

    ("Logistic Regression", LogisticRegression(penalty="l2", solver="liblinear", random_state=0)),

    ("KNN", KNeighborsClassifier(algorithm="auto", n_neighbors=10, weights="distance")),

    ("Support Vector Machine", SVC(C=1, decision_function_shape="ovo", gamma="scale", kernel="linear", probability=True, random_state=0)),

    #("Naive Bayes", GaussianNB()),

    ("Random Forest", RandomForestClassifier(bootstrap=True, criterion="entropy", max_depth=20, max_features="auto",

                                             min_samples_leaf=1, min_samples_split=4, n_estimators=50, random_state=0)),

    ("Ada Boost", AdaBoostClassifier(learning_rate=1, n_estimators=50, random_state=0)),

    ("Gradient Boosting", GradientBoostingClassifier(criterion="friedman_mse", learning_rate=0.25, max_depth=3, 

                                                     n_estimators=150, random_state=0)),

    #("XGBoost", XGBClassifier(learning_rate=0.05, max_depth=2, random_state=0)),

    ("LDA", LinearDiscriminantAnalysis(solver="svd"))

    

]
hard_voting = VotingClassifier(estimators = voting_classifiers, voting="hard")

scores = cross_val_score(hard_voting, X_train, y_train, cv=5)

print("Hard voting score is {:.4f}".format(scores.mean()))
soft_voting = VotingClassifier(estimators = voting_classifiers, voting="soft")

scores = cross_val_score(soft_voting, X_train, y_train, cv=5)

print("Soft voting score is {:.4f}".format(scores.mean()))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
soft_voting = VotingClassifier(estimators = voting_classifiers, voting="soft")

soft_voting.fit(X_train, y_train)

y_pred = soft_voting.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Soft voting score on test data is {:.4f}".format(accuracy))
print(classification_report(y_test, y_pred, digits = 4 ))
cm = confusion_matrix(y_test, y_pred)

print(cm)