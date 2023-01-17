import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv")
print("Shape", df.shape)
print("Columns", df.columns)
print(df.info())
print(df.describe())
v_arr = df.columns.ravel()[1:29]
desc = df[v_arr].describe().T
kurt = pd.Series([df[i].kurt() for i in v_arr], name="Kurt", index=v_arr)
skew = pd.Series([df[i].skew() for i in v_arr], name="Skew", index=v_arr)
desc = pd.concat([desc, kurt, skew], axis=1)
desc.drop(["count"], axis=1, inplace=True)

desc
desc
df_kurt = desc.sort_values("Kurt")
df_skew = desc.sort_values("Skew")
def plot_dist(df, arr):
    mask0 = df["Class"] == 0
    mask1 = df["Class"] == 1

    for i in arr:
        plt.figure(figsize=(14,6))
        plt.title(i)
        sns.distplot(df[i][mask0])
        sns.distplot(df[i][mask1])
        plt.legend(["Not Fraud", "Fraud"])
        plt.grid()
plot_dist(df, df_kurt.index.ravel()[:5])
plot_dist(df, df_skew.index.ravel()[:5])
df.drop("Time", axis=1, inplace=True)
X = df.drop("Class", axis=1)
y = df["Class"]

print("Feature shape", X.shape, "Label shape", y.shape)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
def manual_cross_validation(estimator, X, y, cv=StratifiedKFold(n_splits = 5)):
    arr = []
    for train_index, test_index in cv.split(X, y):
        estimator.fit(X.loc[train_index], y.loc[train_index])
        estimator_probs = estimator.predict_proba(X.loc[test_index])
        arr.append(roc_auc_score(y.loc[test_index], estimator_probs[:,1]))
    fpr, tpr, _ = roc_curve(y.loc[test_index], estimator_probs[:,1])
    plt.figure(figsize=(8,8))
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr)
    return arr, np.mean(arr), np.std(arr)

estimators = []
estimators.append(LogisticRegression(random_state=77))
estimators.append(DecisionTreeClassifier(random_state=77))
estimators.append(LinearDiscriminantAnalysis())
estimators.append(GradientBoostingClassifier(random_state=77))

for estimator in estimators:
    _, mean, std = manual_cross_validation(estimator, X, y)
    print("Mean", mean, "STD", std)
%%time
LR_params = {
    "penalty": ["l2"],
    "tol": [1e-4],
    "C": [1e-2],
    "solver": ["liblinear", "newton-cg", "sag"],
    "max_iter": [500, 1e3],
    "random_state": [77]
}

LR_GS = GridSearchCV(LogisticRegression(), 
                     param_grid=LR_params, 
                     cv=StratifiedKFold(n_splits = 5), 
                     scoring="roc_auc", 
                     verbose=1, n_jobs=4)
LR_GS.fit(X, y)
LR_best = LR_GS.best_estimator_
print(LR_GS.best_params_)
GBC_params = {
    "learning_rate": [1e-3],
    "n_estimators": [500, 800],
    "max_depth": [5],
    "subsample": [0.8],
    "max_features": [None],
    "init": [None],
    "random_state": [77]
}

GBC_GS = GridSearchCV(GradientBoostingClassifier(), 
                      param_grid=GBC_params, 
                      cv=StratifiedKFold(n_splits = 5), 
                      scoring="roc_auc", 
                      verbose=1, n_jobs=4)
GBC_GS.fit(X, y)
GBC_best = GBC_GS.best_estimator_
print(GBC_GS.best_params_)
def plot_learning_result(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # Original code from sklearn page Plotting Learning Curve with slight modification
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

from sklearn.model_selection import learning_curve

plot_learning_result(LR_GS.best_estimator_, "Logistic Reg. Learning Curve", X, y, cv=StratifiedKFold(n_splits = 5), n_jobs=4)

plot_learning_result(LinearDiscriminantAnalysis(), "Linear Discrimanant Analysis Curve", X, y, cv=StratifiedKFold(n_splits = 5), n_jobs=4)

plot_learning_result(GBC_GS.best_estimator_, "Gradient Boosting Learning Curve", X, y, cv=StratifiedKFold(n_splits = 5), n_jobs=4)
