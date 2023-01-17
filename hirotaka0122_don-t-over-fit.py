# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()
test.info()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train.isnull().any().any() # is there any NULL cell ?
train["target"].value_counts().plot("bar")

plt.grid(True)

train["target"].value_counts()
train.head()
plt.figure(figsize=(25, 25))

for i, col in enumerate(list(train.columns)[2:30]):

    plt.subplot(7, 4, i+1)

    sns.distplot(train[col], hist=True, kde=True, rug=False, bins=10)
plt.figure(figsize=(25, 25))

for i, col in enumerate(list(train.columns)[2:30]):

    plt.subplot(7, 4, i+1)

    plt.boxplot(train[col])

    plt.title(col)
corr = train.corr().abs().unstack().sort_values().reset_index()

corr = corr[corr["level_0"] != corr["level_1"]]

corr.tail(10)
from sklearn.model_selection import StratifiedKFold, train_test_split



x_train_ori = train.drop(["id", "target"], axis=1)

y_train_ori = train["target"]

x_test = test.drop(["id"], axis=1).values

kfolds = StratifiedKFold(n_splits=50, shuffle=True, random_state=1)



x_train, x_val, y_train, y_val = train_test_split(x_train_ori.values, y_train_ori.values)
noisy_x_train = x_train_ori + np.random.uniform(0, 0.01, x_train_ori.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import boruta



c = 0

model = RandomForestClassifier()

for train_idx, test_idx in kfolds.split(x_train, y_train):

    c += 1

    model.fit(x_train[train_idx], y_train[train_idx])

    print(c, "Val Accuracy: {:.3f}".format(accuracy_score(y_train[test_idx], model.predict(x_train[test_idx]))))

feat_selector = boruta.BorutaPy(model, n_estimators="auto", verbose=2, alpha=0.05, max_iter=50, random_state=0)

feat_selector.fit(x_train, y_train)
# 選ばれた特徴量のみの配列を作成

selected_x_train = feat_selector.transform(x_train)

selected_x_val = feat_selector.transform(x_val)
import optuna





def objective(trial):

    max_depth = trial.suggest_int("max_depth", 2, 5)

    n_estimators = trial.suggest_int("n_estimators", 50, 100)

    max_features = trial.suggest_categorical("max_features", ["sqrt", "auto", "log2"])

    

    rfc = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)

    rfc.fit(selected_x_train, y_train) 

    return -1 * rfc.score(selected_x_val, y_val)





study = optuna.create_study()

study.optimize(objective, n_trials=30, n_jobs=-1)
print(study.best_params)

print("-"*50)

print(study.best_value)

print("-"*50)

print(study.best_trial)



print("-"*50)

for i in study.trials:

    print("param: {}, eval_value: {}".format(i[5], i[2]))
from sklearn.linear_model import LogisticRegression



def objective_lr(trial):

    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    tol = trial.suggest_loguniform("tol", 1e-4, 1e2)

    

    model = LogisticRegression(penalty=penalty, solver="liblinear", tol=tol)

    model.fit(selected_x_train, y_train)

    

    return - accuracy_score(y_true=y_val, y_pred=model.predict(selected_x_val))

    
lr_study = optuna.create_study()

lr_study.optimize(objective_lr, n_trials=50, n_jobs=-1)
from sklearn.svm import SVC



def objective_svm(trial):

    c = trial.suggest_uniform("c", 0.01, 5.0)

    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])

    gamma = trial.suggest_uniform("gamma", 0.01, 10.0)

    

    model = SVC(C=c, kernel=kernel, gamma=gamma)

    model.fit(selected_x_train, y_train)

    

    return - accuracy_score(y_true=y_val, y_pred=model.predict(selected_x_val))

    
svm_study = optuna.create_study()

svm_study.optimize(objective_svm, n_trials=50, n_jobs=-1)
from sklearn.ensemble import VotingClassifier



clsfs = [

    ("rfc", RandomForestClassifier(max_depth=study.best_params['max_depth'], 

                                 n_estimators=study.best_params['n_estimators'],

                                 max_features=study.best_params['max_features'])

    ),

    ("lr", LogisticRegression(penalty=lr_study.best_params["penalty"],

                              solver="liblinear", tol=lr_study.best_params["tol"])

    ),

    ("svm", SVC(C=svm_study.best_params["c"], kernel=svm_study.best_params["kernel"], gamma=svm_study.best_params["gamma"]))

]





vote = VotingClassifier(clsfs, voting="hard")

vote.fit(selected_x_train, y_train)
model = RandomForestClassifier(max_depth=study.best_params["max_depth"],

                               n_estimators=study.best_params["n_estimators"], 

                               max_features=study.best_params["max_features"])



model.fit(selected_x_train, y_train)

selected_x_test = feat_selector.transform(x_test)
lr = LogisticRegression(penalty=lr_study.best_params["penalty"], solver="liblinear", tol=lr_study.best_params["tol"])

lr.fit(selected_x_train, y_train)

selected_x_test = feat_selector.transform(x_test)
selected_x_test = feat_selector.transform(x_test)

submit = pd.read_csv("../input/sample_submission.csv")

submit["target"] = vote.predict(selected_x_test).astype("int")
submit = pd.read_csv("../input/sample_submission.csv")

submit["target"] = lr.predict(selected_x_test).astype("int")
submit.to_csv("submission.csv", index=False)
submit.head()