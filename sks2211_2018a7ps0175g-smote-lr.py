import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv").drop(

    ["id"], axis=1

)

#train_df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv").drop(

#    ["id"], axis=1

#).drop_duplicates()

test_df = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")


# Ensemble Models

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import StackingClassifier



# Linear

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression



# Neighbours

from sklearn.neighbors import KNeighborsClassifier



# SVM

from sklearn.svm import SVC



# Bayes

from sklearn.naive_bayes import GaussianNB



# Tree

from sklearn.tree import DecisionTreeClassifier



# Imbalanced Ensembles

from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier



# Utils

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.utils import resample

from sklearn.model_selection import (

    train_test_split,

    RepeatedStratifiedKFold,

    cross_val_score,

    GridSearchCV,

)

from imblearn.over_sampling import SMOTE
SEED = 22

# Stacking Classiifer - 0.6313

# LR Meta learner

'''Stack = [LogisticRegression, RandomForestRegressor, KNN, SVM]'''

# Random Forests - 0.6212

'''param_grid = {

    "n_estimators": [100, 200, 300, 400, 500, 600],

    "max_features": ["auto", "sqrt", "log2"],

    "max_depth": [4, 5, 6, 7, 8],

    "criterion": ["gini", "entropy"],

}'''

# Logistic Regression Over_Under Sampling- 0.6414

'''{

    "C": [10 ** X for X in range(1, 4)],

    "penalty": ["l2"],

    "solver": ["newton-cg", "liblinear"],

}'''

# Logistic Regression SMOTE - 0.6983

'''{

    "C": [10 ** X for X in range(1, 4)],

    "penalty": ["l2"],

    "solver": ["newton-cg", "liblinear"],

}'''
def get_sample(n0=1485, n1=1485, data=train_df, split=-1, random_state=22):

    data_1, data_0 = data[data['target']==1], data[data['target']==0]

    if type(n0)==float:

        data_0=data_0.sample(frac=n0, random_state=random_state)

    else:

        data_0=data_0.sample(n0, random_state=random_state)

    if n1==len(data[data['target']==1]):

        pass

    else:

        data_1 = resample(data_1, replace=True, n_samples=n1, random_state=random_state)

    data = data_0.append(data_1)

    if split <= 1 and split>0:

        data=data.values

        X, y = data[:,:-1], data[:,-1]

        return train_test_split(X, y, test_size=split, random_state=random_state)

    elif split == -1:

        data=data.sample(frac=1,random_state=random_state).values

        return data[:,:-1], data[:,-1]

    

# SMOTE

os = SMOTE(random_state=SEED)



# X_train, y_train = get_sample(n0=0.99, n1=100000)

X_train, y_train = os.fit_sample(train_df.values[:, :-1], train_df.values[:, -1])
# Logistic Regression SMOTE

os = SMOTE(random_state=SEED)

X_train, y_train = os.fit_sample(train_df.values[:, :-1], train_df.values[:, -1])

model = LogisticRegression(

    C=1000, penalty="l2", solver="newton-cg", max_iter=1000, random_state=SEED

)

model.fit(X_train, y_train)

preds = model.predict_proba(test_df.drop(["id"], axis=1).values)

sub = pd.DataFrame({"id": test_df["id"], "target": preds[:, 1]})

sub.to_csv("lr_prob_smote.csv", index=False)
'''

# X, y = get_sample(n0=0.99, n1=100000)



model = LogisticRegression(max_iter=1000)

cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=22)

stack_scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)

stack_scores



X, y = get_sample(n0=2000)

level0 = list()

level0.append(("lr", LogisticRegression(max_iter=1000, C=5)))

level0.append(("knn", KNeighborsClassifier(n_neighbors=100)))

level0.append(("svm", SVC()))

level0.append(("gnb", GaussianNB()))

level0.append(

    (

        "rf",

        RandomForestClassifier(

            criterion="gini",

            max_depth=6,

            max_features="auto",

            n_estimators=200,

            random_state=22,

        ),

    )

)

level0.append(("eec", EasyEnsembleClassifier()))

level1 = LogisticRegression(max_iter=1000)

model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)

cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=22)

stack_scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)



preds[:, 1]



X, y = get_sample(n0=2000)

model = LogisticRegression(max_iter=1000, C=5)

model = KNeighborsClassifier(n_neighbors=100)

model = SVC()

model = RandomForestClassifier(

    criterion="gini",

    max_depth=6,

    max_features="auto",

    n_estimators=200,

    random_state=22,

)

cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=22)

scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)

print(scores)

print(scores.mean())



for n0 in [1500, 2000, 2500, 3000]:

    print(n0)

    for n_est in [50, 100, 250, 500]:

        X, y = get_sample(n0=n0)

        model = EasyEnsembleClassifier(n_estimators=n_est)

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=22)

        scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)

        print(scores.mean())



X_train, X_test, y_train, y_test = tts(

    train_df.values[:, :-1], train_df.values[:, -1], test_size=0.2, random_state=22

)

rfc = BalancedRandomForestClassifier(n_estimators=200)

df_1, df_0 = train_df[train_df["target"] == 1], train_df[train_df["target"] == 0]

i = 1

data = df_1.append(df_0.sample(2401, random_state=i))

X_train, X_test, y_train, y_test = tts(

    data.values[:, :-1], data.values[:, -1], test_size=0.2, random_state=22

)

rfc = RandomForestClassifier(

    criterion="gini",

    max_depth=6,

    max_features="auto",

    n_estimators=200,

    random_state=22,

)

rfc.fit(X_train, y_train)

pred = rfc.predict(X_test)

print(metrics.roc_auc_score(y_true=y_test, y_score=pred))

df_1, df_0 = df[df["target"] == 1], df[df["target"] == 0]

for i in range(1, 11):

    data = df_1.append(df_0.sample(1501, random_state=i))

    X_train, X_test, y_train, y_test = tts(

        data.values[:, :-1], data.values[:, -1], test_size=0.2, random_state=22

    )

    rfc = RandomForestClassifier(

        criterion="gini",

        max_depth=6,

        max_features="auto",

        n_estimators=200,

        random_state=22,

    )

    rfc.fit(X_train, y_train)

    pred = rfc.predict(X_test)

    print(metrics.roc_auc_score(y_true=y_test, y_score=pred))

rfc = RandomForestClassifier(random_state=22)

rfc.fit(X_train, y_train)

pred = rfc.predict(X_test)

rfc = RandomForestClassifier(random_state=22)

param_grid = {

    "n_estimators": [100, 200, 300, 400, 500, 600],

    "max_features": ["auto", "sqrt", "log2"],

    "max_depth": [4, 5, 6, 7, 8],

    "criterion": ["gini", "entropy"],

}

CV_rfc = GridSearchCV(

    estimator=rfc, param_grid=param_grid, cv=3, scoring="roc_auc", verbose=8, n_jobs=-1

)

CV_rfc.fit(X_train, y_train)

params = {

    "criterion": "gini",

    "max_depth": 6,

    "max_features": "auto",

    "n_estimators": 200,

    "random_state": 22,

}

rfc = RandomForestClassifier(

    criterion="gini",

    max_depth=6,

    max_features="auto",

    n_estimators=200,

    random_state=22,

)

data = ndf.sample(frac=1)

rfc.fit(data.values[:, :-1], data.values[:, -1])

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print(classification_report(y_pred=y_pred, y_true=y_test))

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

CV_rfc.fit(x_train, y_train)

from sklearn import preprocessing

from sklearn.model_selection import train_test_split as tts



X, y = ndf.drop(["id", "target"], axis=1).values, ndf["target"].values

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.05, random_state=22)

from sklearn.linear_model import LogisticRegression as LR



model = LR(class_weight={0: 0.01, 1: 1})

model = LR()

model = XGBClassifier()

model.fit(X_train, y_train)

model.fit(X, y)

model.fit(X_train, y_train)

print(classification_report(y_true=y_train, y_pred=model.predict(X_train)))

print(classification_report(y_true=y_test, y_pred=model.predict(X_test)))

'''

#END