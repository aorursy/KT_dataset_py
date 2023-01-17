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
# Import Necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import sklearn



from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

from sklearn.metrics import roc_curve, auc, matthews_corrcoef, make_scorer

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler



from sklearn.datasets import load_digits, load_breast_cancer, fetch_20newsgroups_vectorized



from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.svm import LinearSVC, NuSVC, SVC

from sklearn.neural_network import MLPClassifier

from sklearn.cluster import k_means

from sklearn.tree import DecisionTreeClassifier



import imblearn



import itertools

from pprint import pprint



from tqdm import tqdm_notebook as tqdm



# Confiure constants

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5
CV = 5

MAX_PER_TYPE = 5
X_train = pd.read_csv("../input/data_train.csv", index_col="index")

Y_train = pd.read_csv("../input/answer_train.csv", index_col="index")

X_test = pd.read_csv("../input/data_test.csv", index_col="index")

Y_test = pd.read_csv("../input/answer_sample.csv", index_col="index")



print(X_train.shape)



bc = imblearn.over_sampling.SMOTE()

X_train, Y_train = bc.fit_resample(X_train, Y_train)

print(X_train.shape, Y_train.shape)
SCORE = 'f1_macro'



MODELS = {

    "DummyClassifier": [DummyClassifier(), {

        'strategy': ('uniform', 'stratified', 'most_frequent'),

    }],

    "LR_multinomial": [LogisticRegression(solver='saga', multi_class='multinomial'), {

        'penalty': ('none', 'l1', 'l2'),

        'C': [0.01, 0.1, 1, 5],

    }],

    "LR_ovr": [LogisticRegression(solver='liblinear', multi_class='ovr'), {

        'penalty': ('l1', 'l2'),

        'C': [0.01, 0.1, 1, 5],

    }],

    "LR_elasticnet": [LogisticRegression(solver='saga', multi_class='multinomial', penalty='elasticnet'), {

        'C': [0.01, 0.1, 1, 5],

        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],

    }],

    "kNN": [KNeighborsClassifier(), {

        'weights': ('uniform', 'distance'),

        'n_neighbors': [1, 3, 5, 10, 15, 20, 50, 100],

    }],

    "RF": [RandomForestClassifier(), {

        'n_estimators': [50, 100, 150, 200],

        'max_depth': [10, 25, 50, 100, 200],

    }],

    "DT": [DecisionTreeClassifier(), {

        'criterion': ('gini', 'entropy'),

        'max_depth': [10, 25, 50, 100, 200],

    }],

    "GaussianNB": [GaussianNB(), {}],

    "BernoulliNB": [BernoulliNB(), {}],

    "MultinomialNB": [MultinomialNB(), {}],

}



res = []



# with tqdm(MODELS.items()) as t:

#     for name, (model, parameters) in t:

#         t.set_description(name)

#         clf = GridSearchCV(model, parameters, scoring=SCORE, cv=CV, iid=False, n_jobs=4)

#         clf.fit(X_train, Y_train)

#         model_res = pd.DataFrame.from_dict(clf.cv_results_)[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

#         model_res["model_type"] = name

#         res += model_res[["model_type", "params", "mean_test_score", "std_test_score"]].sort_values(by=["mean_test_score"], ascending=False).to_dict('records')[:MAX_PER_TYPE]



# final = pd.DataFrame.from_records(res).sort_values(by=["mean_test_score"], ascending=False).reset_index(drop=True)

# final
model = RandomForestClassifier(n_estimators=100, max_depth=10)

model.fit(X_train, Y_train)

pred = model.predict(X_test)

print(model.score(X_train, Y_train))

print(classification_report(Y_train, model.predict(X_train)))

# print(model.score(X_test, Y_test))

pred_df = pd.DataFrame(data={'default.payment.next.month': pred}).reset_index()

pred_df.to_csv("./pred.csv", index=False)
print(os.listdir("."))
sorted(sklearn.metrics.SCORERS.keys())