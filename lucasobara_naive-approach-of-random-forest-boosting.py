# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

import sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split

# import xgboost as xgb
data = pd.read_json(open("../input/train.json", "r"))

data.head()
print(data.shape)

data.info()
print(len(data["manager_id"].unique()))

print(len(data["building_id"].unique()))
data["manager_id"] = pd.factorize(data["manager_id"])[0]

data["building_id"] = pd.factorize(data["building_id"])[0]

# data["interest_level"] = pd.factorize(data["interest_level"])[0]

data["num_description_words"] = data["description"].apply(lambda x: len(x.split(" ")))

data["num_features"] = data["features"].apply(lambda x: len(x))

data["num_photos"] = data["photos"].apply(lambda x: len(x))

data["created"] = pd.to_datetime(data["created"])

data["created_year"] = data["created"].dt.year

data["created_month"] = data["created"].dt.month

data["created_day"] = data["created"].dt.day



ranking = {"high": 0, "medium": 1, "low": 2}

data["interest_level"] = np.array(data['interest_level'].apply(lambda x: ranking[x]))
data.head()
list(data.columns.values)
features = ["bathrooms", "bedrooms", "building_id", "latitude", "longitude", "manager_id", "price", 

            "num_description_words", "created_year", "created_month", "num_features", "num_photos"]

target = "interest_level"
X = data[features]

y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)
plt.figure(figsize=(15, 10))



# N Estimators

plt.subplot(2, 3, 1)

feature_param = range(100, 150)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(n_estimators=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(scores, ".-")

plt.axis("tight")

plt.xlabel("parameter")

plt.ylabel("score")

plt.title("N Estimators")

plt.grid()



# Criterion

plt.subplot(2, 3, 2)

feature_param = ["gini","entropy"]

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(criterion=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(scores, ".-")

plt.title("Criterion")

plt.xticks(range(len(feature_param)), feature_param)

plt.grid()



# Max Features

plt.subplot(2, 3, 3)

feature_param = ["auto", "sqrt", "log2", None]

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_features=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(scores, ".-")

plt.axis("tight")

plt.title("Max Features")

plt.xticks(range(len(feature_param)), feature_param)

plt.grid()



# Max Depth

plt.subplot(2, 3, 4)

feature_param = range(1, 21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_depth=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(feature_param, scores, ".-")

plt.axis("tight")

plt.title("Max Depth")

plt.grid()



# Min Weight Fraction Leaf

plt.subplot(2, 3, 5)

feature_param = np.linspace(0, 0.5, 10)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_weight_fraction_leaf=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(feature_param, scores, ".-")

plt.axis("tight")

plt.title("Min Weight Fraction Leaf")

plt.grid()



# Max Leaf Nodes

plt.subplot(2, 3, 6)

feature_param = range(2, 21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_leaf_nodes=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(feature_param, scores, ".-")

plt.axis("tight")

plt.title("Max Leaf Nodes")

plt.grid()
plt.figure(figsize=(15, 10))



# N Estimators

plt.subplot(2, 3, 1)

feature_param = range(1, 21)

scores=[]

for feature in feature_param:

    clf = GradientBoostingClassifier(n_estimators=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(scores, ".-")

plt.axis("tight")

plt.title("N Estimators")

plt.grid()



# Learning Rate

plt.subplot(2, 3, 2)

feature_param = np.linspace(0.1, 1, 10)

scores=[]

for feature in feature_param:

    clf = GradientBoostingClassifier(learning_rate=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(scores, ".-")

plt.title("Learning Rate")

plt.grid()



# Max Features

plt.subplot(2, 3, 3)

feature_param = ["auto", "sqrt", "log2", None]

scores=[]

for feature in feature_param:

    clf = GradientBoostingClassifier(max_features=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(scores, ".-")

plt.axis("tight")

plt.title("Max Features")

plt.grid()



# Max Depth

plt.subplot(2, 3, 4)

feature_param = range(1, 11)

scores=[]

for feature in feature_param:

    clf = GradientBoostingClassifier(max_depth=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(feature_param, scores, ".-")

plt.axis("tight")

plt.title("Max Depth")

plt.grid()



# Min Weight Fraction Leaf

plt.subplot(2, 3, 5)

feature_param = np.linspace(0, 0.5, 10)

scores=[]

for feature in feature_param:

    clf = GradientBoostingClassifier(min_weight_fraction_leaf =feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(feature_param, scores, ".-")

plt.axis("tight")

plt.title("Min Weight Fraction Leaf")

plt.grid()



# Max Leaf Nodes

plt.subplot(2, 3, 6)

feature_param = range(2, 21)

scores=[]

for feature in feature_param:

    clf = GradientBoostingClassifier(max_leaf_nodes=feature)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    score = log_loss(y_test, y_pred)

    scores.append(score)

plt.plot(feature_param, scores, ".-")

plt.axis("tight")

plt.title("Max Leaf Nodes")

plt.grid()
