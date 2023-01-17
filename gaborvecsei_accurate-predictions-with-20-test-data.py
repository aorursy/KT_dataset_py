import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
data = pd.DataFrame.from_csv("../input/adult.csv", header=0, index_col=None)
data.head(5)
data["income"].value_counts()
data.isnull().values.any()
data = data.replace("?", np.nan)
data.isnull().sum()
null_data = data[pd.isnull(data).any(1)]

null_data["income"].value_counts()
data.dropna(inplace=True)
bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

categories = pd.cut(data["age"], bins, labels=group_names)

data["age"] = categories
sns.countplot(y='native.country',data=data)
sns.countplot(y='education', hue='income', data=data)
sns.countplot(y='occupation', hue='income', data=data)
# education = education.num

data.drop(["education", "fnlwgt"], axis=1, inplace=True)
from sklearn import preprocessing
for f in data:

    if f in ["age", "workclass", "marital.status", "occupation", "relationship", "race", "sex", "native.country", "income"]:

        le = preprocessing.LabelEncoder()

        le = le.fit(data[f])

        data[f] = le.transform(data[f])

data.head(5)
y = data["income"]

X = data.drop(["income"], axis=1)
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=100,random_state=0)



forest.fit(X, y)



importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)

indices = np.argsort(importances)[::-1]



plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
X = data.drop(["race", "native.country", "sex", "capital.loss", "workclass", "age"], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
forest = RandomForestClassifier(10)

forest.fit(X_train, y_train)
predictions = forest.predict_proba(X_test)

predictions = [np.argmax(p) for p in predictions]
precision = accuracy_score(predictions, y_test) * 100
print("Precision: {0}%".format(precision))