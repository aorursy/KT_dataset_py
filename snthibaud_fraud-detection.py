import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv")

del df["isFlaggedFraud"]  # According to description this is a binarization of amount - not necessary for our model

df["type"] = df["type"].astype("category")
frauds = df[df["isFraud"] == 1]

fraud_count = frauds.shape[0]
frauds["nameOrig"].nunique()/fraud_count
frauds["nameDest"].nunique()/fraud_count
from seaborn import lineplot

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(25,7))

plt.title("Fraud counts over time (hours)")

lineplot(data=frauds.groupby("step")["isFraud"].count())
del df["nameOrig"]

del df["nameDest"]

df = df.sort_values("step")  # Ensure rows are sorted by step so that test set contains more recent transactions than training set

del df["step"]

df
from seaborn import distplot

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(25,7))

for i, a in enumerate(["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]):

    df[a] = np.log1p(df[a])

    fig.add_subplot(230+i+1)

    distplot(df[a])
df.groupby("isFraud")["isFraud"].count()
df["amountRelativeToOrigin"] = (df.amount/df.oldbalanceOrg).replace(np.inf, np.finfo(np.float32).max).fillna(0)

df["amountRelativeToDestination"] = (df.amount/df.oldbalanceDest).replace(np.inf, np.finfo(np.float32).max).fillna(0)
df = pd.get_dummies(df)  # One-hot encoding of type attribute

df
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.model_selection import train_test_split



# Create datasets for classification - step attribute is excluded since it contains no information that should be used in the model

attributes = list(sorted(set(df.columns) - {"isFraud"}))

X, y = df[attributes], df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False) # I noticed most frauds were recent (see below), so I use a small 5% test set

del X

del y

clf = DecisionTreeClassifier(class_weight="balanced")
y_train.sum()
y_test.sum()
from scipy.stats import uniform, randint

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from sklearn.metrics import roc_auc_score, make_scorer



def _roc_auc_score(y_true, y_score, **kwargs):  # Scoring function override to see scores during hyper parameter tuning

    score = roc_auc_score(y_true, y_score, **kwargs)

    print("AUC score: {}".format(score))

    return score



distributions = dict(criterion=["gini", "entropy"], max_depth=randint(2, 13), min_samples_split=uniform(loc=0, scale=0.5))

rscv = RandomizedSearchCV(clf, distributions, scoring=make_scorer(_roc_auc_score), cv=3, n_iter=10, random_state=42, verbose=2)

search = rscv.fit(X_train, y_train)

print("Best AUC score: {}".format(search.best_score_))

print(search.best_params_)

final_model = rscv.best_estimator_
roc_auc_score(y_test, final_model.predict(X_test))
# Based on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, final_model.predict(X_test))

lw = 2

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show()
plt.figure(figsize=(25,10), dpi=300)

plot_tree(final_model, feature_names=attributes)