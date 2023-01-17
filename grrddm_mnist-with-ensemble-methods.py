from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Loading the training dataset and splitting it into features and labels
train = pd.read_csv("../input/train.csv")
y_train, x_train = train.label.values, train.drop("label", axis=1).values
plt.imshow(x_train[0].reshape(28, 28), interpolation="gaussian");
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, random_state=1643)
# **Considering the models suggested to build the ensemble:**
rf_clf = RandomForestClassifier(n_jobs=-1)

et_clf = ExtraTreesClassifier(n_jobs=-1)

svm_clf = Pipeline([
    ("standarize", StandardScaler()),
    ("svc", SVC(verbose=2))
])
hard_voting_ensemble = VotingClassifier(estimators=[
    ("random_forest", rf_clf),
    ("extra_trees", et_clf),
    ("svm", svm_clf)
], voting="hard")
rf_clf.fit(x_train, y_train)
et_clf.fit(x_train, y_train)

svm_clf.fit(x_train, y_train)

hard_voting_ensemble.fit(x_train, y_train)
tf_predict = rf_clf.predict(x_validate)
et_predict = et_clf.predict(x_validate)
svm_predict = svm_clf.predict(x_validate)
hv_predict = hard_voting_ensemble.predict(x_validate)
test = pd.read_csv("../input/test.csv").values
test_predict = hard_voting_ensemble.predict(test)
len(range(1, len))
test_predict_df = pd.DataFrame({"ImageId": range(1, len(test_predict) + 1),
                                 "Label": test_predict})
test_predict_df.index = test_predict_df.ImageId
test_predict_df.drop("ImageId", axis=1, inplace=True)
test_predict_df.head()
test_predict_df.to_csv("MNIST_test_pred.csv", header=True)