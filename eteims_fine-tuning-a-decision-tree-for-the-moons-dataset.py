import numpy as np

import matplotlib.pyplot as plt
# This function will help in visualization of our dataset. 

def plot_dataset(X, y, axes):

    plt.figure(figsize=(10,6))

    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs",alpha = 0.5)

    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^",alpha = 0.2)

    plt.axis(axes)

    plt.grid(True, which='both')

    plt.xlabel(r"$x_1$", fontsize=20)

    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
from sklearn.datasets import make_moons



X, y = make_moons(n_samples=10000, noise=0.4, random_state=21)
plot_dataset(X, y, [-3, 5, -3, 3])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
from sklearn.tree import DecisionTreeClassifier



tree_clf = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV



parameter = {

             'criterion' : ["gini", "entropy"],

             'max_leaf_nodes': list(range(2, 50)), 

             'min_samples_split': [2, 3, 4]

            }
clf = GridSearchCV(tree_clf, parameter, cv = 5,scoring = "accuracy",return_train_score=True,n_jobs=-1)
clf.fit(X_train, y_train)
clf.best_params_
cvres = clf.cv_results_

for mean_score, params in zip(cvres["mean_train_score"], cvres["params"]):

    print(mean_score, params)
clf.score(X_train, y_train)
from sklearn.metrics import confusion_matrix



pred = clf.predict(X_train)

confusion_matrix(y_train,pred)
from sklearn.metrics import precision_score, recall_score



pre = precision_score(y_train, pred)

re  = recall_score(y_train, pred)

print(f"Precision: {pre}  Recall:{re}")
from sklearn.metrics import f1_score



f1_score(y_train, pred)
clf.score(X_test, y_test)