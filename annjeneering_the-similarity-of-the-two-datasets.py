from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

from scipy import stats

import pandas as pd

import numpy as np



from sklearn import metrics
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)



y = iris.target



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42)
stats.ks_2samp(X_train.iloc[:, 0], X_test.iloc[:, 0])


fig, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(18,5))

ax = ax.ravel() 



for idx in range(X_train.shape[1]):

    ax[idx].hist(X_train.iloc[:, idx],  alpha=0.5, label="Train")

    ax[idx].hist(X_test.iloc[:, idx],  alpha=0.5, label="Test")

    ax[idx].set_title(X_train.columns[idx])

    ax[idx].legend()
def modelsGeneration(X_train, X_test, y_train, y_test):



    train_error = []



    for i in range(10, 1000, 10):

        

        model = RandomForestClassifier(n_estimators=i,max_depth=3, oob_score=True)

        model.fit(X_train, y_train)



        y_pred_train = model.predict(X_train)

        y_pred_test = model.predict(X_test)





        score_train = 1 - model.oob_score_



        train_error.append(score_train)

        

    return train_error



train_error = modelsGeneration(X_train, X_test, y_train, y_test)