import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from sklearn.metrics import make_scorer, roc_auc_score

from sklearn.linear_model import LogisticRegression
train_data = pd.read_csv('../input/train.csv', na_values='-1')

train_data.fillna(value=train_data.median(), inplace=True)
X_train, y_train = train_data.iloc[:, 2:], train_data.iloc[:, 1]



del train_data
def gini_normalized(y_actual, y_pred):

    """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""

    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1

    return gini(y_actual, y_pred) / gini(y_actual, y_actual)
lr = LogisticRegression()



train_sizes, train_scores, test_scores = learning_curve(

    estimator=lr,

    X=X_train,

    y=y_train,

    train_sizes=np.linspace(0.05, 1, 6),

    cv=5,

    scoring=make_scorer(gini_normalized)

)
train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 

         color='blue', marker='o', 

         markersize=5, 

         label='training gini')

plt.fill_between(train_sizes, 

                 train_mean + train_std,

                 train_mean - train_std, 

                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, 

         color='green', linestyle='--', 

         marker='s', markersize=5, 

         label='validation gini')

plt.fill_between(train_sizes, 

                 test_mean + test_std,

                 test_mean - test_std, 

                 alpha=0.15, color='green')

plt.grid()

plt.xlabel('Number of training samples')

plt.ylabel('Normalized Gini')

plt.legend(loc='lower right')

plt.ylim([-0.25, 0.25])

plt.show()
def gini_normalized(y_actual, y_pred):

    """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""

    

    # If the predictions y_pred are binary class probabilities

    if y_pred.ndim == 2:

        if y_pred.shape[1] == 2:

            y_pred = y_pred[:, 1]

    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1

    return gini(y_actual, y_pred) / gini(y_actual, y_actual)
lr = LogisticRegression()



train_sizes, train_scores, test_scores = learning_curve(

    estimator=lr,

    X=X_train,

    y=y_train,

    train_sizes=np.linspace(0.05, 1, 6),

    cv=5,

    scoring=make_scorer(gini_normalized, needs_proba=True)

)
train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 

         color='blue', marker='o', 

         markersize=5, 

         label='training gini')

plt.fill_between(train_sizes, 

                 train_mean + train_std,

                 train_mean - train_std, 

                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, 

         color='green', linestyle='--', 

         marker='s', markersize=5, 

         label='validation gini')

plt.fill_between(train_sizes, 

                 test_mean + test_std,

                 test_mean - test_std, 

                 alpha=0.15, color='green')

plt.grid()

plt.xlabel('Number of training samples')

plt.ylabel('Normalized Gini')

plt.legend(loc='lower right')

plt.ylim([0.2, 0.3])

plt.show()