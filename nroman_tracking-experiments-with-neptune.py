!pip install neptune-client
import neptune

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



# Initializing a neptune. First argument is your-user-name/project-name, second argument is your API Token.

# It is strongly recommended to store it in an environment variable NEPTUNE_API_TOKEN

neptune.init('kaggle-presentation/kaggle', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIwNTM2NjM2OS1mY2YxLTQyNWQtODQyZi03NWQ5NDhhMWI3YWYifQ==')



# Creating a dataset

X, y = make_classification()

# Splitting it into training and testing

X_train, X_test, y_train, y_test = X[:70], X[70:], y[:70], y[70:]



# Creating experiment in the project defined above

neptune.create_experiment()



# Fitting a model

lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train, y_train)

auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])



# Sending a metric to the experiment

neptune.send_metric('AUC', auc)



# Stop the experiment

neptune.stop()
from sklearn.ensemble import RandomForestClassifier



# Dictionary with parameters

params = {'n_estimators':10,

          'criterion': 'gini',

          'max_depth': 5,

          'min_samples_split': 10,

          'min_samples_leaf': 2,

          'random_state': 47}



# This time we are sending parameters

neptune.create_experiment(params=params)



clf = RandomForestClassifier(**params)

clf.fit(X_train, y_train)

auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])



# Sending a metric to the experiment

neptune.send_metric('AUC', auc)



# Stop the experiment

neptune.stop()
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt





# Dictionary with parameters

params = {'hidden_layer_sizes': (200,),

          'activation': 'relu',

          'max_iter': 500,

          'learning_rate': 'adaptive',

          'random_state': 47}



neptune.create_experiment(params=params)



clf = MLPClassifier(**params)

clf.fit(X_train, y_train)

auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])



# Sending a metric to the experiment

neptune.send_metric('AUC', auc)



plt.plot(clf.loss_curve_)

plt.savefig('loss_curve.png')

neptune.send_image('loss_curve', 'loss_curve.png')

neptune.stop()
import joblib



with neptune.create_experiment():

    clf = LogisticRegression(solver='lbfgs', random_state=47)

    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    neptune.send_metric('AUC', auc)

    joblib.dump(clf, 'logistic_regression.pkl')

    neptune.send_artifact('logistic_regression.pkl')