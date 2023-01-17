from hyperopt import fmin, tpe, hp
best = fmin(
    fn = lambda x: x,
    space = hp.uniform('x', 0, 1),
    algo = tpe.suggest,
    max_evals = 10
)
print(best)
from sklearn.model_selection import train_test_split
import pandas as pd
SEED = 98105

df = pd.read_csv("../input/Iris.csv")
y = pd.factorize(df["Species"])[0]
X = df.drop(["Species", "Id"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = SEED)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK, Trials

# Define cross validation scoring method
def cv_score(params):
    clf = KNeighborsClassifier(**params)
    score = cross_val_score(clf, X_train, y_train).mean()
    return {'loss': -score, 'status': STATUS_OK}

# space for searching
space = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 50))
}
trials = Trials() # Recorder for trails

# Train
best = fmin(cv_score, space, algo=tpe.suggest, max_evals=100, trials=trials)

# Reporting
import matplotlib.pyplot as plt
n_neighbors = [t['misc']['vals']['n_neighbors'][0] for t in trials.trials]
n_neighbors_df = pd.DataFrame({'n_neighbors': n_neighbors, 'loss': trials.losses()})
plt.scatter(n_neighbors_df.n_neighbors, n_neighbors_df.loss)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

space = hp.choice('classifier_type', [ # Here we are dealing with a classification problem
    {
        'type': 'naive_bayes',
        'alpha': hp.uniform('alpha', 0.0, 2.0),
    },
    {
        'type': 'svm',
        'C': hp.uniform('C', 0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0),
    },
    {
        'type': 'randomforest',
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1]),
        'n_jobs': -1
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', range(1,50)),
        'n_jobs': -1
    }
])

def cv(params):
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X, y).mean()

def cv_score(params):
    score = cv(params)
    return {'loss': -score, 'status': STATUS_OK}
trials = Trials()
best = fmin(cv_score, space, algo=tpe.suggest, max_evals=1500, trials=trials)
print(best)
arr = trials.losses()
plt.scatter(range(len(arr)), arr, alpha = 0.3)
plt.ylim(-1, -0.8)
plt.show()
print("Lowest score:" ,trials.average_best_error())
import hyperopt
from sklearn.metrics import accuracy_score

def train(params):
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    model = clf.fit(X, y)
    return model

model = train(hyperopt.space_eval(space, best))

preds = model.predict(X_test)

print("On test data, we acccheived an accuracy of: {:.2f}%".format(100 * accuracy_score(y_test, preds)))
