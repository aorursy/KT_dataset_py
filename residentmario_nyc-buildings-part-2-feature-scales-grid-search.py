import pandas as pd

buildings = pd.read_csv("../input/MN.csv")

pd.set_option('max_columns', None)

X = buildings.loc[buildings['BldgClass'].astype(str).map(lambda v: v[0]) == 'D',

                  ['LotArea', 'BldgArea', 'ComArea', 'ResArea', 'OfficeArea', 'RetailArea',

                   'GarageArea', 'StrgeArea', 'FactryArea', 'OtherArea', 'NumFloors',

                   'UnitsTotal', 'AssessLand', 'AssessTot', 'ExemptLand', 'ExemptTot']]

y = X['UnitsTotal']

X = X.drop('UnitsTotal', axis='columns')



from sklearn.linear_model import ElasticNetCV

clf = ElasticNetCV(l1_ratio=0.5)

clf.fit(X, y)

clf.alpha_
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

X.loc[:, 'NumFloors'].plot.hist(bins=50)
import numpy as np



def rescale(X):

    return (X - np.min(X)) / (np.max(X) - np.min(X))



rescale(X.NumFloors).plot.hist(bins=50)
def mean_rescale(X):

    return (X - np.mean(X)) / (np.max(X) - np.min(X))



mean_rescale(X.NumFloors).plot.hist(bins=50)
def normalize(X):

    return (X - np.mean(X)) / np.sqrt(np.var(X))
normalize(X.NumFloors).plot.hist(bins=50)
clf = ElasticNetCV(l1_ratio=0.5)

clf.fit(normalize(X), normalize(y))

clf.alpha_
clf.coef_
import itertools

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



class GridSearchCV:

    def __init__(self, estimator, param_grid, scoring=r2_score):

        self.estimator, self.param_grid, self.metric = estimator, param_grid, scoring

    

    def fit(self, X, y):

        # Generate all possible hyperparameter combinations.

        hyperparams = list(itertools.product(*list(self.param_grid.values())))



        # Get ready for cross validation.

        X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.4)

        estimators = [self.estimator(*hyp).fit(X_train, y_train) for hyp in hyperparams]

        scores = [self.metric(e.predict(X_test), y_test) for e in estimators]

        self.subclf = estimators[np.argmax(scores)]

        

    def predict(X, y):

        return subclf.predict(X, y)
from sklearn.linear_model import ElasticNet



clf = GridSearchCV(ElasticNet, {'alpha': [0, 0.5, 0.1, 0.01, 0.001],

                                'l1_ratio': [0, 0.25, 0.5, 0.75, 1]},

                   scoring=r2_score)

clf.fit(normalize(X), 

        normalize(y[:, np.newaxis]))

clf.subclf
clf.subclf.coef_
from sklearn.model_selection import GridSearchCV