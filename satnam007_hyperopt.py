import pandas as pd

import numpy as np



from sklearn import ensemble

from sklearn import metrics

from sklearn import model_selection

from sklearn import preprocessing

from sklearn import decomposition

from sklearn import pipeline
from functools import partial

from skopt import space

from skopt import gp_minimize
from hyperopt import fmin

from hyperopt import hp

from hyperopt import tpe

from hyperopt import Trials
from hyperopt.pyll.base import scope #for the format int
df = pd.read_csv('../input/mobile-price-classification/train.csv')

X  = df.drop('price_range', axis = 1).values

y  = df['price_range'].values
def optimize(params, x, y):

    

    model  = ensemble.RandomForestClassifier(**params) #**params to read the dict

    kf     = model_selection.StratifiedKFold(n_splits = 5)

    

    accuracies = []

    for idx in kf.split(X=x, y=y):

        train_idx, test_idx = idx[0], idx[1]

        

        xtrain = x[train_idx]

        ytrain = y[train_idx]

        xtest = x[test_idx]

        ytest = y[test_idx]

        

        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)

        

        accuracies.append(fold_acc)

    

    return -1*np.mean(accuracies)
#dictionalry

param_space = {

    "max_depth":scope.int(hp.quniform("max_depth", 3,15, 1)), #hp.quniform(label, low, high, q)

    "n_estimators":scope.int(hp.quniform("n_estimators", 100, 600, 1)),

    "criterion":hp.choice("criterion", ["gini", "entropy"]),

    

    "max_features":hp.uniform("max_features", 0.1,1)

}
optimization_func = partial(optimize, x = X, y = y)
trials = Trials()
result = fmin(fn = optimization_func, space = param_space, algo = tpe.suggest, max_evals = 15, trials = trials, verbose = 10)
print(result)
classifier = ensemble.RandomForestClassifier(criterion ='entropy', max_depth = 8, 

                                        max_features = 0.8917683974762745, n_estimators = 411, n_jobs=-1)
from sklearn.model_selection import cross_val_score

score = cross_val_score(classifier,X,y, cv=10)

print('scores\n',score)

print('\ncv values', score.shape)

print('\nScore_Mean', score.mean())