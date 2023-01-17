import numpy as np

import pandas as pd

import os

import scipy as sp

from functools import partial

from sklearn import metrics

from collections import Counter

import json
# put numerical value to bins

def to_bins(x, tresholds):

    if x <= tresholds[0]:

        return 0

    for i in range(1, len(tresholds)):

        if x > tresholds[i - 1] and x <= tresholds[i]:

            return i

    return len(tresholds)



class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0

    

    def _loss(self, coef, X, y, idx):

        X_p = np.array([to_bins(pred, coef) for pred in X])

        ll = -metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return ll



    def fit(self, X, y):

        coefs = []

        nsplits = 4

        for split_i in range(nsplits):

            coef = [1.5, 2.0, 2.5, 3.0]

            golden1 = 0.618

            golden2 = 1 - golden1

            ab_start = [(1, 2), (1.5, 2.5), (2, 3), (2.5, 3.5)]

            for it1 in range(10):

                for idx in range(4):

                    # golden section search

                    a, b = ab_start[idx]

                    # calc losses

                    coef[idx] = a

                    la = self._loss(coef, X[split_i::nsplits], y[split_i::nsplits], idx)

                    coef[idx] = b

                    lb = self._loss(coef, X[split_i::nsplits], y[split_i::nsplits], idx)

                    for it in range(20):

                        # choose value

                        if la > lb:

                            a = b - (b - a) * golden1

                            coef[idx] = a

                            la = self._loss(coef, X[split_i::nsplits], y[split_i::nsplits], idx)

                        else:

                            b = b - (b - a) * golden2

                            coef[idx] = b

                            lb = self._loss(coef, X[split_i::nsplits], y[split_i::nsplits], idx)

            coefs.append(coef)

        coef = list(np.array(coefs).mean(axis=0))

        self.coef_ = {'x': coef}

    

    def predict(self, X, coef):

        X_p = np.array([to_bins(pred, coef) for pred in X])

        return X_p



    def coefficients(self):

        return self.coef_['x'] 
optR = OptimizedRounder()

optR.fit(valid_predictions, targets)

coefficients = optR.coefficients()

valid_predictions = optR.predict(valid_predictions, coefficients)

test_predictions = optR.predict(test_predictions, coefficients)