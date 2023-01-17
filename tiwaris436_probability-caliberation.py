import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# SVM reliability diagram

from sklearn.datasets import make_classification

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.calibration import calibration_curve

from matplotlib import pyplot

from sklearn.calibration import CalibratedClassifierCV
# generate 2 class dataset

X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)

# split into train/test sets

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=2)

# fit a model

model = SVC()

model.fit(trainX, trainy)

# predict probabilities

probs = model.decision_function(testX)

# reliability diagram

fop, mpv = calibration_curve(testy, probs, n_bins=10, normalize=True)

# plot perfectly calibrated

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot model reliability

pyplot.plot(mpv,fop,marker='.')

pyplot.show()

fop
mpv
calibration_curve(testy, probs, n_bins=10, normalize=True)
# generate 2 class dataset

X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)

# split into train/test sets

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

# fit a model

model = SVC()

calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)

calibrated.fit(trainX, trainy)

# predict probabilities

probs = calibrated.predict_proba(testX)[:, 1]

# reliability diagram

fop, mpv = calibration_curve(testy, probs, n_bins=10, normalize=True)

# plot perfectly calibrated

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot calibrated reliability

pyplot.plot(mpv, fop, marker='.')

pyplot.show()
# generate 2 class dataset

X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)

# split into train/test sets

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

# fit a model

model = SVC()

calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)

calibrated.fit(trainX, trainy)

# predict probabilities

probs = calibrated.predict_proba(testX)[:, 1]

# reliability diagram

fop, mpv = calibration_curve(testy, probs, n_bins=10, normalize=True)

# plot perfectly calibrated

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot calibrated reliability

pyplot.plot(mpv, fop, marker='.')

pyplot.show()