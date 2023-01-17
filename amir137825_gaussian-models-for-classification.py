import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import Image
features = [ 'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist' ]

raw_data = pd.read_csv('../input/magic-gamma-telescope-dataset/telescope_data.csv', names=features + ['class'], skiprows=1)
raw_data
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(raw_data['class'])

raw_data['class'] = le.transform(raw_data['class'])

X = raw_data[features].values

y = raw_data['class'].values
raw_data['class'].plot.hist()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
def evaluate_clf(X_train, X_test, y_train, y_test, clf, name="Classifier"):

  from sklearn.metrics import f1_score, accuracy_score

  # fit the classifier

  clf.fit(X_train, y_train)

  pred = clf.predict(X_test)

  # evaluate prediction using acc and f1 score

  score_f1 = f1_score(y_test, pred)

  score_acc = accuracy_score(y_test, pred)

  print('{} acc-score: {}'.format(name, score_acc))

  print('{} f1-score: {}'.format(name, score_f1))
from sklearn.tree import DecisionTreeClassifier

evaluate_clf(X_train, X_test, y_train, y_test, DecisionTreeClassifier(), "Decision Tree")
Image('../input/gaussiannotebookimg/2.png')
from sklearn.naive_bayes import GaussianNB

evaluate_clf(X_train, X_test, y_train, y_test, GaussianNB(), "Gaussian NB")
raw_data[features].corr()
raw_data[features].plot.scatter('fSize', 'fConc')
Image('../input/gaussiannotebookimg/3.png')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

evaluate_clf(X_train, X_test, y_train, y_test, LinearDiscriminantAnalysis(), "LDA")
Image('../input/gaussiannotebookimg/4.png')
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

evaluate_clf(X_train, X_test, y_train, y_test, QuadraticDiscriminantAnalysis(), "QDA")
Image('../input/gaussiannotebookimg/5.png')
from sklearn.base import BaseEstimator

from sklearn.mixture import GaussianMixture



class GaussianMixtureClassifier(BaseEstimator):

  

  def __init__(self, n_components=1):

    self.n_components = n_components



  def fit(self, X, y):

    # find number of classes

    self.n_classes = int(y.max() + 1)

    # create a GM for each class

    self.gm_densities = [GaussianMixture(self.n_components, covariance_type='full') for _ in range(self.n_classes)]

    # fit the Mixture densities for each class

    for c in range(self.n_classes):

      # find the correspond items

      temp = X[np.where(y == c)]

      # estimate density parameters using EM

      self.gm_densities[c].fit(temp)



  def predict(self, X):

    # calculate log likelihood for each class

    log_likelihoods = np.hstack([ self.gm_densities[c].score_samples(X).reshape((-1, 1)) for c in range(self.n_classes) ])

    # return the class whose density maximizes the log likelihoods

    log_likelihoods = log_likelihoods.argmax(axis=1)

    return log_likelihoods
evaluate_clf(X_train, X_test, y_train, y_test, GaussianMixtureClassifier(n_components=2), "Gaussian Mixture")