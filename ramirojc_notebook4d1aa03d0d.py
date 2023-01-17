# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

import pandas

import numpy

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.datasets import load
model = LogisticRegression()

rfe = RFE(model, 2)

rfe = rfe.fit(dataset.data, dataset.target)
print(rfe.support_)

print(rfe.ranking_)
from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()

model.fit(dataset.data, dataset.target)
print(model.feature_importances_)
import pandas as pd

import numpy as np

import csv

import requests





from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2







array = dataframe.values