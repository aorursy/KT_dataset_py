# 初回のみ実行すればよい

!pip install mglearn
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import mglearn
mglearn.plots.plot_cross_validation()
from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression



iris = load_iris()

logreg = LogisticRegression()



scores = cross_val_score(logreg, iris.data, iris.target)

print('交差検証スコア：',scores)
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)

print('交差検証スコア：',scores)
print('交差検証スコアの平均：',scores.mean())
iris = load_iris()

print('Irisラベル：\n',iris.target)
mglearn.plots.plot_stratified_cross_validation()
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)

scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)

print('交差検証スコアの平均：',scores)
kfold = KFold(n_splits=3)

scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)

print('交差検証スコアの平均：',scores)
kfold = KFold(n_splits=3, shuffle=True, random_state=0)

scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)

print('交差検証スコアの平均：',scores)
mglearn.plots.plot_shuffle_split()
from sklearn.model_selection import ShuffleSplit

shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)

scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)

print("Cross-validation scores:\n{}".format(scores))