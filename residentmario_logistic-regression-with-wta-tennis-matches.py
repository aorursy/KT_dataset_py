import pandas as pd

matches = pd.read_csv("../input/wta_matches_2015.csv")

matches.head()
import numpy as np



point_diff = (matches.winner_rank_points - matches.loser_rank_points).dropna()

X = point_diff.values[:, np.newaxis]

y = (point_diff > 0).values.astype(int).reshape(-1, 1)



sort_order = np.argsort(X[:, 0])

X = X[sort_order, :]

y = y[sort_order, :]
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

clf.fit(X, y.ravel())

y_hat = clf.predict(y[:, 0].reshape(-1, 1))

y_hat = y_hat[sort_order]
pd.Series(y[:, 0]).value_counts()
pd.Series(y_hat).value_counts()
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



pd.Series(y_hat == y[:, 0]).value_counts().plot.bar()
clf.predict_proba(y[:, 0].reshape(-1, 1))
clf.coef_
import math



math.exp(1)**clf.coef_