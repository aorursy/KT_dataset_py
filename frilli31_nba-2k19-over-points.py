from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model



pd.options.mode.chained_assignment = None
df1 = pd.read_csv('../input/nba-2019-season-ratings-stats-details/NBA Stats.csv', delimiter=',')



nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
columns = ['pts']



y = df1[['Ratings']]

X = df1[columns]





plt.scatter(X, y)
scores = []

best_alphas = []



n_alphas = 200

alphas = np.logspace(-0, 8, n_alphas)



reg = linear_model.RidgeCV(alphas=alphas)

reg.fit(X, y)

scores.append(reg.score(X, y))

best_alphas.append(reg.alpha_)



for i in range(2, 10):

    X["pts^{}".format(i)] = X["pts"]**i

    reg = linear_model.RidgeCV(alphas=alphas)

    reg.fit(X, y)

    scores.append(reg.score(X, y))

    best_alphas.append(reg.alpha_)



res = list(zip(scores, best_alphas))

for x in enumerate(res):

    print("Grade: {}\tScore: {} Best alpha: {}".format(x[0]+1, x[1][0], x[1][1]))
X = X[['pts', 'pts^2', 'pts^3']]



reg = linear_model.Ridge(alpha=50000)

reg.fit(X, y)

print(reg.coef_)



reg_0 = linear_model.Ridge(alpha=0)

reg_0.fit(X, y)

print(reg_0.coef_)
y_pred = reg.predict(X)

y_pred_0 = reg_0.predict(X)



plt.plot(X['pts'],y,'.')

plt.plot(X['pts'],y_pred_0,'.')

plt.plot(X['pts'],y_pred,'.')



plt.figure(figsize=(20, 10))

plt.show()