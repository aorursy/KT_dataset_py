from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import Ridge
df1 = pd.read_csv('../input/nba-2019-season-ratings-stats-details/NBA Stats.csv', delimiter=',')



nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
columns = ['Age', 'Weight', 'gp', 'min', 'pts', 'fgm', 'fga', 'fg%', '3pm', '3p%', 'ftm', 

         "fta","ft%","reb","ast","stl","blk","tov","eff"]



y = df1[['Ratings']]

X = df1[columns]
ridge_0 = Ridge(alpha=0)

ridge_0.fit(X, y)



ridge_1 = Ridge(alpha=1)

ridge_1.fit(X, y)
print(ridge_0.score(X,y))

print(ridge_1.score(X,y))
from sklearn import linear_model



mY = [el[0] for el in y.values]

mX = df1[columns]



n_alphas = 200

alphas = np.logspace(-2, 8, n_alphas)



coefs = []

for a in alphas:

    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)

    ridge.fit(mX, mY)

    coefs.append(ridge.coef_)



ax = plt.gca()



ax.plot(alphas, coefs)

ax.set_xscale('log')

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coefficients as a function of the regularization')

plt.axis('tight')
reg = linear_model.RidgeCV(alphas=np.logspace(-2, 8, n_alphas))

reg.fit(mX, mY)
print(reg.alpha_)
for x in sorted(list(zip(columns, reg.coef_)), key=lambda x: x[1], reverse=True):

    print("{}: {}".format(x[0], x[1]))