import pandas as pd

import scipy.stats as ss

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv")
train.columns
cols = train.columns[1:116]



for i in cols:

    print(train[i].unique())
X = train.ix[:,117:131]

y = train['loss']
X.dtypes
cvs = cross_val_score(estimator=RandomForestRegressor(), 

                      X=X, 

                      y=y, 

                      cv=5, 

                      scoring='mean_absolute_error')
cvs.mean()
y.describe()
plt.hist(y, bins=1000)

plt.show()
fit_alpha, fit_loc, fit_beta=ss.gamma.fit(y)

print(fit_alpha, fit_loc, fit_beta)
pred = ss.gamma.rvs(a=fit_alpha, loc=fit_loc, scale=fit_beta, size = X.shape[0])
diffs = abs(y - pred)
diffs.mean()