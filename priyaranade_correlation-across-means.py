import kagglegym

env = kagglegym.make()

observation = env.reset()

train = observation.train
import statsmodels.api as smapi

import numpy as np

train.fillna(0, inplace=True)



import statsmodels.api as smapi

import scipy.stats as st

train.fillna(0, inplace=True)

   

corr = {}



for col in train.columns:

    cond = ~(train[col] > train[col].mean() + 4 * train[col].std())

    cond = cond & ~(train[col] < train[col].mean() - 4 * train[col].std())

    

    df = train.loc[cond, train.columns]

    y = train.y.reindex(df.index)

    corr[col] = np.corrcoef(df[col], y)[0, 1]
import operator

tuples = sorted(rsquared.items(), key=operator.itemgetter(1), reverse=True)

print(tuples)
