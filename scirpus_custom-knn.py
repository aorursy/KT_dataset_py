import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
train = pd.read_csv('../input/train.csv')
cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
        '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
        'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
        '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
        'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
        '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
        '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']
rows = np.array([2072,3493,379,2972,2367,4415,2791,3980,194,1190,3517,811,4444])-1
df = train.loc[rows, list(['ID','target'])+list(cols)].copy().reset_index(drop=True)
df
def mydist(x, y, **kwargs):
    for lag in range(kwargs['lag']+1):
        z = np.sum(((x[2+lag:])-(y[2+lag:]))**2)
        if(z==0):
            return z
    return 10000000.

X = df[cols].values
Y = range(X.shape[0])
knncustom = KNeighborsRegressor(n_neighbors=1, algorithm='ball_tree',
                metric=mydist, metric_params={"lag": 0},n_jobs=1)
knncustom.fit(X, Y)

print(df.target)
print(df.target[knncustom.predict(X[:])])
print('Error',np.sqrt(mean_squared_error(np.log1p(df.target),np.log1p(df.target[knncustom.predict(X)]))))