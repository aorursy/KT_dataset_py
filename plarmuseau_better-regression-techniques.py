import numpy as np

import xgboost as xgb

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression, TheilSenRegressor,RANSACRegressor,HuberRegressor





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

total = train.append(test)

estimators = [('OLS', LinearRegression()),

              ('Theil-Sen', TheilSenRegressor(random_state=42)),

              ('RANSAC', RANSACRegressor(random_state=42)),

              ('HuberRegressor', HuberRegressor())]

lw = 2

cellen=['X'+str(w) for w in range(10,385) if str(w) not in ['25','72','121','149','188','193','303','381']]

cellen=['X47','X95','X314','X315','X232','X119','X311','X76','X329','X238','X232','X340','X362','X119','X137']

X_=train[cellen]

y_=train['y']

for name, estimator in estimators:

    estimator.fit(X_, y_)

    print(name,estimator.score(X_,y_))

    y_2=estimator.predict(X_)  

    print(y_2)

        

    sub = pd.DataFrame()

    sub['ID'] = test['ID']

    sub['y'] = estimator.predict(test[cellen])

    print(name,sub.T)

    sub.to_csv('Ransac'+name+'.csv', index=False)
