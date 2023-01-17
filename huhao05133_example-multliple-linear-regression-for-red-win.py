import numpy as np

import pandas as pd

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression
df = pd.read_csv('../input/wine-whitered/Wine_red.csv',sep=';')

df.head(3)
features = df.drop('quality',axis=1)

targets = df.quality
#standarize featrure keep col info

features = (features - features.mean())/features.std()
#fit mulitiple linear regression

lr = LinearRegression()

lr.fit(features,targets)

R2_train = lr.score(features,targets)

print('training R2',R2_train.round(2))
features.columns
#examine coefficients

coef = lr.coef_

coef = pd.Series(coef,index=features.columns)

coef.sort_values(ascending=False).round(2)
#cross-validation

results = cross_validate(lr,features,targets,return_train_score=True)

R2_train= results['train_score'].mean()

R2_test= results['test_score'].mean()

print('train R2',R2_train.round(2))

print('test R2',R2_test.round(2))