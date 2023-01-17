import numpy as np 

import pandas as pd 

import statsmodels.formula.api as sm

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

total = train.append(test)



#OLS model.... y = X0 + X1 + X5

res = sm.ols(formula="y ~ X0 +X1 +X5 ", data=total).fit()

print(res.summary())



#prediction doesnt work here... don't know why

#prediction = res.predict(exog=test[['X0','X1','X5']].as_matrix())

#prediction = pd.DataFrame(prediction)

#prediction['ID']=test['ID']

#prediction.to_csv('submission_OLS.csv', index=False)
res = sm.ols(formula="y ~ X0 +X1 +X2 +X3 +X4 +X5 +X6 +X8 ", data=total).fit()

print(res.summary())

import statsmodels.formula.api as sm

res = sm.ols(formula="y ~ X0 +X1 +X5 +X6 + X8 +X47 +X2 +X3 +X77+X105 +X345 +X3 +X142 +X26 +X322+X46+X267+X151+X240+X342+X287+X152+X140+X65+X95+X70+X116+X265+X354+X177+X362+X52+X383+X273+X58+X157+X156+X64+X131+X355+X173+X73+X31+X338+X225+X271+X230+X71+X174+X27+X163+X141+X327+X127+X51+X292", data=total).fit()

print(res.summary())









print('Parameters: ', res.params)

print('R2: ', res.rsquared)

print('Standard errors: ', res.bse)

print('Predicted values: ', res.predict())










