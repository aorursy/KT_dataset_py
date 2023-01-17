import numpy as np 

import pandas as pd 

import gc



from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OrdinalEncoder, StandardScaler



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv').fillna('Unknown')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv').fillna('Unknown')

# sample = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
X = train.copy()

X_test = test.copy()



del train

gc.collect()
X['Date'] = pd.DatetimeIndex(X['Date'])

X_test['Date'] = pd.DatetimeIndex(X_test['Date'])



X['Month'] = X['Date'].dt.month

# X['Day'] = X['Date'].dt.day



X_test['Month'] = X_test['Date'].dt.month

# X_test['Day'] = X_test['Date'].dt.day



# X['dayofweek'] = X['Date'].dt.dayofweek

X['dayofyear'] = X['Date'].dt.dayofyear

X['quarter'] = X['Date'].dt.quarter

X['weekofyear'] = X['Date'].dt.weekofyear



# X_test['dayofweek'] = X_test['Date'].dt.dayofweek

X_test['dayofyear'] = X_test['Date'].dt.dayofyear

X_test['quarter'] = X_test['Date'].dt.quarter

X_test['weekofyear'] = X_test['Date'].dt.weekofyear
columns = ['Country_Region', 'Target']



oe = OrdinalEncoder()

oe.fit(X[columns])

X[columns] = oe.transform(X[columns])

X_test[columns] = oe.transform(X_test[columns])
Y = X['TargetValue']

X = X.drop(['County','Province_State','Id','TargetValue', 'Date'], axis=1)

X_test = X_test.drop(['County','Province_State','ForecastId', 'Date'], axis=1)



X
print(Y.nunique())

print(Y.shape)
ss1 = StandardScaler()

ss1.fit(X['Population'].values.reshape(-1,1))

X['Population'] = ss1.transform(X['Population'].values.reshape(-1,1))

X_test['Population'] = ss1.transform(X_test['Population'].values.reshape(-1,1))



# ss2 = StandardScaler()

# Y = ss2.fit_transform(Y.values.reshape(-1,1))

# Y = Y.reshape(Y.shape[0])



X = X.values

X_test = X_test.values



# Y = pd.Series(Y)

Y = Y.values
# from sklearn.model_selection import train_test_split



# X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
# rf = RandomForestRegressor(n_jobs=-1)

# xgb = XGBRegressor(n_jobs=-1)
from sklearn.model_selection import KFold



skf = KFold(n_splits=3)

best = None

best_eval = 1000000

for train_index, test_index in skf.split(X, Y):

    X_train, x_test = X[train_index], X[test_index]

    y_train, y_test = Y[train_index], Y[test_index]

    rf = RandomForestRegressor(n_jobs=-1)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(x_test)

    e = np.sqrt(mean_squared_error(y_pred, y_test))

    if e < best_eval or not best:

        best = rf

        best_eval = e



print(best_eval)
skf = KFold(n_splits=3)

bestx = None

best_evalx = 1000000

for train_index, test_index in skf.split(X, Y):

    X_train, x_test = X[train_index], X[test_index]

    y_train, y_test = X[train_index], Y[test_index]

    xgb = XGBRegressor(n_jobs=-1)

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(x_test)

    e = np.sqrt(mean_squared_error(y_pred, y_test))

    if e < best_evalx or not best:

        bestx = xgb

        best_evalx = e



print(best_evalx)
# xgb.fit(X_train, y_train)

# y_pred = xgb.predict(x_test)

# np.sqrt(mean_squared_error(y_pred, y_test))
estimators = [('rf',best), ('xgb',bestx)]



model = VotingRegressor(estimators, n_jobs=-1)
model.fit(X, Y)
pred = model.predict(X_test)

pred = ss2.inverse_transform(pred.reshape(-1,1))

pred = pred.reshape(pred.shape[0])

pred
test['q_0.05'] = pred*0.05

test['q_0.5'] = pred*0.5

test['q_0.95'] = pred*0.95
test['Date'] = pd.DatetimeIndex(test['Date'])

test.set_index('Date', inplace=True)
# x1 = test['q_0.05'].resample('1D').quantile(0.05)

# x2 = test['q_0.5'].resample('1D').quantile(0.5)

# x3 = test['q_0.95'].resample('1D').quantile(0.95)
gc.collect()
# for d in x1.index:

#     test.loc[d, 'q_0.05'] = x1[d]

#     test.loc[d, 'q_0.5'] = x2[d]    

#     test.loc[d, 'q_0.95'] = x3[d]   

    

# del x1, x2, x3

# gc.collect()
# test
df = pd.melt(test, 'ForecastId', ['q_0.05', 'q_0.5', 'q_0.95'])
df['variable']=df['variable'].str.replace("q","", regex=False)

df['ForecastId_Quantile']=df['ForecastId'].astype(str)+df['variable']

df['TargetValue']=df['value']
df[['ForecastId_Quantile', 'TargetValue']].to_csv('submission.csv', index=False)

df[['ForecastId_Quantile', 'TargetValue']]