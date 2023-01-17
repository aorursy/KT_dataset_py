import os
print(os.listdir("../input"))
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
data_dir = Path('../input')

train_df = pd.read_csv(data_dir / 'train.csv')
test_df = pd.read_csv(data_dir / 'test.csv')
sample_submission = pd.read_csv(data_dir / 'sampleSubmission.csv')
print(train_df.shape)
train_df.head()
print(test_df.shape)
test_df.head()
# features = ['open', 'high', 'low', 'close', 'volume', 'trades', 'macd',
#             'macd_hist', 'macd_signal', 'adx', 'di_plus', 'di_minus',
#             'rsi', 'cci', 'adl']
features = ['di_minus', 'rsi', 'cci']
X_train = train_df[features]
y_train = train_df['y']
X_test = test_df[features]
scaler = StandardScaler(with_std=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                                  max_depth=7)
regressor.fit(X_train, y_train)

y_rfr = regressor.predict(X_test)

print('RandomForestRegressor')
print(y_rfr.min())
print(y_rfr.max())
from catboost import CatBoostRegressor
regressor = CatBoostRegressor(silent=True)

regressor.fit(X_train, y_train)

y_cbr = regressor.predict(X_test)
print('CatBoostRegressor')
print(y_cbr.min())
print(y_cbr.max())
from sklearn.ensemble import AdaBoostRegressor

regressor = AdaBoostRegressor()
regressor.fit(X_train, y_train)

y_abr = regressor.predict(X_test)
print('AdaBoostRegressor')
print(y_abr.min())
print(y_abr.max())
from sklearn.ensemble import BaggingRegressor

regressor = BaggingRegressor()
regressor.fit(X_train, y_train)

y_br = regressor.predict(X_test)

print('BaggingRegressor')
print(y_br.min())
print(y_br.max())
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_lr = regressor.predict(X_test)
print('LinearRegressor')
print(y_lr.min())
print(y_lr.max())
from sklearn.linear_model import BayesianRidge

regressor = BayesianRidge()
regressor.fit(X_train, y_train)

y_bridge = regressor.predict(X_test)

print('BayesianRidge')
print(y_bridge.min())
print(y_bridge.max())
from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor()
regressor.fit(X_train, y_train)

y_gbr = regressor.predict(X_test)
print('GradientBoostingRegressor')
print(y_gbr.min())
print(y_gbr.max())
y_test = (y_gbr + y_bridge + y_lr + y_br + y_abr + y_cbr + y_rfr) / 7

print('Total')
print(y_test.min())
print(y_test.max())
sample_submission['expected'] = y_test
sample_submission.to_csv('submission.csv', index=False)
