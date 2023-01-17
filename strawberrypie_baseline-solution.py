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
features = ['open', 'high', 'low', 'close', 'volume', 'trades', 'macd',
            'macd_hist', 'macd_signal', 'adx', 'di_plus', 'di_minus',
            'rsi', 'cci', 'adl']
X_train = train_df[features]
y_train = train_df['y']
X_test = test_df[features]
scaler = StandardScaler(with_std=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                                  max_depth=7)
regressor.fit(X_train, y_train)
y_test = regressor.predict(X_test)
y_test.min()
sample_submission['expected'] = y_test
sample_submission.to_csv('submission.csv', index=False)
