import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns
sphist = pd.read_csv('../input/sp500-index-data/sphist.csv')
print(sphist.describe())
print("\ndf shape: ", sphist.shape)
sphist.head()
# Convert 'Date' column to Pandas date type
sphist['Date'] = pd.to_datetime(sphist['Date'])

# Sort df by that column
sphist.sort_values(by=['Date'], inplace=True)
sphist.head()
# Add new indicators to each observation:
# 1 
sphist['avg_price_5'] = sphist['Close'].rolling(5).mean()
sphist['avg_price_5'] = sphist['avg_price_5'].shift() # Avoid using current day's price by reindexing

# 2
sphist['avg_price_365'] = sphist['Close'].rolling(365).mean()
sphist['avg_price_365'] = sphist['avg_price_365'].shift() # Avoid using current day's price by reindexing

# 3
sphist['avg_price_5_365'] = sphist['avg_price_5'] / sphist['avg_price_365']

# 4
sphist['std_price_5'] = sphist['Close'].rolling(5).std()
sphist['std_price_5'] = sphist['std_price_5'].shift() # Avoid using current day's price by reindexing

# 5
sphist['std_price_365'] = sphist['Close'].rolling(365).std()
sphist['std_price_365'] = sphist['std_price_365'].shift() # Avoid using current day's price by reindexing

# 6
sphist['std_price_5_365'] = sphist['std_price_5'] / sphist['std_price_365']

# 7 
sphist['avg_volume_5'] = sphist['Volume'].rolling(5).mean()
sphist['avg_volume_5'] = sphist['avg_volume_5'].shift() # Avoid using current day's price by reindexing

# 8
sphist['avg_volume_365'] = sphist['Volume'].rolling(365).mean()
sphist['avg_volume_365'] = sphist['avg_volume_365'].shift() # Avoid using current day's price by reindexing

# 9
sphist['avg_volume_5_365'] = sphist['avg_volume_5'] / sphist['avg_volume_365']

# 10
min_last_year = sphist['Close'].rolling(365).min()
sphist['last_min_current_ratio'] = min_last_year / sphist['Close']
sphist['last_min_current_ratio'] = sphist['last_min_current_ratio'].shift()
print("# of observations before: ", sphist.shape[0])
print("NaN values before: \n\n", sphist.isnull().sum())

sphist = sphist[sphist['Date'] > datetime(year=1951, month=1, day=2)]
sphist.dropna(axis=0, inplace=True)

print("\n# of observations after: ", sphist.shape[0])
print("NaN values after: \n\n", sphist.isnull().sum())
train = sphist[sphist["Date"] < datetime(year=2013, month=1, day=1)]
test = sphist[sphist["Date"] >= datetime(year=2013, month=1, day=1)]

print("Train: ", train.shape)
print("Test: ", test.shape)
# Sorted correlations with target column 'Close'
sorted_corrs = sphist.corr()['Close'].sort_values()

print(sorted_corrs)
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(sphist[sorted_corrs.index].corr())
features = ['avg_price_5', 'avg_price_365', 'avg_price_5_365', 'std_price_5', 
            'std_price_365', 'std_price_5_365', 'avg_volume_5', 'avg_volume_365', 
            'avg_volume_5_365', 'last_min_current_ratio']

X_train = train[features]
y_train = train['Close']

X_test = test[features]
y_test = test['Close']

# Train
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
closing_price_pred_lr = lr.predict(X_test)

# --------------------------------------------------
# Performance metrics
# --------------------------------------------------

# Calculate MSE
mse_lr = mean_squared_error(y_test, closing_price_pred_lr)

# Calculate the absolute errors and MAPE
errors_lr = abs(closing_price_pred_lr - y_test)
mape_lr = 100 * (errors_lr / y_test)

# MAE
mae_lr = round(np.mean(errors_lr), 2)

# Accuracy
accuracy_lr = 100 - np.mean(mape_lr)

print("-----------------\nLinear regression\n-----------------")
print("MSE: ", mse_lr)
print("MAE: ", mae_lr, "degrees")
print('Accuracy:', round(accuracy_lr, 2), '%.')
rf = RandomForestRegressor(n_estimators=150, random_state=1, min_samples_leaf=2)

# Train 
rf.fit(X_train, y_train)

# Predict
closing_price_pred_rf = rf.predict(X_test)

# --------------------------------------------------
# Performance metrics
# --------------------------------------------------

# Calculate the absolute errors and MAPE
errors_rf = abs(closing_price_pred_rf - y_test)
mape_rf = 100 * (errors_rf / y_test)

# MAE
mae_rf = round(np.mean(errors_rf), 2)

# Accuracy
accuracy_rf = 100 - np.mean(mape_rf)

print("-----------------\nRandom Forest\n-----------------")
print("MAE: ", mae_rf, "degrees")
print('Accuracy:', round(accuracy_rf, 2), '%.')
