import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet
# Load the data

train = pd.read_csv("../input/train_1.csv")

keys = pd.read_csv("../input/key_1.csv")

ss = pd.read_csv("../input/sample_submission_1.csv")

train.head()
# Drop Page column

X_train = train.drop(['Page'], axis=1)

X_train.head()
# Check the data

X_train.isnull().any().describe()
y = X_train.as_matrix()[0]

df = pd.DataFrame({ 'ds': X_train.T.index.values, 'y': y})
# With outliers

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=10)

forecast = m.predict(future)

m.plot(forecast);
# Remove outliers

y = X_train.dropna(0).as_matrix()[0] # Replace NaN to 0 for list comprehension

y = [ None if i >= np.percentile(y, 95) or i <= np.percentile(y, 5) else i for i in y ]

df_na = pd.DataFrame({ 'ds': X_train.T.index.values, 'y': y})
# Fit the modal

m = Prophet()

m.fit(df_na)
# Show future dates

future = m.make_future_dataframe(periods=10)

future.tail()

# Forecast future data

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Plot forecaset

m.plot(forecast);
