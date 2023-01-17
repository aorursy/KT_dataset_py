import numpy as np   

import pandas as pd 

from fbprophet import Prophet

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/sp500-daily-19862018/spx.csv", parse_dates=['date'])
df.info()
df.head()
df.columns = ['ds', 'y']

train_df, test_df = train_test_split(df, train_size=0.95, shuffle=False)
print(train_df.shape, test_df.shape)

train_df.head()
model = Prophet()



# Pass historical data to fit the model

model.fit(train_df)

# Obtain forecasts for dates in future

forecasts = model.predict(test_df[['ds']])
forecasts.tail()
forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
### Plotting the forecast

fig1 = model.plot(forecasts)
# Plotting forecast components



fig2 = model.plot_components(forecasts)

test_df['forecast'] = forecasts['yhat'].values
test_df.set_index('ds')[['y', 'forecast']].plot(figsize=(14,6))

plt.show()
forecasts.set_index('ds')[['yhat_lower','yhat','yhat_upper']].plot(figsize=(14,6))

plt.show()