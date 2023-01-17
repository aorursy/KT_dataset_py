import pandas as pd
from fbprophet import Prophet

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/BeerWineLiquor.csv')
data.head()
data.info()
# Column names for Prophet should be ds and y, so renaming the columns of above dataset
data.columns = ['ds', 'y']
data.head()
# Converting ds column of above dataset to datetime
data['ds'] = pd.to_datetime(data['ds'])
data.head()
data.info()
m = Prophet()
m.fit(data)
# Placeholder to hold future predictions
future = m.make_future_dataframe(periods=24, freq='MS')
len(data)
len(future)
forecast = m.predict(future)
forecast.shape
forecast.head()
forecast.columns
forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].tail(12)
m.plot(forecast);
# To remove "ConversionError: Failed to convert value(s) to axis units: '2014-01-01'" error after running the below code
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
m.plot(forecast)
plt.xlim('2014-01-01', '2020-01-01')
forecast.plot(x='ds', y='yhat', figsize=(8,10))
m.plot_components(forecast);
