import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/daily-min-temperature/daily-min-temperatures.csv')
data.head()
data.info()
data['Date'] = data['Date'].astype('datetime64[ns]')
data['month'] = data['Date'].dt.month

data['day'] = data['Date'].dt.day
lag = pd.DataFrame(data['Temp'])

lag_data = pd.concat([lag.shift(1), lag], axis=1)

lag_data.columns = ['t-1', 't+1']
lag_data.head(2)
org = pd.DataFrame(data['Temp'])

shifted = org.shift(1)

window = shifted.rolling(window=2)

means = window.mean()

rolled_data = pd.concat([means, org], axis=1)

rolled_data.columns = ['mean(t-2,t-1)', 't+1']
rolled_data.head()
temps = data['Temp']

window = temps.expanding()

expand_data = pd.concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)

expand_data.columns = ['min', 'mean', 'max', 't+1']
expand_data.head()