import numpy as np
import pandas as pd
# pip install seaborn 
import seaborn as sns
import matplotlib.pyplot as plt
# First, look at everything.
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
dtype = {'DayOfWeek': np.uint8, 'DayofMonth': np.uint8, 'Month': np.uint8 , 'Cancelled': np.uint8, 
         'Year': np.uint16, 'FlightNum': np.uint16 , 'Distance': np.uint16, 
         'UniqueCarrier': str, 'CancellationCode': str, 'Origin': str, 'Dest': str,
         'ArrDelay': np.float16, 'DepDelay': np.float16, 'CarrierDelay': np.float16,
         'WeatherDelay': np.float16, 'NASDelay': np.float16, 'SecurityDelay': np.float16,
         'LateAircraftDelay': np.float16, 'DepTime': np.float16}
# change the path if needed
path = '../input/2008.csv'
flights_df = pd.read_csv(path, usecols=dtype.keys(), dtype=dtype)
print(flights_df.shape)
print(flights_df.columns)
flights_df.head()
flights_df.head().T
flights_df.info()
flights_df.describe().T
flights_df['UniqueCarrier'].nunique()
flights_df.groupby('UniqueCarrier').size().plot(kind='bar');
flights_df.groupby(['UniqueCarrier','FlightNum'])['Distance'].sum().sort_values(ascending=False).iloc[:3]
flights_df.groupby(['UniqueCarrier','FlightNum'])\
  .agg({'Distance': [np.mean, np.sum, 'count'],
        'Cancelled': np.sum})\
  .sort_values(('Distance', 'sum'), ascending=False)\
  .iloc[0:3]
pd.crosstab(flights_df.Month, flights_df.DayOfWeek)
plt.imshow(pd.crosstab(flights_df.Month, flights_df.DayOfWeek),
           cmap='seismic', interpolation='none');
flights_df.hist('Distance', bins=20);
flights_df['Date'] = pd.to_datetime(flights_df.rename(columns={'DayofMonth': 'Day'})[['Year', 'Month', 'Day']])
num_flights_by_date = flights_df.groupby('Date').size()
num_flights_by_date.plot();
num_flights_by_date.rolling(window=7).mean().plot();
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here


