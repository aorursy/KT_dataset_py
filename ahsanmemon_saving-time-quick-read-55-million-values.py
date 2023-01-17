# In this kernel, we will store the humongous 55 Million readings in Feather format
# This format takes much lesSser time to read (approx 7 seconds)
import os
import pandas as pd
print(os.listdir("../input/"))
%%time
# Reading File
train_path  = '../input/train.csv'

# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())

train_df = pd.read_csv(train_path, usecols=cols, dtype=traintypes)
%%time
# Save into feather format, about 1.5Gb. 
train_df.to_feather('nyc_taxi_data_raw.feather')
%%time
# load the same dataframe next time directly, without reading the csv file again!
train_df = pd.read_feather('nyc_taxi_data_raw.feather')

# It took less than one tenth of time to read the file