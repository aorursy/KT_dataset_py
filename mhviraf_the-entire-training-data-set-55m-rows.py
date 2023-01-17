#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline 
import matplotlib.pyplot as plt
types = {'fare_amount': 'float32',
         'pickup_longitude': 'float64',
         'pickup_latitude': 'float64',
         'dropoff_longitude': 'float64',
         'dropoff_latitude': 'float64',
         'passenger_count': 'uint8'}
cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
train_data = pd.read_csv('../input/train.csv', dtype=types, usecols=cols, infer_datetime_format=True, parse_dates=["pickup_datetime"]) # total nrows = 55423855
#test_data = pd.read_csv('../input/test.csv', nrows=0)
train_data.info()
train_data.describe()
train_data.isnull().sum()
counts = train_data[train_data.passenger_count<6].passenger_count.value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel('No. of passengers')
plt.ylabel('Frequency')
plt.xticks(range(0,7))
print(counts)
# to capture 75% of the training dataset
train_data[(train_data.fare_amount<125) & (train_data.fare_amount>0)].fare_amount.hist(bins=175, figsize=(15,4))
plt.xlabel('fare $USD')
plt.ylabel('Frequency')
plt.xlim(xmin=0);