# Importing the libraries.

import pandas as pd

#math operations

import numpy as np

#data visualization

import matplotlib.pyplot as plt

dataset = pd.read_csv("../input/earthquake/earthquake_database.csv")

dataset = dataset[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# print(dataset.head())

# print(dataset.count)

# Converting the Date and Time to Unix Time : 



import datetime

import time



timestamp = []

for d, t in zip(dataset['Date'], dataset['Time']):

    try:

        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')

        timestamp.append(time.mktime(ts.timetuple()))

    except ValueError:

        # print('ValueError')

        timestamp.append('ValueError')

timeStamp = pd.Series(timestamp)

dataset['Timestamp'] = timeStamp.values

final_data = dataset.drop(['Date', 'Time'], axis=1)

final_data = final_data[final_data.Timestamp != 'ValueError']

print(final_data)



# s = final_data.iloc[:,0]

# print(s)
# Converting the Date and Time to Unix Time : 



import datetime

import time



timestamp = []

for d, t in zip(dataset['Date'], dataset['Time']):

    try:

        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')

        timestamp.append(time.mktime(ts.timetuple()))

    except ValueError:

        # print('ValueError')

        timestamp.append('ValueError')

timeStamp = pd.Series(timestamp)

dataset['Timestamp'] = timeStamp.values

final_data = dataset.drop(['Date', 'Time'], axis=1)

final_data = final_data[final_data.Timestamp != 'ValueError']

# final_data.head()

final_data.min()