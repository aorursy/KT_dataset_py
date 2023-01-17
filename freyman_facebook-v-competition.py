import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data.head()
N = 100

frames = []

for i in range(N):

    place_id = data.loc[i]['place_id']

    frames.append(data[data.place_id == place_id])



dataN = pd.concat(frames)
plt.figure(figsize=(10,10))

plt.scatter(dataN['x'], dataN['y'], s=1, c=dataN['place_id'], linewidths=0)
MINUTES_IN_DAY = 60 * 24

MINUTES_IN_MONTH = 30 * MINUTES_IN_DAY

START = min(dataN['time'])



day_of_week = ((dataN['time'] - START) // MINUTES_IN_DAY) % 7

time_of_day = (dataN['time'] - START) % MINUTES_IN_DAY

month_of_year = ((dataN['time'] - START) // MINUTES_IN_MONTH) % 12



dataN['day_of_week'] = day_of_week

dataN['time_of_day'] = time_of_day

dataN['month_of_year'] = month_of_year
dataN.head()
plt.hist(dataN[dataN['place_id']==8523065625]['day_of_week'])
dataN[dataN['place_id']==8523065625]['day_of_week']