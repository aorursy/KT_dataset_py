# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
flights = pd.read_csv('../input/flight-delay/final_data.csv')


flights.info()
flights_sub = flights[['FL_DATE','YEAR', 'QUARTER', 'MONTH', 'UNIQUE_CARRIER', 'TAIL_NUM','FL_NUM', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID','CRS_DEP_TIME','DEP_DELAY','DEP_TIME','CRS_ARR_TIME','ARR_DELAY','ARR_TIME']]
flights_sub.info()
flights_sub.head()
print(flights_sub[['DEP_DELAY', 'ARR_DELAY']].max())
print(flights_sub[['DEP_DELAY', 'ARR_DELAY']].min())
print(flights_sub.shape)
flights_sub = flights_sub.drop_duplicates(subset = ['FL_DATE','UNIQUE_CARRIER', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])
print(flights_sub.shape)
print(flights_sub[['DEP_DELAY', 'ARR_DELAY']].agg([np.mean, np.median, np.max, np.min]))
print(flights_sub['UNIQUE_CARRIER'].value_counts())
print(flights_sub['UNIQUE_CARRIER'].value_counts(normalize = True)*100)
flights_sub.head()
flights_sub.groupby('UNIQUE_CARRIER')[['DEP_DELAY', 'ARR_DELAY']].agg([np.mean, np.median])
flights_sub.pivot_table(values = 'DEP_DELAY', index = 'FL_DATE', columns = 'UNIQUE_CARRIER', aggfunc = np.mean, fill_value = 0)
flights_sub.pivot_table(values = 'ARR_DELAY', index = 'FL_DATE', columns = 'UNIQUE_CARRIER', aggfunc = np.mean, fill_value = 0)
flights_sub.plot(x = 'QUARTER', y ='DEP_DELAY', kind = 'line')
plt.show()
flights_sub['DELAYED'] = flights_sub['DEP_DELAY'] > 30

print(flights_sub['DELAYED'].value_counts(normalize = True))
flights_sub.head()
flights_sub[flights_sub['DELAYED']]['DEP_DELAY'].hist()

flights_sub[(flights_sub['DELAYED']) & (flights_sub['DEP_DELAY'] > 100)]['DEP_DELAY'].hist(bins = 3)
flights_sub[(flights_sub['DELAYED']) & (flights_sub['DEP_DELAY'] > 500)]['DEP_DELAY'].hist(bins = 3)
flights_sub[(flights_sub['DELAYED']) & (flights_sub['DEP_DELAY'] > 500)]['DEP_DELAY'].hist()
flights_sub_outlier = flights_sub[flights_sub['DEP_DELAY'] > 500]
flights_sub_outlier
flights_sub = flights_sub[flights_sub['DEP_DELAY'] < 1000]
flights_sub.head()
flights_sub[flights_sub['DEP_DELAY']< 0]['DEP_DELAY'] = 0
flights_sub['DEP_DELAY'].hist(bins= 3)