# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

tempData = pd.read_csv('../input/historydata/history_data.csv')

to_drop = ['Location', 'Resolved Address', 'Minimum Temperature', 'Maximum Temperature']
tempData.drop(to_drop, inplace=True, axis=1)
tempData = tempData.iloc[:,0: 3]

temp = tempData['Temperature'][:-1].to_numpy()
date = tempData['Date time'][:-1].to_numpy()
plt.plot(date, temp)


covidData = pd.read_csv('../input/covied19confirmedcases/time_series_covid19_confirmed_global.csv')
# usData = covidData.loc[80]
# usData
covidData = covidData.loc[covidData['Country/Region'] == "US"]
to_drop = ['Province/State', 'Lat', 'Long', 'Country/Region']
covidData.drop(to_drop, inplace=True, axis=1)
cleanedData = covidData.iloc[:, 39:].to_numpy()

# plt.plot(cleanedData.columns, list(cleanedData.iloc[0]))
# print(list(cleanedData.iloc[0]))
# plt.show()
# cleanedData.columns.to_numpy()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(cleanedData, list(map(lambda x: x/15000, list(cleanedData))), s=10, c='b', marker=".", label='COVID-19 Cases')
ax1.scatter(cleanedData, temp, s=10, c='r', marker="o", label='Temperature')
plt.legend(loc='upper left');
plt.show()

#np.corrcoef(cleanedData.iloc[0], temp)
final = [temp, cleanedData.iloc[0]]
np.corrcoef(final)
model = LinearRegression()
model.fit(temp, cleanedData)
