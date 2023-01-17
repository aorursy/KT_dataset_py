import numpy as np # linear algebra

import pandas as pd # 

from pandas import read_csv

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# load the dataset and print the first 5 rows

series = read_csv('/kaggle/input/perth-temperatures-and-rainfall/PerthTemperatures.csv', header=0)

print(series.head())
#Combining year month and day

series['Date'] = pd.to_datetime(series[['Year','Month','Day']])
#Setting date as Index

series.set_index('Date', inplace=True)
series=series.drop(['Year','Month','Day'], axis=1)
series.resample('2D').mean()
series.resample('2D').sum()
series.resample('2M').mean()
series.resample('5Y').mean()
# Linear upsampling 

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='linear')['Maximum temperature (Degree C)'][0:3],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='linear')['Maximum temperature (Degree C)'][0:3])
plt.plot(series.resample(rule='0.5D').mean().interpolate(method='linear')['Maximum temperature (Degree C)'][0:40],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='linear')['Maximum temperature (Degree C)'][0:40])
# Quadratic

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='quadratic')['Maximum temperature (Degree C)'][0:3],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='quadratic')['Maximum temperature (Degree C)'][0:3])
# Quadratic

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='quadratic')['Maximum temperature (Degree C)'][0:40],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='quadratic')['Maximum temperature (Degree C)'][0:40])
# Nearest interpolations

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='nearest')['Maximum temperature (Degree C)'][0:3],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='nearest')['Maximum temperature (Degree C)'][0:3])
plt.plot(series.resample(rule='0.25D').mean().interpolate(method='nearest')['Maximum temperature (Degree C)'][0:40],'ro')

plt.plot(series.resample(rule='0.25D').mean().interpolate(method='nearest')['Maximum temperature (Degree C)'][0:40])
#slinear

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='slinear')['Maximum temperature (Degree C)'][0:3],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='slinear')['Maximum temperature (Degree C)'][0:3])
plt.plot(series.resample(rule='0.5D').mean().interpolate(method='slinear')['Maximum temperature (Degree C)'][0:40],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='slinear')['Maximum temperature (Degree C)'][0:40])
#spline with order 2

#plt.plot(series.resample(rule='0.5D').mean().interpolate(method='spline',order=2)['Maximum temperature (Degree C)'][0:3],'ro')

#plt.plot(series.resample(rule='0.5D').mean().interpolate(method='spline',order=2)['Maximum temperature (Degree C)'][0:3])
#plt.plot(series.resample(rule='0.5D').mean().interpolate(method='spline',order=2)['Maximum temperature (Degree C)'][0:40],'ro')

#plt.plot(series.resample(rule='0.5D').mean().interpolate(method='spline',order=2)['Maximum temperature (Degree C)'][0:40])
# Polynomial with order 2

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='polynomial',order=2)['Maximum temperature (Degree C)'][0:3],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='polynomial',order=2)['Maximum temperature (Degree C)'][0:3])
plt.plot(series.resample(rule='0.5D').mean().interpolate(method='polynomial',order=2)['Maximum temperature (Degree C)'][0:40],'ro')

plt.plot(series.resample(rule='0.5D').mean().interpolate(method='polynomial',order=2)['Maximum temperature (Degree C)'][0:40])
#krogh

#plt.plot(series.resample(rule='0.5D').mean().interpolate(method='krogh',order=2)['Maximum temperature (Degree C)'][0:3],'ro')

#plt.plot(series.resample(rule='0.5D').mean().interpolate(method='krogh',order=2)['Maximum temperature (Degree C)'][0:3])