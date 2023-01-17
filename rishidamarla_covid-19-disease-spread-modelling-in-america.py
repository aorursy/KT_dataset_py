import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import json

import warnings 

warnings.filterwarnings('ignore')
df = pd.read_json("../input/american-covid19-cases/covid19-USA.txt")
df.head()
# Dropping all unnecessary features.

df = df.drop(columns=['CountryCode', 'Province', 'City', 'CityCode', 'Lat', 'Lon', 'Status'])
df.info()
df.describe()
df.dtypes
# Adding up all the number of cases in a particular day across all states.

df2 = df.groupby("Date", as_index=False).Cases.sum()
df2.head()
# Inserting a new index like column.

df2.insert(0, 'Days Since First Confirmed Case', range(0, 0 + len(df2)))
df2.head()
# Visualizing the data.

plt.figure(figsize=(10,10))

plt.plot(df2["Days Since First Confirmed Case"], df2["Cases"], 'ro')

plt.title("Number of confirmed cases since the day of the first reported case in America")

plt.xlabel("Days Since First Confirmed Case")

plt.ylabel('Confirmed Cases')

plt.show()
# Creating a sigmoid function.

def sigmoid(x, L ,x0, k, b):

    y = L / (1 + np.exp(-k*(x-x0)))+b

    return (y)
x_data = df2["Days Since First Confirmed Case"]

y_data = df2["Cases"]

dates = df2['Date']
# Plotting the model against the real data.

from scipy.optimize import curve_fit

p0 = [max(y_data), np.median(x_data),1,min(y_data)]

plt.figure(figsize=(10, 10))

popt, pcov = curve_fit(sigmoid, x_data, y_data,p0, method='dogbox')

plt.title("Number of confirmed cases in America")

plt.xlabel("Days Since First Confirmed Case")

plt.ylabel('Confirmed Cases')

plt.plot(x_data,y_data)

plt.plot(x_data,sigmoid(x_data,*popt))

plt.legend(['Actual', 'Model'])