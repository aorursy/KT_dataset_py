import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic visualization tool

import seaborn as sns # advanced visualization tool



import os

print(os.listdir("../input"))

data = pd.read_csv('../input/timesData.csv')
data.info()
city = ["New York", "Tokyo", "Istanbul"]

population = ["19,5", "35", "15"]

list_label =["city", "population"]

list_column = [city, population]

zipped = list(zip(list_label, list_column))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
df["country"] = ["usa", "japan", "turkey"] # adding a new column

df
df["income"] = 0 # Default value assignment

df
data[data['country'] == 'Turkey'].info() # Number of universities at the Turkey. (Index number)
data[500:520] # Printing dataframes between 500-520.
data1 = data.loc[:,["teaching", "research", "citations"]]

data1.plot()

plt.show()
data1.plot(subplots = True, grid = True, figsize = (18,10))

plt.xlabel("Number of Universities")

plt.ylabel("Points", y = 1.7) # Adjusting y axis of y-label

plt.show()
data1.plot(kind = "scatter", x = "teaching", y = "research", alpha = .5, grid = True, figsize = (18,10))

plt.xlabel("Teaching")

plt.ylabel("Research")

plt.title('Teaching - Research at Universities Scatter Plot')

plt.show()
data1.plot( kind = 'hist', y = 'research', bins = 100, range = (0,100), normed = True, figsize = (18,10), grid = True)

plt.title('Frequency of the Research')
fig, axes = plt.subplots(nrows = 2, ncols = 1)

data1.plot(kind = 'hist', y = 'research', bins = 50, range = (0,100), normed = True, ax = axes[0], figsize = (15,5), title = 'Histogram of Research')

data1.plot(kind = 'hist', y = 'research', bins = 50, range = (0,100), normed = True, ax = axes[1], cumulative = True, title = 'Cumulative Graph')

plt.savefig('graph.png')

plt

data.describe() # Basic statistical calculations
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list))

datatime_object = pd.to_datetime(time_list) # Conversion to the time series

print(type(datatime_object))
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"] # Sample dates

datatime_object = pd.to_datetime(date_list)

data2['date'] = datatime_object

data2 = data2.set_index('date') # Setting the index to the dates.

data2
print(data2.loc["1993-03-16"]) # Printing specified date.

print(data2.loc["1992-03-10":"1993-03-16"]) # Printing the specified date range.
data2.resample("A").mean() # Year to year datas.
data2.resample("M").mean() # Month ro month datas. The data is filled by NaN values. Because data2 not includes all months.
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")