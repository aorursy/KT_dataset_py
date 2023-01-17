# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import register_matplotlib_converters
Temperature = pd.read_excel("/kaggle/input/Temperature_measurement.xlsx")

Light = pd.read_excel("/kaggle/input/Light_measurement.xlsx")
print(Light.head())

print(Light.tail())

print("Length Light dataset: {}".format(len(Light)))

print("\n")

#---------------------------------------------------

print(Temperature.head())

print(Temperature.tail())



print("Length Temperature dataset: {}".format(len(Temperature)))
print("Data type Light : {}".format(type(Light['Light'][0])))

print("Data type Time in Light-dataset : {}".format(type(Light['Time'][0])))

print("Data type Temperature : {}".format(type(Temperature['Temperature'][0])))

print("Data type Time in Temperature-dataset : {}".format(type(Temperature['Time'][0])))
sns.lineplot(x="Time",y="Light", data = Light)

sns.lineplot(y="Temperature", x="Time", data = Temperature)

plt.show()
plt.plot(Temperature['Temperature'])

plt.plot(Light['Light'])

plt.show()
def normalize_time(series):

    series = pd.to_datetime(series, format="%H:%M:%S")

    series += pd.to_timedelta(series.lt(series.shift()).cumsum(), unit="D")

    return series
Light["Time"] = normalize_time(Light["Time"])

Temperature["Time"] = normalize_time(Temperature["Time"])
Light["Time"]
import matplotlib.dates



ax = plt.subplot()



sns.lineplot(x="Time", y="Light", data=Light)

sns.lineplot(x="Time", y="Temperature", data=Temperature)



ax.xaxis.set_major_formatter(

    matplotlib.dates.DateFormatter("%H:%M")

)



plt.show()