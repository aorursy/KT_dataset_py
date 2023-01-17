# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir('../input'))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')
# dataframe from dictionary
country = ["France", "Turkey", "Germany", "Spain"]
population = ["15", "20", "32", "50"]
list_label = ["country", "population"]
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dict=dict(zipped)
df = pd.DataFrame(data_dict)
df
### Add new columns
df["capital"] = ["Paris", "Ankara", "Berlin", "Madrid"]
df
#broadcasting
df["income"]=0#broadcasting entire column
df
data.head()
data.info()
#Plotting all data
data1 = data.loc[:,["total_hospital_beds", "available_hospital_beds", "potentially_available_hospital_beds"]]
data1.plot()

#For another column
data2 = data.loc[:,["percentage_of_potentially_available_icu_beds_needed_six_months", "percentage_of_potentially_available_icu_beds_needed_twelve_months", "percentage_of_potentially_available_icu_beds_needed_eighteen_months"]]
data2.plot()
#Subplots
data1.plot(subplots=True)
plt.show()

data2.plot(subplots=True)
plt.show()
### Scatter plot
data.plot(kind = "scatter", x="available_icu_beds", y = "potentially_available_icu_beds")
plt.show()
#Hist plot (measure the frequency)
data.plot(kind = "hist", y = "potentially_available_icu_beds", bins = 50, range = (0, 250))

#Histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows = 2, ncols = 1)
data.plot(kind = "hist", y = "potentially_available_icu_beds", bins = 50, range = (0, 250),  ax = axes[0])
data.plot(kind = "hist", y = "potentially_available_icu_beds", bins = 50, range = (0, 250),  ax = axes[1], cumulative = True)
plt.savefig('graph.png')
plt

#For another column
fig, axes = plt.subplots(nrows = 2, ncols = 1)
data.plot(kind = "hist", y = "available_icu_beds", bins = 50, range = (0, 250),  ax = axes[0])
data.plot(kind = "hist", y = "available_icu_beds", bins = 50, range = (0, 250),  ax = axes[1], cumulative = True)
plt.savefig('graph.png')
plt


