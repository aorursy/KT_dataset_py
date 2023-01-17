import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))
df = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv") #read file



#create new data

df['dt']=pd.to_datetime(df['dt'], format='%Y%m%d', errors='ignore')

df['Year']=pd.DatetimeIndex(df['dt']).year

df['Month']=pd.DatetimeIndex(df['dt']).month

df['Day']=pd.DatetimeIndex(df['dt']).day



#visulation to all yer

plt.figure(figsize = (20,5))

sns.lineplot(df["Year"],df["AverageTemperature"])

plt.title("All Year")

plt.show()



#visulation to before 1820 because before 1820 missing data end determine in anomaly templature 

before_1820 = df[df["Year"]<1820]

plt.figure(figsize = (20,5))

plt.title("Before 1820")

sns.lineplot(before_1820["Year"],before_1820["AverageTemperature"],color = "green")

plt.show()



#visulation to after 1850 

#the data after 1850 it seems fresh

after_1850 = df[df["Year"]>1849]

plt.figure(figsize = (20,5))

plt.title("After 1850")

sns.lineplot(after_1850["Year"],after_1850["AverageTemperature"],color = "red")

plt.show()