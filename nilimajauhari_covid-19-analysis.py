# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the data set
covid = pd.read_csv("/kaggle/input/covid19-all-countries-data/Covid Data for all countries.csv", encoding="iso-8859-1")
# Let us look at the shape of the data
covid.shape
# Visualizing the first few rows of the data set
covid.head(3)
# Let us take a look at the data types first before replacing the missing values
covid.dtypes
# Let us see which all columns have missing values
covid.isnull().sum()
# Replacing missing values in all columns by 0
covid.fillna(0, inplace = True)
covid.isnull().sum()
# Let us take a look at the countries mostly affected by corona
covid.sort_values(by = "Total Deaths", ascending = False).head(5)
covid.sort_values(by = "Total Cases", ascending = False).head(5)
# Total cases in countries

labels = 'USA', 'Brazil', 'Russia', 'India', 'Spain'
sizes = '2890588', '1543341', '674515', '649889', '297625'
plt.rcParams['figure.figsize'] = 8, 8

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow =True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
covid.sort_values(by = "Active Cases", ascending = False).head(5)
# Active cases in countries

labels = 'USA', 'Brazil', 'India', 'Russia', 'Peru'
sizes = '1522999', '501472', '236809', '217609', '99521'
plt.rcParams['figure.figsize'] = 8, 8

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow =True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
covid.sort_values(by = "Total Tests", ascending = False).head(5)
# Total tests in countries

labels = 'China', 'USA', 'Russia', 'UK', 'India'
sizes = '90410000', '36297195', '20451110', '10120276', '9540132'
plt.rcParams['figure.figsize'] = 8, 8

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow =True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
# Let us take a look at the countries who have had no active cases recently
covid.sort_values(by = "Total Recovered", ascending = False).head(5)