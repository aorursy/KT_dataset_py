# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplolib.pyplot as plt



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt
df = pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")

df.head(10)
df.Year.value_counts()
def avg_temp(year):

    cur_year = df[(df["Year"] == year) & (df["Region"] == "Africa")]

    cur_year_mean = cur_year["AvgTemperature"].mean()

    return cur_year_mean

avg_temp(2020)
columns = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

data = []

for i in columns:

    mean_afr = avg_temp(i)

    data.append(mean_afr)

data
avg_temp_by_years = pd.DataFrame(data)

plt.plot(columns, avg_temp_by_years)

plt.xlabel("years") 

plt.ylabel("Avg Temperature")