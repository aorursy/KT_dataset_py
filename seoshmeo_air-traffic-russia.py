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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 
df = pd.read_csv("/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv")
df.head(10)
df.rename(columns={'Airport name': 'airport', 'Year': 'year', "Whole year" : "whole_year"}, inplace=True)

df.airport.value_counts()
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
def by_years(year):
    year_of_traffic = df[df["year"] == year]
    mean_traffic = year_of_traffic["whole_year"].mean()
    return mean_traffic
data = []
for i in years:
    traf_by_years = by_years(i)
    data.append(traf_by_years)

plt.plot(years, data)
most_traffic_airport = df[df["whole_year"] == df["whole_year"].max()]
most_traffic_airport 