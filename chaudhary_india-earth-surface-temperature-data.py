import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.tools import plotting



five_thirty_eight = ["#30a2da", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b"]



sns.set_palette(five_thirty_eight)

sns.palplot(sns.color_palette())

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import functools

reader = functools.partial(pd.read_csv, parse_dates='dt'.split())

country = reader('../input/GlobalLandTemperaturesByCountry.csv')

city = reader('../input/GlobalLandTemperaturesByCity.csv')

major_city = reader('../input/GlobalLandTemperaturesByMajorCity.csv')
print(city.head())

print(major_city.head())

print(country.head())

print(country.info())

print(country.describe())
india = country[country.Country == 'India']

india.set_index(['dt'], inplace=True)

india.head()
india_avg = india.AverageTemperature
india.plot(figsize=(9, 6))
india_avg.hist(bins=100, figsize=(9, 6))
try:

    plotting.bootstrap_plot(india_avg)

except ValueError:

    pass
plotting.autocorrelation_plot(india_avg)
plotting.lag_plot(india_avg)
plotting.scatter_matrix(india, alpha=0.8, figsize=(9, 6), diagonal='kde')