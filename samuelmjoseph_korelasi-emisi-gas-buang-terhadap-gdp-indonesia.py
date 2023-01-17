# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Indicators.csv')
data.shape
data.head(10)
countries = data.CountryName.unique().tolist()
countries
len(countries)
years = data['Year'].unique().tolist()
len(years)
indicators = data['IndicatorName'].unique().tolist()
len(indicators)
print(min(years)," sampai ",max(years))
mask = data['CountryName'].str.contains("Indonesia")
data_indonesia = data[mask]
data_indonesia.head(10)
len(data_indonesia)
mask2 = data['IndicatorName'].str.contains('CO2 emissions \(metric') 
data_co2_indonesia = data[mask & mask2]
data_co2_indonesia.head()
print(min(data_co2_indonesia['Year'])," sampai ",max(data_co2_indonesia['Year']))
mask3 = data['IndicatorName'].str.contains('GDP per capita \(constant 2005') 
data_gdp_indonesia = data[mask & mask3]
data_gdp_indonesia.head()
print(min(data_gdp_indonesia['Year'])," sampai ",max(data_gdp_indonesia['Year']))
# get the years
years = data_co2_indonesia['Year'].values
# get the values 
co2 = data_co2_indonesia['Value'].values

# create
plt.bar(years,co2)
plt.show()
# switch to a line plot
plt.plot(data_gdp_indonesia['Year'].values, data_gdp_indonesia['Value'].values)

# Label the axes
plt.xlabel('Tahun')
plt.ylabel(data_gdp_indonesia['IndicatorName'].iloc[0])

#label the figure
plt.title('GDP Per Kapita Indonesia')

# to make more honest, start they y axis at 0
plt.axis([1959, 2011,0,47000])

plt.show()
data_gdp_indonesia_norm = data_gdp_indonesia[data_gdp_indonesia['Year'] < 2012]
print(len(data_co2_indonesia))
print(len(data_gdp_indonesia_norm))
fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_title('Emisi gas buang vs. GDP per kapita',fontsize=10)
axis.set_xlabel(data_gdp_indonesia_norm['IndicatorName'].iloc[0],fontsize=10)
axis.set_ylabel(data_co2_indonesia['IndicatorName'].iloc[50],fontsize=9)

X = data_gdp_indonesia_norm['Value']
Y = data_co2_indonesia['Value']

axis.scatter(X, Y)
plt.show()
np.corrcoef(data_gdp_indonesia_norm['Value'],data_co2_indonesia['Value'])
