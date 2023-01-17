# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
filename = check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv("../input/" + filename, thousands=",")
df.head()
df.dtypes
df['category'].unique()
df['country_or_area'].unique()
table = pd.pivot_table(df, values='value', index=['country_or_area', 'year'], columns=['category'])

table.head()
table.plot()
gasnames = table.columns.values
def country_plot(nameOfCountry):

    data = table.loc[nameOfCountry]

    plt.plot(data)

    plt.legend(gasnames)

    plt.title(nameOfCountry)
country_plot('Germany') # Germany is reducing gas emission in time!
country_plot('Japan') # For Japan, a sudden decrease near 2010. Why? New laws?
country_plot('United States of America')
def gas_plot(nameOfGas): 

    table.plot( y = nameOfGas) 

    

gas_plot(gasnames[0])
table2 = pd.pivot_table(df, values='value', index=['category', 'year'], columns=['country_or_area'])

table2.head()
countryNames = table2.columns.values
def gas_plot2(nameOfGas):

    data = table2.loc[nameOfGas]

    plt.plot(data)

    plt.legend(countryNames)

    plt.title(nameOfGas)
gas_plot2(gasnames[0])
def gas_countries_plot3(nameOfGas, namesOfCountries):

    data = table2.loc[nameOfGas]

    data.plot( y = namesOfCountries)

    plt.legend(namesOfCountries)

    plt.title(nameOfGas)

gas_countries_plot3(gasnames[0],countryNames[:4])
countryNames
interestingCountries = ['United States of America','European Union','Russian Federation', 'Japan', 

'Germany', 'Canada',  'Italy',  'Sweden', 'Switzerland']



for gas in gasnames:

    gas_countries_plot3(gas,interestingCountries)
gas_countries_plot3(gasnames[0],interestingCountries)
gas_countries_plot3(gasnames[3],interestingCountries)
gas_countries_plot3(gasnames[4],interestingCountries)
gas_countries_plot3(gasnames[5],interestingCountries)
gas_countries_plot3(gasnames[6],interestingCountries)
gas_countries_plot3(gasnames[7],interestingCountries)
gas_countries_plot3(gasnames[8],interestingCountries)
gas_countries_plot3(gasnames[9],interestingCountries)