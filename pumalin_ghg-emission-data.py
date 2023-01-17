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
# import libraries and load data

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()

df = pd.read_csv('/kaggle/input/co2-and-ghg-emission-data/emission data.csv', delimiter=',')

df.head()
# copy the dataframe

df_copy = df.copy()

# create transposed dataframe to have the years as column indices

df_trans = df_copy.T

#print(df_trans.head())

time = df_trans.index.values[1:]

# convert string array of years to int array of years

time = list(map(int,time))



# unfortunately I cannot find the units in this dataset
# Analyse dataframe with respect to double counted values as EU and single countries

# What about Europe/America -> when analysing single countries EU etc. have to be taken out



countries = df_copy["Country"]

#for i in range(len(countries)):

#    print(countries[i:i+1])



# drop doubled measurements: 

# Africa -> index 1, 

# Americas (others) -> index 4

# Asia and Pacific (other) -> index 13, 

# EU-28 -> index 64, 

# Europe (other) -> index 72

# World -> index 227

df_onlyCountries = df_copy.drop([1,4,13,64,72,227])

df_onlyCountries = df_onlyCountries.reset_index(drop=True)
# First inspection of the data

# sum of the columns (years) gives total number on GHG contribution for each year

y = df_onlyCountries.sum().iloc[:1]

plt.figure(figsize=(20,5))

plt.plot(time, df_onlyCountries.sum().iloc[1:]*10e-9)

plt.ylabel("CO2 and GHG emission in 10^-9")

plt.xlabel("years")

plt.show()
# plot first year (1751)



plt.figure(figsize=(4,3))

plt.plot(df_onlyCountries.iloc[:,1]*10e-9)

plt.ylabel("CO2 and GHG emission in 10^9")

plt.xlabel("years")

plt.ylim(-0.01,0.1)

plt.show()



# print the index with non-zero GHG

for i in range(len(df_onlyCountries.iloc[:,1])):

    if df_onlyCountries.iloc[i,1] != 0:

        print(i)

        print(df_onlyCountries.iloc[i,0])



# Previous plot shows there is only 1 country with non-zero GHG (index 214 - United Kingdon)

# There is probably a data problem at the beginning of the time series



# How is the data availability?

# Plot year vs. number of countries that have non-zero GHG



plt.figure(figsize=(20,5))

plt.stem(time,df_onlyCountries.astype(bool).sum(axis=0).iloc[1:], basefmt="m")

plt.ylabel("Nr of countries with non-zero entry")

plt.show()
# There are several jumps in the data availability

# Divide analysis in 2 and cut in 1950, consider only from 1950 on

# From 1950 on a lot more data are non-zero

# From 1960 on anthropogenic climate change is more visible

eidx = 1950 - 1751 

delidx = np.linspace(1,eidx,eidx)

# convert to integer

delidx = list(map(int, delidx)) 



df_onlyCountries1950 = df_onlyCountries.drop(df.columns[delidx], axis=1)
# Transpose dataframe in order to have time as column

df_onlyCountries1950T = df_onlyCountries1950.T

#print(df_onlyCountries1950T.head())

time1950 = df_onlyCountries1950T.index.values[1:]

# convert string array of years to int array of years

time1950 = list(map(int,time1950))

# list of countries

countries = []

for i in range(df_onlyCountries1950.shape[0]):

    countries.append(df_onlyCountries1950["Country"][i])

# dataframe without country names and years

df_onlyCountries1950T_noNames = df_onlyCountries1950T.drop(["Country"],axis=0) #drop 1. row

#plot sum of the contribution for each country from 1950 on

locs = np.linspace(1,df_onlyCountries1950.shape[0],df_onlyCountries1950.shape[0])

plt.figure(figsize=(40,10))

plt.stem(df_onlyCountries1950.sum(axis=1,numeric_only=True)*10e-9, basefmt="m")

plt.ylabel("CO2 and GHG emission in 10^9")

#locs, labels = plt.xticks() 

plt.xticks(locs, countries, rotation='vertical')

plt.show()



drop_list = []

for i in range(df_onlyCountries1950.shape[0]):

    if(df_onlyCountries1950.sum(axis=1,numeric_only=True)[i]*10e-9<=20000):

        drop_list.append(i)

df_highestSumCountries1950 = df_onlyCountries1950.drop(drop_list)

df_highestSumCountries1950 = df_highestSumCountries1950.reset_index(drop = True)

# plot single countries emissions of the countries with the highest sum



highestSumCountries = []

for i in range(df_highestSumCountries1950.shape[0]):

    highestSumCountries.append(df_highestSumCountries1950["Country"][i])

    

# Transpose dataframe in order to have time as column

df_highestSumCountries1950T = df_highestSumCountries1950.T



plt.figure(figsize=(20,5))

for i in range(len(highestSumCountries)):

     plt.plot(df_onlyCountries1950T.index[1:],df_highestSumCountries1950.iloc[i,1:]*10e-9)

plt.legend(highestSumCountries)

plt.ylabel("CO2 and GHG emission in 10^-9")

locs, labels = plt.xticks() 

plt.xticks(locs,time1950, rotation='vertical')

plt.show()

# Calculate trend for each country

from scipy import stats



xi = np.arange(len(df_onlyCountries1950T)-1)

xi = xi.reshape(-1, 1)



df_onlyCountries1950N  = df_onlyCountries1950.set_index('Country', drop=True)

df_onlyCountries1950NT = df_onlyCountries1950.T

from sklearn import linear_model



linearTrend = []



reg = linear_model.LinearRegression()

for i in range(len(df_onlyCountries1950)):

    reg.fit(xi, df_onlyCountries1950NT.iloc[1:,i])

    linearTrend.append(float(reg.coef_))

df_onlyCountries1950["linear trend"] = linearTrend

df_onlyCountries1950.head()

# plot trends as stemplot

locs = np.linspace(1,df_onlyCountries1950NT.shape[1],df_onlyCountries1950NT.shape[1])

plt.figure(figsize=(40,10))

plt.stem(df_onlyCountries1950.iloc[1:,-1]*10e-9, basefmt ="m")

plt.ylabel("CO2 and GHG emission linear trend in 10^-9/year")

#locs, labels = plt.xticks() 

plt.xticks(locs, countries, rotation='vertical')

plt.show()

# plot map of trends

import plotly.express as px

fig = px.choropleth(df_onlyCountries1950, color="linear trend",locationmode='country names',locations="Country",

                    hover_name="Country",color_continuous_scale=px.colors.sequential.Bluered,range_color=[0,5000000000],

                    title="CO2 and GHG emission linear trend per year from 1950 - 2017")

fig.show()
