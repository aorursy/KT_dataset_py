import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os

%matplotlib inline
print(os.listdir("../input"))
# Reading the data

df = pd.read_excel("../input/city_region.xls")

df.head()
df.shape
# Expanding the df for adding data acording to years

df = pd.concat([df]*20, ignore_index=True)

df.head()
# Sorting 

df = df.sort_values("city")

df.head()
# Resetting the index

df.reset_index(inplace=True)

df.head()
# Deleting old index column

df.drop("index", axis=1, inplace=True)

df.head()
# Defining years series

years = pd.Series([i for i in range(2000, 2020)])

years
# Expanding years series for each city

years = pd.concat([years]*82, ignore_index=True)

years.head()
# Adding year values

df["years"] = years

df.head()
# Bringing population data

population = pd.read_excel("../input/population.xlsx")

population
# Adding year 2019, it needs to match with df

population[2019]=0
population = population.sort_values("city")

population.head()
# Resetting the index

population.reset_index(inplace=True)

population.head()
# Deleting old index column

population.drop("index", axis=1, inplace=True)

population.head()
population.head()
# Adding population column to the df

df["pop"] = 0
# Adding populations to the dataframe

m = 0

for a in range(len(population.index)): 

    for b in range(len(population.columns)-1): #I need -1 becouse of city column   

        df.iloc[m, 3] = population.iloc[a, b+1]

        m += 1
df.head()
#Cheacking the results

df.groupby("city").pop.mean().values == population.mean(axis=1).values # Looks like the process is true
audiences = pd.read_excel("../input/cine_audience.xls")

audiences
# Matching new data set with our main data set.

# Creating Turkey column.

audiences["Türkiye / Turkey"] = audiences.sum(axis=1) - audiences["year"]

audiences.head()
#Adding two new rows for years 2018 and 2019

audiences = audiences.append(audiences.iloc[0:2], ignore_index=True)

audiences.tail()
# Setting new rows to zero since we have no data

audiences.iloc[18:20] = 0

audiences.tail()
# No need for year column anymore

audiences.drop("year", axis=1, inplace=True)
# We need to sort by columns to match it with main df

audiences.sort_index(axis=1, inplace=True)
# creating a series from audiences dataframes columns.

cine_audiences=pd.Series()

cine_audiences = cine_audiences.append([audiences.iloc[:, i] for i in range(len(audiences.columns))], ignore_index=True)

cine_audiences.head()
# Assigning it to main df

df["cinema_audiences"] = cine_audiences
# Changing data type to int

df = df.astype(int, errors="ignore")
df.dtypes
# Checking the values

df.groupby("city").cinema_audiences.mean().values == audiences.mean().values
# Reading phd data

phd = pd.read_excel("../input/tr_phd.xls")

phd
phd.drop("years", axis=1, inplace=True)

phd.head()
# Adding turkey column

phd["Türkiye / Turkey"] = phd.sum(axis=1)

phd.head()
phd.sort_index(axis=1, inplace=True)

phd.head()
#defining new serie from all columns of has_phd

has_phd =pd.Series()

has_phd = has_phd.append([phd.iloc[:, i] for i in range(len(phd.columns))], ignore_index=True,)
has_phd
# Adding new column

df["has_phd"] = 0
# Filtering the df in has_phd range

df[((2007 < df["years"]) & (df["years"] < 2019))]
index_1 = df[((2007 < df["years"]) & (df["years"] < 2019))].index
has_phd.index=index_1
has_phd
# Asigning the values

df[(2007 < df["years"]) & (df["years"] < 2019)]["has_phd"] = has_phd
# Use loc to get rid of SettingWithCopyWarning. Python get confused if you try to update the main df or copy of it.

df.loc[((2007 < df["years"]) & (df["years"] < 2019)), "has_phd"] = has_phd
df.head(10)
# Saving the data.

df.to_excel("main_df.xls")
df.dtypes
# Creating new data frame to plot

df_1 = df[(2007 < df["years"]) & (df["years"] < 2018)].copy()
# Some zero values cousing some math problem(divided by zero, infinity problems). I will replace them with 1, hence,

# it will have no impact on data and no couse infinity problems.

df_1.pop == 0
df_1.cinema_audiences == 0
df_1[df_1.cinema_audiences == 0]["cinema_audiences"] = 0
df_1.loc[(df_1.cinema_audiences == 0), "cinema_audiences" ] = 1
df_1[df_1.has_phd == 0] # No zero value
# I will exclude also the Turkey data. It ruins the chart due to high values.

df_1[df_1.city.str.contains("Türkiye")]
df_1.drop(df_1[df_1.city.str.contains("Türkiye")].index, inplace=True)
from __future__ import division

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

from bubbly.bubbly import bubbleplot 
figure = bubbleplot(dataset=df_1, x_column='cinema_audiences', y_column='has_phd',

    bubble_column='city', time_column='years', size_column='pop', color_column='region', 

    x_title="Sinema Seyirci Sayısı / Cinema Audiences", y_title="Doktoralı Sayısı / Number of PhD", 

    title='Türkiye Doktoralı Sayısı ve Sinema Seyircisi Sayısı / Turkey Phd and Cinema Audiences Comparison ',

    x_logscale=True, y_logscale=True, scale_bubble=1, width=1050, height=600)



iplot(figure)
# More populated cities are on the rigth top corner. Lets try to compare our values proportionally. And I will

# add also Turkey data.

# Creating new data frame to plot

df_2 = df[(2007 < df["years"]) & (df["years"] < 2018)].copy()
# Setting up to zero values to one

df_2.loc[(df_2.cinema_audiences == 0), "cinema_audiences" ] = 1
df_2.head()


df_2["cine_aud_pop"] = df_2["cinema_audiences"] / df_2["pop"]
df_2["has_phd_pop"] = df_2["has_phd"] / df_2["pop"]
df_2.head()
figure_1 = bubbleplot(dataset=df_2, x_column="cine_aud_pop", y_column="has_phd_pop",

    bubble_column='city', time_column='years', size_column='pop', color_column='region', 

    x_title="Sinema Seyirci Sayısı(oransal) \ Cinema Audiences(proportional)", 

    y_title="Doktoralı Sayısı(oransal) \ Number of PhD(proportional)", 

    title='Türkiye Doktoralı Sayısı ve Sinema Seyircisi Sayısı(Oransal) / Turkey Phd and Cinema Audiences Comparison(Proportional) ',

    x_logscale=False, y_logscale=False, scale_bubble=3, width=1050, height=600)



iplot(figure_1)