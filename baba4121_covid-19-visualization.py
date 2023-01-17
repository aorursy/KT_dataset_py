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
# Importing the all required libraries
import pandas as pd
import requests
import smtplib
import urllib.request
from bs4 import BeautifulSoup
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
import folium
#Importing the data
df = pd.read_csv("../input/covid19-dataset/covid.csv")
df
# Data exploring and preprocessing
#At first we will be dropping the Unnamed column
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)],axis=1,inplace=True)
#Searching for missing values
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  
# Adding Mortality rate as one new column in our data base
# We have to replace all NaN with zero for calculation
# We will first change the data type of columns to perform mathematical operations on them as well
df = df.replace(np.nan, 0, regex=True)
df["Total_Deaths"] = df["Total_Deaths"].astype("int64")
df["Total_Recovered"] = df["Total_Recovered"].astype("int64")
df["Mortality_Rate"] = np.round(100*df["Total_Deaths"]/df["Total_Cases"],2)
# Grouping the data by Continets so that we can analyze the cases in each continents
df_continents_cases = df.groupby("Continent").sum()
# Visualising the the data of all the contents with table
df_continents_cases.style.background_gradient(cmap='Blues',subset=["Total_Cases"])\
                        .background_gradient(cmap='Reds',subset=["Total_Deaths"])\
                        .background_gradient(cmap='Greens',subset=["Total_Recovered"])\
                        .background_gradient(cmap='Purples',subset=["Active_Cases"])\
                        .background_gradient(cmap='Pastel1_r',subset=["Critical_Cases"])\
                        .background_gradient(cmap='Oranges',subset=["Cases/Million_Population"])\
                        .background_gradient(cmap='BuPu',subset=["Deaths/Million_Population"])\
                        .background_gradient(cmap='pink',subset=["Total_Tests"])\
                        .background_gradient(cmap='YlOrBr',subset=["Mortality_Rate"])\
                        .format("{:.2f}")\
                        .format("{:.0f}",subset=["Total_Cases","Total_Deaths","Total_Recovered","Active_Cases"])
#Exploring Cases across countries
df_countries_cases = df.drop(['Continent','Cases/Million_Population','Deaths/Million_Population','Tests/Million_Population','Population'],axis=1)
df_countries_cases.index = df_countries_cases["Country"]
df_countries_cases = df_countries_cases.drop(['Country'],axis=1)
df_countries_cases.sort_values('Total_Cases', ascending= False).style.background_gradient(cmap='Blues',subset=["Total_Cases"])\
                        .background_gradient(cmap='Reds',subset=["Total_Deaths"])\
                        .background_gradient(cmap='Greens',subset=["Total_Recovered"])\
                        .background_gradient(cmap='Purples',subset=["Active_Cases"])\
                        .background_gradient(cmap='Pastel1_r',subset=["Critical_Cases"])\
                        .background_gradient(cmap='twilight',subset=["Total_Tests"])\
                        .background_gradient(cmap='YlOrBr',subset=["Mortality_Rate"])\
                        .format("{:.2f}")\
                        .format("{:.0f}",subset=["Total_Cases","Total_Deaths","Total_Recovered","Active_Cases"])
# Top 15 countries with most number of cases
f = plt.figure(figsize=(14,8))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Total_Cases')["Total_Cases"].index[-15:],df_countries_cases.sort_values('Total_Cases')["Total_Cases"].values[-15:],color="Red")
plt.tick_params(size=5,labelsize = 18)
plt.xlabel("Total Number Of Cases",fontsize=18)
plt.title("Top 15 Countries with Most Number of Cases",fontsize=20)
plt.grid(alpha=0.3)
# Top 15 Countries with most number of deaths
f = plt.figure(figsize=(14,8))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Total_Deaths')["Total_Deaths"].index[-15:],df_countries_cases.sort_values('Total_Deaths')["Total_Deaths"].values[-15:],color="Green")
plt.tick_params(size=5,labelsize = 18)
plt.xlabel("Total Number Of Deaths",fontsize=18)
plt.title("Top 15 Countries with Most Deaths",fontsize=20)
plt.grid(alpha=0.3,which='both')
# Most Number of active cases
f = plt.figure(figsize=(14,8))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Active_Cases')["Active_Cases"].index[-15:],df_countries_cases.sort_values('Active_Cases')["Active_Cases"].values[-15:],color="darkorange")
plt.tick_params(size=5,labelsize = 18)
plt.xlabel("Active Cases",fontsize=18)
plt.title("Top 15 Countries with Active Cases",fontsize=20)
plt.grid(alpha=0.3,which='both')
# Recovered Cases
f = plt.figure(figsize=(14,8))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Total_Recovered')["Total_Recovered"].index[-15:],df_countries_cases.sort_values('Total_Recovered')["Total_Recovered"].values[-15:],color="grey")
plt.tick_params(size=5,labelsize = 18)
plt.xlabel("Recovered Cases",fontsize=18)
plt.title("Top 15 countries with most number of Recovered Cases",fontsize=20)
plt.grid(alpha=0.3,which='both')