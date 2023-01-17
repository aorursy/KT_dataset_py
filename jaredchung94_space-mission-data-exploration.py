import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data graphing

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import datetime 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Reading in data file

ds = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv")
ds.head()
# Checking number of rows and cols in the data file

ds.shape
# Label header of column

ds.columns
# Drop one of the columns 

ds = ds.drop(columns = ["Unnamed: 0"])



# Giving better column name

ds = ds.rename(columns={"Unnamed: 0.1" : "ID"})
# Checking Null values

ds.isna().sum()
# Changing data type to datetime

ds.Datum = ds.Datum.apply(pd.to_datetime)
# Create a new Column Date to seprate date and time

ds['Date'] = [d.date() for d in ds['Datum']]
# Extracting year and month from 'Date' to new columns

ds['Year'] = pd.DatetimeIndex(ds['Date']).year

ds['Month'] = pd.DatetimeIndex(ds['Date']).month
# Checking data consisting new columns

ds.head()
# Getting the Country

df = ds["Location"].apply(lambda x: x.split(","))

ds["country"] = df.apply(lambda x: x[-1].split()[0])
ds.head()
# Gathering the data by Country

Country_list = ds.groupby("country")["ID"].count().sort_values()
Country_list.plot.barh()
# Getting the top 2

ds.groupby("country")["ID"].count().nlargest(2)
# Top-10 Companies with space missions from 1957

ds.groupby("Company Name")["ID"].count().nlargest(10).sort_values().plot.barh()
Name = ds.groupby("Company Name")["ID"].count().idxmax()

value = ds.groupby("Company Name")["ID"].count().max()

print(Name, 'has the highest space missions of', value, 'times')
# Numbers of space missions by year

ds.groupby("Year")["ID"].count().plot()
ds_filtered = ds.query('Year > 2010')

ds_filtered.groupby("Company Name")["ID"].count().nlargest(10).sort_values().plot.barh()
ds_filtered.groupby("country")["ID"].count().nlargest(10).sort_values().plot.barh()
ds_filtered.groupby("country")["ID"].count().nlargest(10)
ds.groupby("Status Mission")["ID"].count().plot.pie(autopct='%1.2f%%')
ds_filtered.groupby("Status Mission")["ID"].count().plot.pie(autopct='%1.2f%%')
ds.groupby("Status Rocket")["ID"].count().plot.pie(autopct='%1.2f%%')