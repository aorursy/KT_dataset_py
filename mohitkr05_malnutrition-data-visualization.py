# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# figure size in inches

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_country = pd.read_csv("/kaggle/input/malnutrition-across-the-globe/country-wise-average.csv")

df_world = pd.read_csv("/kaggle/input/malnutrition-data-unicef/World_Malnutrition_Data.csv")

df_region = pd.read_csv("/kaggle/input/malnutrition-data-unicef/Region_Data.csv")
df_country.head()
# Income classification is a category and can be converted to int

df_country["Income Classification"] =  df_country["Income Classification"].astype("int32")
df_country["Severe Wasting"].describe()
plt.figure(figsize=(16, 8))

sns.set(style="whitegrid")

cols = ["Income Classification","Severe Wasting","Wasting","Overweight","Stunting","Underweight" ]

sns.pairplot(df_country[cols], height = 2.5 )

plt.show();
plt.figure(figsize=(16, 8))

x = df_country.groupby(["Income Classification"])["Severe Wasting"].mean()

sns.set(style="whitegrid")

ax = sns.barplot(x.index, x)

ax.set_title('Severe Wasting')

ax.set_ylabel('% Severe Wasting')

ax.set_xlabel('Income Classification')

plt.xticks(rotation = 90)
#Plotting on the WorldMap using plotly

x = df_country.groupby(["Country"])["Severe Wasting"].mean()

data = dict(type = 'choropleth',

            locations = x.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= x.index,

            z=x,

            colorbar = {'title':'Severe Wasting %', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="Severe Wasting % around the world")

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
plt.figure(figsize=(16, 8))

x = df_country.groupby(["Income Classification"])["Wasting"].mean()

sns.set(style="whitegrid")

ax = sns.barplot(x.index, x)

ax.set_title('Wasting')

ax.set_ylabel('% Wasting')

ax.set_xlabel('Income Classification')

plt.xticks(rotation = 90)
#Plotting on the WorldMap using plotly

x = df_country.groupby(["Country"])["Wasting"].mean()

data = dict(type = 'choropleth',

            locations = x.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= x.index,

            z=x,

            colorbar = {'title':'Wasting %', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="Wasting % around the world")

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
plt.figure(figsize=(16, 8))

x = df_country.groupby(["Income Classification"])["Overweight"].mean()

sns.set(style="whitegrid")

ax = sns.barplot(x.index, x)

ax.set_title('Overweight')

ax.set_ylabel('% Overweight')

ax.set_xlabel('Income Classification')

plt.xticks(rotation = 90)
#Plotting on the WorldMap using plotly

x = df_country.groupby(["Country"])["Overweight"].mean()

data = dict(type = 'choropleth',

            locations = x.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= x.index,

            z=x,

            colorbar = {'title':'Overweight %', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="Overweight % around the world")

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
plt.figure(figsize=(16, 8))

x = df_country.groupby(["Income Classification"])["Stunting"].mean()

sns.set(style="whitegrid")

ax = sns.barplot(x.index, x)

ax.set_title('Stunting')

ax.set_ylabel('% Stunting')

ax.set_xlabel('Income Classification')

plt.xticks(rotation = 90)
x = df_country.groupby(["Country"])["Stunting"].mean()

data = dict(type = 'choropleth',

            locations = x.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= x.index,

            z=x,

            colorbar = {'title':'stunting %', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="stunting % around the world")

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
plt.figure(figsize=(16, 8))

x = df_country.groupby(["Income Classification"])["Underweight"].mean()

sns.set(style="whitegrid")

ax = sns.barplot(x.index, x)

ax.set_title('Underweight')

ax.set_ylabel('% Underweight')

ax.set_xlabel('Income Classification')

plt.xticks(rotation = 90)
x = df_country.groupby(["Country"])["Underweight"].mean()

data = dict(type = 'choropleth',

            locations = x.index,

            locationmode = 'country names',

            colorscale= 'Portland',

            text= x.index,

            z=x,

            colorbar = {'title':'Underweight %', 'len':200,'lenmode':'pixels' })

layout = dict(geo = {'scope':'world'},title="Underweight % around the world")

col_map = go.Figure(data = [data],layout = layout)

col_map.show()
df_world.rename(columns = {"TIME_PERIOD" : "Year", "OBS_VALUE" : "value"}, inplace =True)
df_world_mort = df_world[df_world["Indicator"] == "Under-five mortality rate"]

df_world_stunting = df_world[df_world.Indicator == "Height-for-age <-2 SD (stunting)"]

df_world_underwt = df_world[df_world.Indicator == "Weight-for-age <-2 SD (Underweight)"]
df_world_mort.drop(["Indicator"],axis="columns",inplace = True)
df_world_mort.Year.unique()
plt.figure(figsize=(16, 8))

x = df_world_mort[(df_world_mort["Sex"] == "Total") & (df_world_mort["Year"] == 2018) ].sort_values(by="value", ascending=False).head(20)

sns.set(style="darkgrid")

ax = sns.barplot(x["Country"], x["value"])

ax.set_title('Mortality Rate Childs/1000 Births in 2018')

ax.set_ylabel('Number of deaths')

ax.set_xlabel('Country')

plt.xticks(rotation = 90)
df_world_mort.head()
plt.figure(figsize=(16, 8))

sns.set(style="darkgrid")

cols = ["Total","Male","Female"]

for col in cols:

    x = df_world_mort[df_world_mort["Sex"] == col].groupby("Year")["value"].mean()

    ax = sns.lineplot(x.index, x, label=col)

ax.set_title('World Wide mortality Rate of Infants / 1000 births')

ax.set_ylabel('Number of deaths / 1000 births')

ax.set_xlabel('Year')

ax.legend()

plt.xticks(rotation = 90)
plt.figure(figsize=(16, 8))

x = df_world_stunting[(df_world_stunting["Sex"] == "Total") & (df_world_stunting["Year"] == 2018) ].sort_values(by="value", ascending=False).head(20)

sns.set(style="darkgrid")

ax = sns.barplot(x["Country"], x["value"])

ax.set_title('Height-for-age <-2 SD (stunting) in 2018 in percentage')

ax.set_ylabel('Percentage')

ax.set_xlabel('Country')

plt.xticks(rotation = 90)
plt.figure(figsize=(16, 8))

sns.set(style="darkgrid")

cols = ["Total","Male","Female"]

for col in cols:

    x = df_world_stunting[df_world_stunting["Sex"] == col].groupby("Year")["value"].mean()

    ax = sns.lineplot(x.index, x, label=col)

ax.set_title('World Wide Height-for-age <-2 SD (stunting)')

ax.set_ylabel('Percent')

ax.set_xlabel('Year')

ax.legend()

plt.xticks(rotation = 90)
plt.figure(figsize=(16, 8))

x = df_world_underwt[(df_world_underwt["Sex"] == "Total") & (df_world_underwt["Year"] == 2018) ].sort_values(by="value", ascending=False).head(20)

sns.set(style="darkgrid")

ax = sns.barplot(x["Country"], x["value"])

ax.set_title('Weight-for-age <-2 SD (Underweight) in 2018 in percentage')

ax.set_ylabel('Percentage')

ax.set_xlabel('Country')

plt.xticks(rotation = 90)
plt.figure(figsize=(16, 8))

sns.set(style="darkgrid")

cols = ["Total","Male","Female"]

for col in cols:

    x = df_world_underwt[df_world_underwt["Sex"] == col].groupby("Year")["value"].mean()

    ax = sns.lineplot(x.index, x, label=col)

ax.set_title('World Wide Weight-for-age <-2 SD (Underweight)')

ax.set_ylabel('Percent')

ax.set_xlabel('Year')

ax.legend()

plt.xticks(rotation = 90)
df_region.head()
df_region["Indicator"].unique()
df_region["Geographic Area"].unique()
continents = ['Asia','Africa','Australia and New Zealand','South America','North America']

plt.figure(figsize=(16, 8))

sns.set(style="darkgrid")

for col in continents:

    x = df_region[(df_region["Geographic Area"] == col) &  (df_region["Indicator"] == "Height-for-age <-2 SD (stunting)")].groupby("TIME_PERIOD")["OBS_VALUE"].mean()

    ax = sns.lineplot(x.index, x, label=col)

ax.set_title('World Wide Height-for-age <-2 SD (stunting)')

ax.set_ylabel('Percent')

ax.set_xlabel('Year')

ax.legend()

plt.xticks(rotation = 90)
continents = ['Asia','Africa','Australia and New Zealand','South America','North America']

plt.figure(figsize=(16, 8))

sns.set(style="darkgrid")

for col in continents:

    x = df_region[(df_region["Geographic Area"] == col) &  (df_region["Indicator"] == "Weight-for-age <-2 SD (Underweight)")].groupby("TIME_PERIOD")["OBS_VALUE"].mean()

    ax = sns.lineplot(x.index, x, label=col)

ax.set_title('World Wide Weight-for-age <-2 SD (Underweight)')

ax.set_ylabel('Percent')

ax.set_xlabel('Year')

ax.legend()

plt.xticks(rotation = 90)