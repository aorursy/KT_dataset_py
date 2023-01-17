#Author Anna Durbanova

#Theme: Life Expectancy

#Date 16.08.2020



import pandas as pd

import os

import glob

import numpy as np

import holoviews as hv

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

hv.extension('bokeh')

%matplotlib inline
!pip install hvplot

import hvplot

from hvplot import hvPlot

import hvplot.pandas
!pip install pingouin

import pingouin as pg

import difflib as dfl

from functools import partial

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data=pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv')



data.info()
data["Income composition of resources"].replace(0.0, np.nan, inplace=True)

data["Schooling"].replace(0.0, np.nan, inplace=True)

data.sort_values(["Country", "Year"], inplace=True)

data
data_2015=(data[data.Year==2015]

    .groupby("Country")

    ["Country", "Life expectancy "]

    .median()

    .sort_values(by="Life expectancy ", ascending=False))



data_2015.reset_index().hvplot.bar(x="Country", y="Life expectancy ", rot=90,width=2000, height=550, title = "Life Expectancy for ALL Countries for 2015")
mask = data.Year==2015

plt.rcParams["figure.figsize"]=(20,20)

sns.heatmap(data[mask].corr(), cmap="BuPu", annot=True).set_title("Correlation Table for all columns, 2015");
pg.corr(data["Life expectancy "], data["Income composition of resources"])
plt.rcParams["figure.figsize"]=(12,8)

sns.regplot(x="Income composition of resources", y="Life expectancy ", data=data).set_title("The Effect of Income Composition of Resources on Life expectancy for all data");
plt.rcParams["figure.figsize"]=(12,8)

g = sns.FacetGrid(data=data, col="Year", col_wrap=3, height=15)

g.map_dataframe(sns.regplot, x="Income composition of resources", y="Life expectancy ");
pg.corr(data["Life expectancy "], data["Schooling"])
sns.regplot(x="Schooling", y="Life expectancy ", data=data).set_title("Schooling has an effect on Life Expectancy, 2000-2015");
sns.FacetGrid(data=data,col="Year",col_wrap=4, height=20)

g.map_dataframe(sns.regplot,x="Schooling", y="Life expectancy ");
pg.corr(data["under-five deaths "], data["infant deaths"])
sns.regplot(data=data, x="infant deaths", y="under-five deaths ").set_title("Correlation between Infant Deaths and Under-Five-Deaths, all years");
fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.regplot(data=data[data.Year == 2010], x="Hepatitis B", y="Life expectancy ", ax=ax1).set_title("Hepatitis B has an effect on Life Expectancy, 2010");

sns.regplot(data=data[data.Year == 2015], x="Hepatitis B", y="Life expectancy ", ax=ax2).set_title("Hepatitis B has an effect on Life Expectancy, 2015");
fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.regplot(data=data[data.Year == 2000], x="Polio", y="Life expectancy ", ax=ax1).set_title("Polio has an effect on Life Expectancy, 2000");

sns.regplot(data=data[data.Year == 2015], x="Polio", y="Life expectancy ", ax=ax2).set_title("Polio has an effect on Life Expectancy, 2015");
sns.regplot(data=data, x="Diphtheria ", y= "Life expectancy ").set_title("Diphtheria has an effect on Life Expectancy, all years");
fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.regplot(data=data[data.Year == 2000], x="Diphtheria ", y="Life expectancy ", ax=ax1).set_title("Diphtheria has an effect on Life Expectancy, 2000");

sns.regplot(data=data[data.Year == 2015], x="Diphtheria ", y="Life expectancy ", ax=ax2).set_title("Diphtheria has an effect on Life Expectancy, 2015");
fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.regplot(data=data[data.Year == 2000], x="Alcohol", y="Life expectancy ", ax=ax1).set_title("Alcohol has an effect on Life Expectancy, 2000");

sns.regplot(data=data[data.Year == 2015], x="Alcohol", y="Life expectancy ", ax=ax2).set_title("Alcohol has an effect on Life Expectancy, 2015");



#sns.FacetGrid(data=data, col="Year", col_wrap=4, height=15)

#g.map_dataframe(sns.regplot, x="Alcohol", y="Life expectancy ")
sns.regplot(data=data[data.Year == 2000], x="Total expenditure", y="Life expectancy ").set_title("Positive effect of expenditure on healthcare on the life expectancy, 2000");

fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.regplot(data=data[data.Year == 2000], x="infant deaths", y="Life expectancy ", ax=ax1).set_title("Infant Mortality has an effect on Life Expectancy, 2000");

sns.regplot(data=data[data.Year == 2015], x="infant deaths", y="Life expectancy ", ax=ax2).set_title("Infant Mortality has an effect on Life Expectancy, 2015");
fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.regplot(data=data[data.Year == 2000], x="Adult Mortality", y="Life expectancy ", ax=ax1).set_title("Adult Mortality has an effect on Life Expectancy, 2000");

sns.regplot(data=data[data.Year == 2015], x="Adult Mortality", y="Life expectancy ", ax=ax2).set_title("Adult Mortality has an effect on Life Expectancy, 2015");
top_life_exp_2000=(data[data.Year==2000]

.groupby("Country")

 ["Country", "Life expectancy ", "Year"]

 .median()

 .sort_values("Life expectancy ", ascending=False)

 .head(10)

)



top_life_exp_2015=(data[data.Year==2015]

.groupby("Country")

 ["Country", "Life expectancy ", "Year"]

 .mean()

 .sort_values("Life expectancy ", ascending=False)

 .head(10)

)

bottom_life_exp_2000=(

    data[data.Year==2000]

    .groupby("Country")

    ["Country", "Life expectancy ", "Year"]

    .median()

    .sort_values("Life expectancy ", ascending=True)

    .head(10)

)



bottom_life_exp_2015=(

    data[data.Year==2015]

    .groupby("Country")

    ["Country", "Life expectancy ", "Year"]

    .mean()

    .sort_values("Life expectancy ", ascending=True)

    .head(10)

)



plot_long_2000 = top_life_exp_2000.hvplot.bar(x="Country", y="Life expectancy ", stacked=True, rot=45, title="Countries with the longest and shortest life expectancy, 2000")

plot_long_2015 = top_life_exp_2015.hvplot.bar(x="Country", y="Life expectancy ", stacked=True, rot=45, title="Countries with the longest and shortest life expectancy, 2015")



plot_short_2000= bottom_life_exp_2000.hvplot.bar(x="Country", y="Life expectancy ", stacked=True, rot=45)

plot_short_2015= bottom_life_exp_2015.hvplot.bar(x="Country", y="Life expectancy ", stacked=True, rot=45)



plot_long_2000*plot_short_2000
plot_long_2015*plot_short_2015

status_2000=(data[data.Year==2000]

.groupby("Status")

 [["Country"]]

 .count()

)



status_2015=(data[data.Year==2015]

             .groupby("Status")

             [["Country"]]

             .count()

            )





f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

status_2000.plot.pie(y="Country", autopct='%1.0f%%', ax=ax1, figsize=(12,7)).set_title("There were 83% of developing countries and 17% of developed countries, 2000 and 2015");

status_2015.plot.pie(y="Country", autopct='%1.0f%%', ax=ax2, figsize=(12,7));
data["Compare Status"]=data.Status == data.groupby("Country").Status.shift()
data
mask=(data["Compare Status"]==False) & (data["Year"]!=2000)

data[mask]