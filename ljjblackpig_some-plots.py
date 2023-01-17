import pandas as pd

import numpy as np

import geopandas as gpd

import shapefile as shp

import matplotlib.pyplot as plt

import seaborn as sns

from shapely.geometry import Point, Polygon
pd.__version__
covid_train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

covid_test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
covid_train.head()
covid_train.info()
sorted_df = covid_train.groupby("Country/Region").agg("sum").sort_values(by=['Fatalities'], ascending=False).reset_index()
def div(df_row):

    if df_row.ConfirmedCases == 0:

        return 0

    else:

        return df_row.Fatalities / df_row.ConfirmedCases
covid_cnt_by_country = sorted_df[["Country/Region", "ConfirmedCases", "Fatalities"]]

covid_cnt_by_country['DeathRatio'] = covid_cnt_by_country.apply(lambda x: div(x), axis = 1)
covid_cnt_by_country.head(20)
crs = {'init': 'epsg:4326'}

geometry = [Point(xy) for xy in zip(covid_train['Long'], covid_train['Lat'])]

geo_df = gpd.GeoDataFrame(covid_train, crs = crs, geometry = geometry)
fig, ax = plt.subplots(figsize = (25, 20))

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world.plot(ax = ax, color = "black")

geo_df.plot(ax = ax, markersize = 5, color = "yellow", marker = "o")
fig, ax = plt.subplots(1, 2, figsize = (25, 20))

fig1 = sns.barplot(covid_cnt_by_country["Country/Region"][:10], np.log(covid_cnt_by_country["Fatalities"][:10]), 

                   ax = ax[0], palette=sns.color_palette("GnBu_d"))

fig2 = sns.barplot(covid_cnt_by_country["Country/Region"][:10], covid_cnt_by_country["DeathRatio"][:10], 

                   ax = ax[1], palette=sns.color_palette("GnBu_d"))

fig1.set_xticklabels(fig1.get_xticklabels(), rotation=45)

fig2.set_xticklabels(fig2.get_xticklabels(), rotation=45)

fig1.set(ylabel = "Log Fatalities")

fig1.set_title("Top 10 Log Fatality by Countries", size = 20)

fig2.set_title("Death Ratio by Countries", size = 20)