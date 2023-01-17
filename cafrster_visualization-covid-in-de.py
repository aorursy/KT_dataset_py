# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

%matplotlib inline

import math

import geopandas as gpd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid = pd.read_csv("/kaggle/input/covid19-tracking-germany/covid_de.csv")

covid["state"] = covid["state"].replace("Baden-Wuerttemberg", "Baden-Württemberg")

covid["state"] = covid["state"].replace("Thueringen", "Thüringen")

covid.head()
# What's the inverval of the data?

print(covid["date"].min())

print(covid["date"].max())



# Create variable for last day with format Day-Month-Year:

first_date = covid["date"].min()

last_date = covid["date"].max()

last_date_f = datetime.strptime(covid["date"].max(), "%Y-%m-%d").strftime("%d-%B-%Y")

second2last_date = (datetime.strptime(covid["date"].max(), "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
covid_by_county = covid

covid_by_county = covid_by_county.drop(columns="state")

covid_by_county = covid_by_county.groupby("county", as_index=False).sum()

covid_by_county.head()
#covid_max_date_summed_up = covid_max_date.groupby("state", as_index=False).sum()

# covid_max_date_summed_up.head()

covid = covid.groupby("state", as_index=False).sum()

covid.head()
# Here we call the geopandas as gpd

fp = "/kaggle/input/covid19-tracking-germany/de_state.shp"

map_df = gpd.read_file(fp)

map_df = map_df.drop(columns=["ADE", "RS", "RS_0"])

# join the geodataframe with the csv dataframe

merged = map_df.merge(covid, how='left', left_on="GEN", right_on="state")

merged = merged[["state", "geometry", "cases", "deaths", "recovered"]]

Highest_number_cases = merged["cases"].max()

Highest_number_deaths = merged["deaths"].max()

Highest_number_recovered = merged["recovered"].max()

Highest_number_cases, Highest_number_deaths, Highest_number_recovered

print(merged)
merged.head(16)
# set the value column that will be visualised

variable = "cases"

# set the range for the choropleth values

vmin, vmax = 0, Highest_number_cases

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(20, 5))

# remove the axis

ax.axis('off')

# add a title and annotation

ax.set_title("# Covid-positiv Fälle nach Bundesland Region", fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate("Source: Robert Koch Institut", xy=(0.6, .05), xycoords='figure fraction', fontsize=12, color='#555555')

# Create colorbar legend

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm.set_array([]) # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it

# add the colorbar to the figure

fig.colorbar(sm) # Alternativ: fig.colorbar(sm, orientation="horizontal", fraction=0.036, pad=0.1, aspect = 30)

# create map

merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')



# Add Labels

merged["coords"] = merged["geometry"].apply(lambda x: x.representative_point().coords[:])

merged["coords"] = [coords[0] for coords in merged["coords"]]

for idx, row in merged.iterrows():

    plt.annotate(s=row["state"], xy=row['coords'],horizontalalignment='center')
# set the value column that will be visualised

variable = "deaths"

# set the range for the choropleth values

vmin, vmax = 0, Highest_number_deaths

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(20, 5))

# remove the axis

ax.axis('off')

# add a title and annotation

ax.set_title("# Covid-Todes-Fälle nach Bundesland Region", fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate("Source: Robert Koch Institut", xy=(0.6, .05), xycoords='figure fraction', fontsize=12, color='#555555')

# Create colorbar legend

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm.set_array([]) # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it

# add the colorbar to the figure

fig.colorbar(sm) # Alternativ: fig.colorbar(sm, orientation="horizontal", fraction=0.036, pad=0.1, aspect = 30)

# create map

merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')



# Add Labels

merged["coords"] = merged["geometry"].apply(lambda x: x.representative_point().coords[:])

merged["coords"] = [coords[0] for coords in merged["coords"]]

for idx, row in merged.iterrows():

    plt.annotate(s=row["state"], xy=row['coords'],horizontalalignment='center')
# set the value column that will be visualised

variable = "recovered"

# set the range for the choropleth values

vmin, vmax = 0, Highest_number_recovered

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(20, 5))

# remove the axis

ax.axis('off')

# add a title and annotation

ax.set_title("# Covid-Erholungs-Fälle nach Bundesland Region", fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate("Source: Robert Koch Institut", xy=(0.6, .05), xycoords='figure fraction', fontsize=12, color='#555555')

# Create colorbar legend

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm.set_array([]) # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it

# add the colorbar to the figure

fig.colorbar(sm) # Alternativ: fig.colorbar(sm, orientation="horizontal", fraction=0.036, pad=0.1, aspect = 30)

# create map

merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')



# Add Labels

merged["coords"] = merged["geometry"].apply(lambda x: x.representative_point().coords[:])

merged["coords"] = [coords[0] for coords in merged["coords"]]

for idx, row in merged.iterrows():

    plt.annotate(s=row["state"], xy=row['coords'],horizontalalignment='center')
covid_by_county.count()
covid_by_county.tail()
# Here we call the geopandas as gpd

fp_county = "/kaggle/input/covid19-tracking-germany/de_county.shp"

map_df_county = gpd.read_file(fp_county)

map_df_county = map_df_county.drop(columns=["ADE", "RS", "RS_0", "GF", "BSG", "AGS", "SDV_RS", "IBZ", "BEM", "NBD", "SN_L", "SN_R", "SN_K", "SN_V1", "SN_V2", "SN_G", "FK_S3", "NUTS", "AGS_0", "WSK", "DEBKG_ID"])

map_df_county.BEZ.unique()

map_df_county["BEZ_AbK"] = map_df_county["BEZ"]



map_df_county["BEZ_AbK"] = map_df_county["BEZ_AbK"].replace("Kreisfreie Stadt", "SK")

map_df_county["BEZ_AbK"] = map_df_county["BEZ_AbK"].replace("Landkreis", "LK")

map_df_county["BEZ_AbK"] = map_df_county["BEZ_AbK"].replace("Kreis", "LK")

map_df_county["BEZ_AbK"] = map_df_county["BEZ_AbK"].replace("Stadtkreis", "StadtRegion")

map_df_county["BEZ_AbK_merged"] = map_df_county["BEZ_AbK"] + " " + map_df_county["GEN"]





# You can rearrange columns directly by specifying their order:

# df = df[['a', 'y', 'b', 'x']]



map_df_county = map_df_county[["BEZ_AbK_merged", "GEN", "BEZ", "geometry", "BEZ_AbK"]]



map_df_county.head()
# join the geodataframe with the csv dataframe

merged_by_county = map_df_county.merge(covid_by_county, how='left', left_on="BEZ_AbK_merged", right_on="county")

merged_by_county = merged_by_county[["county", "geometry", "cases", "deaths", "recovered"]]

merged_by_county.columns = ["county", "geometry", "cases_by_county", "deaths_by_county", "recovered_by_county"]

# Highest_number_cases_by_county = merged_by_county["cases"].max()

# Highest_number_deaths_by_county = merged_by_county["deaths"].max()

# Highest_number_recovered_by_county = merged_by_county["recovered"].max()

# df['DataFrame Column'] = df['DataFrame Column'].fillna(0)

# map_df_county["cases_by_county"] = map_df_county["cases_by_county"].fillna("0")

uniqueValues = merged_by_county.nunique(dropna=False)

print(uniqueValues)
merged_by_county
# set the value column that will be visualised

variable = "recovered"

# set the range for the choropleth values

vmin, vmax = 0, 5000

# create figure and axes for Matplotlib

fig, ax = plt.subplots(1, figsize=(20, 5))

# remove the axis

ax.axis('off')

# add a title and annotation

ax.set_title("# Covid-Erholungs-Fälle nach Bundesland Region", fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate("Source: Robert Koch Institut", xy=(0.6, .05), xycoords='figure fraction', fontsize=12, color='#555555')

# Create colorbar legend

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm.set_array([]) # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it

# add the colorbar to the figure

fig.colorbar(sm) # Alternativ: fig.colorbar(sm, orientation="horizontal", fraction=0.036, pad=0.1, aspect = 30)

# create map

merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
