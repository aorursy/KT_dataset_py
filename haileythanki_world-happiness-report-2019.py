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
df19 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
df19
df19.info()
import matplotlib.pyplot as plt 

import geopandas as gpd

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world
world[~world.name.isin(df19["Country or region"])]
df19[~df19["Country or region"].isin(world.name)]

world.at[176, "name"] = "South Sudan"

world.at[4, "name"] = "United States"

world.at[17,"name"] = "Dominican Republic"

world.at[160,"name"] = "Northern Cyprus"

world.at[11,"name"] = "Congo (Kinshasa)"

world.at[153,"name"] = "Czech Republic"

world.at[171,"name"] = "North Macedonia"

world.at[79,"name"] = "Palestinian Territories"

world.at[175,"name"] = "Trinidad & Tobago"

world.at[170,"name"]="Bosnia and Herzegovina"

world.at[60,"name"]="Ivory Coast"

world.at[66,"name"]="Central African Republic"

world.at[73,"name"]="Swaziland"

world.at[67,"name"]="Congo (Brazzaville)"
df19[~df19["Country or region"].isin(world.name)]
for_plotting1 = world.merge(df19, left_on = 'name', right_on = "Country or region", how="left")

for_plotting2 = for_plotting1

for_plotting1
ax = for_plotting1.dropna().plot(column='Score', cmap = 'viridis', figsize=(20,15),scheme='quantiles', k=8, legend = True);

for_plotting2[for_plotting2.Score.isna()].plot(color='lightgrey', ax=ax)

ax.set_title('Happiness score of countries (2019)', fontdict= {'fontsize':15})

ax.set_axis_off()

ax.get_legend().set_bbox_to_anchor((.12,.12))

unknown_factors = df19["Score"]-(df19["GDP per capita"]+df19["Social support"]+df19["Healthy life expectancy"]+df19["Freedom to make life choices"]+df19["Generosity"]+df19["Perceptions of corruption"])

for_plotting3 = df19.copy()

for_plotting3["unknown contributing factors"] = unknown_factors

for_plotting3.set_index("Country or region",drop=True,inplace=True)

for_plotting3[:20][["GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption","unknown contributing factors"]].plot(kind="bar", figsize=(12,8), cmap = "viridis", stacked=True)

plt.title("Contribution of each factor to happiness score for top 20 happiest countries")

plt.xlabel("Countries")

plt.ylabel("Contribution")

ax.get_legend().set_bbox_to_anchor((.12,.12))
unknown_factors = df19["Score"]-(df19["GDP per capita"]+df19["Social support"]+df19["Healthy life expectancy"]+df19["Freedom to make life choices"]+df19["Generosity"]+df19["Perceptions of corruption"])

for_plotting4 = df19.copy()

for_plotting4["unknown contributing factors"] = unknown_factors

for_plotting4.set_index("Country or region",drop=True,inplace=True)

for_plotting4[-20:][["GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption","unknown contributing factors"]].plot(kind="bar", figsize=(12,8), cmap = "viridis", stacked=True)

plt.title("Contribution of each factor to happiness score for the 20 unhappiest countries")

plt.xlabel("Countries")

plt.ylabel("Contribution")

ax.get_legend().set_bbox_to_anchor((.12,.12))
import seaborn as sns

corr_matrix = df19.corr()

corr_matrix

mask = np.zeros_like(corr_matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)]= True

f, ax = plt.subplots(figsize=(11, 15)) 

heatmap = sns.heatmap(corr_matrix, mask = mask,square = True,linewidths = .5, cmap = "viridis", cbar_kws = {'shrink': .4, 'ticks' : [-1, -.5, 0, 0.5, 1]}, vmin = -1, vmax = 1, annot = True, annot_kws = {"size": 12})

ax.set_yticklabels(corr_matrix.columns, rotation = 0)

ax.set_xticklabels(corr_matrix.columns)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})