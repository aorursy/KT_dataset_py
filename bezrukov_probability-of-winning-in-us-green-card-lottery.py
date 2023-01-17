# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math



import geopandas as gpd

import geoplot.crs as gcrs

import geoplot as gplt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Adjust display settings

pd.set_option('display.max_rows', 500)
df = pd.read_csv('../input/DV_lottery_dataset.csv', sep=';')

df.fillna(0, inplace = True)

#print(df.info())
path = gpd.datasets.get_path('naturalearth_lowres')

gdf = gpd.read_file(path)

gdf = gdf[gdf.name!="Antarctica"]

print(gdf.info())
# We will merge it on 'name' column

df['name'] = df['Foreign State']



# Fixing discrepancy in the country names

df.loc[df['Foreign State']== 'French Southern and Antarctic Lands', ['name']] = 'Fr. S. Antarctic Lands'

df.loc[df['Foreign State']== 'Bosnia and Herzegovina', ['name']] = 'Bosnia and Herz.'

df.loc[df['Foreign State']== 'Cote d\'Ivoire', ['name']] = 'Côte d\'Ivoire'

df.loc[df['Foreign State']== 'Congo, Democratic Republic of The', ['name']] = 'Dem. Rep. Congo'

df.loc[df['Foreign State']== 'Congo, Republic of The', ['name']] = 'Congo'

df.loc[df['Foreign State']== 'Czech Republic', ['name']] = 'Czech Rep.'

df.loc[df['Foreign State']== 'Equatorial Guinea', ['name']] = 'Eq. Guinea'

df.loc[df['Foreign State']== 'North Korea', ['name']] = 'Dem. Rep. Korea'

df.loc[df['Foreign State']== 'Western Sahara', ['name']] = 'W. Sahara'

df.loc[df['Foreign State']== 'South Sudan', ['name']] = 'S. Sudan'

df.loc[df['Foreign State']== 'Solomon Islands', ['name']] = 'Solomon Is.'

df.loc[df['Foreign State']== 'Laos', ['name']] = 'Lao PDR'

df.loc[df['Foreign State']== 'Central African Republic', ['name']] = 'Central African Rep.'

df.loc[df['Foreign State']== 'Solomon Islands', ['name']] = 'Solomon Is.'

gdf = gdf.merge(df, on='name', how="left")

#print(gdf.info())
df[['Foreign State', '2018_Total']].sort_values(by='2018_Total',ascending= False)
gplt.choropleth(gdf, hue=gdf['2018_Total'],projection=gcrs.Robinson(),

                cmap='Purples', linewidth=0.5, edgecolor='gray', 

                k=None, legend=True, figsize=(20, 8))

plt.title("Total number of participants in 2018")
gplt.choropleth(gdf, hue=gdf['2007_Total'],projection=gcrs.Robinson(),

                cmap='Purples', linewidth=0.5, edgecolor='gray', 

                k=None, legend=True, figsize=(20, 8))

plt.title("Total number of participants in 2007")
df[['Foreign State', '2017_visas']].sort_values(by='2017_visas',ascending= False)
gplt.choropleth(gdf, hue=gdf['2017_visas'],projection=gcrs.Robinson(),

                cmap='Purples', linewidth=0.5, edgecolor='grey', 

                k=None, legend=True, figsize=(20, 8))

plt.title("Number of winners in 2017")
gplt.choropleth(gdf, hue=gdf['2008_visas'],projection=gcrs.Robinson(),

                cmap='Purples', linewidth=0.5, edgecolor='gray', 

                k=None, legend=True, figsize=(20, 8))

plt.title("Number of winners in 2008")
df['2017_p'] = df['2017_visas']/df['2017_Total']

df['2017_p'].describe()
df[['Foreign State', '2017_p']].sort_values(by='2017_p',ascending= False)
gdf['2017_p'] = gdf['2017_visas']/gdf['2017_Total']

gplt.choropleth(gdf, hue=gdf['2017_p'],projection=gcrs.Robinson(),

                cmap='Purples', linewidth=0.5, edgecolor='gray', 

                k=None, legend=True, figsize=(20, 8))

plt.title("Probability of winning in 2017")
gdf['2010_p'] = gdf['2010_visas']/gdf['2010_Total']

gplt.choropleth(gdf, hue=gdf['2010_p'],projection=gcrs.Robinson(),

                cmap='Purples', linewidth=0.5, edgecolor='gray', 

                k=None, legend=True, figsize=(20, 8))

plt.title("Probability of winning in 2010")
def nestim(x, win_prob):

    if (x>0) and (x<=1):

        return math.log(1-win_prob)/math.log(1-x)

    else: return np.nan



df['2017_Nyears'] = df['2017_p'].apply(lambda x: nestim(x, 0.95)) 

print(df['2017_Nyears'].describe())
df[['Foreign State', '2017_Nyears']].sort_values(by='2017_Nyears',ascending= False)
gdf['2017_Nyears'] = gdf['2017_p'].apply(lambda x: nestim(x, 0.95)) 

gplt.choropleth(gdf, hue=gdf['2017_Nyears'],projection=gcrs.Robinson(),

                cmap='Purples', linewidth=0.5, edgecolor='gray', 

                k=None, legend=True, figsize=(20, 8))

plt.title("Number of years one needs to play to win with 0.95 probability")
# outcome: 1 - win, 0 -lose

p = 0.005457

q = 1 - p

outcome, probability, n = [0,1], [q, p], 548

results = []

for i in range(100000):

    outcomes = np.random.choice(outcome, size=n, p=probability) 

    if 1 in outcomes: results.append(1)

    else: results.append(0)

print('Probability of winning in {0} trials is {1}'.format(n, sum(results)/len(results)))
# outcome: 1 - win, 0 -lose

p = 0.014904

q = 1 - p

outcome, probability, n = [0,1], [q, p], 200

results = []

for i in range(100000):

    outcomes = np.random.choice(outcome, size=n, p=probability) 

    if 1 in outcomes: results.append(1)

    else: results.append(0)

print('Probability of winning in {0} trials is {1}'.format(n, sum(results)/len(results)))