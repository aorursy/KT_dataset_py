import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

import scipy



import plotly.express as px



#geo stuff

import geoplot

import geopandas

import geopandas.tools

import geoplot.crs as gcrs





%matplotlib inline

df = pd.read_csv("../input/cleaning-etsy-shops-raw-data/shops_add.csv")

df.head()
df.describe()
locations = df.copy()

locations.loc[(locations['seller_country'] == 'England'), 'seller_country'] = 'United Kingdom'

locations.loc[(locations['seller_country'] == 'Wales'), 'seller_country'] = 'United Kingdom'

locations.loc[(locations['seller_country'] == 'Scotland'), 'seller_country'] = 'United Kingdom'

locations.loc[(locations['seller_country'] == 'Northern Ireland'), 'seller_country'] = 'United Kingdom'



by_country = locations[['seller_country']].groupby(by='seller_country').size().sort_values(ascending=False).rename('number_of_shops')

by_country.head(n=10).plot.bar()



# set(df[ df['seller_town'].isna() & df['seller_country'].notna() ]['seller_country'])
countries_geocodes = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))[['name','geometry', 'iso_a3']]

by_country_geo = countries_geocodes.set_index('name').join(by_country, how='left').reset_index()

by_country_geo['no_shops'] = by_country_geo['number_of_shops'].isna()

by_country_geo['number_of_shops_log'] = np.log(by_country_geo['number_of_shops'])

# https://residentmario.github.io/geoplot/plot_references/plot_reference.html#choropleth

geoplot.choropleth(by_country_geo, hue='no_shops', cmap='Accent', figsize=(20, 10), )

geoplot.choropleth(by_country_geo, hue='number_of_shops_log', cmap='Greens', k = None,legend=True, figsize=(20, 10))

by_join_date = df[['seller_join_date']].groupby(by='seller_join_date').size().rename('number_of_shops')

by_join_date
b = by_join_date.plot.bar(title='# of shops per year')
by_join_date_reg = by_join_date[(by_join_date.index > 2000) & (by_join_date.index < 2019)] 

lr = scipy.stats.linregress(by_join_date_reg.index, by_join_date_reg.values)

lr
by_join_date_reg_df = by_join_date_reg.to_frame()

by_join_date_reg_df['number_of_shops_pred'] = [ x*lr.slope + lr.intercept for x in by_join_date_reg.index ]
plt.figure(figsize=(10, 5))

plt.xticks(by_join_date_reg_df.index)

plt.bar(by_join_date_reg_df.index, by_join_date_reg_df['number_of_shops'].values)

plt.plot(by_join_date_reg_df.index, by_join_date_reg_df['number_of_shops_pred'].values, 'r-')

plt.suptitle("Checking idea of linear growth of shops number")

plt.show()
df[ df['seller_join_date'] == 1969]
#let's remove this outlier - it'll imrpove data

df = df [df['seller_join_date'] >= 2005]
df['number_of_sales'].describe()
x = df['number_of_sales']



plt.figure(figsize=(10, 5))

n, bins, patches = plt.hist(x, 30, density=False, log=True, facecolor='g', alpha=0.75)

plt.xlabel('sales')

plt.ylabel('shops')

plt.title('Histogram of Sales')

plt.grid(True)

plt.show()

x = df['number_of_sales']



plt.figure(figsize=(10, 5))

n, bins, patches = plt.hist(x, 20, range=(0, 70000), density=False, log=True, facecolor='g', alpha=0.75)

plt.xlabel('sales')

plt.ylabel('shops')

plt.title('Histogram of Sales')

plt.grid(True)

plt.show()

n, bins
bins[32:34]
x = df['number_of_sales']



plt.figure(figsize=(10, 5))

n, bins, patches = plt.hist(x, 100, range=(50, 1000), density=False, log=False, facecolor='g', alpha=0.75)

plt.xlabel('sales')

plt.ylabel('shops')

plt.title('Histogram of Sales 50<x<1000')

plt.grid(True)

plt.show()

x = df['number_of_sales']



plt.figure(figsize=(10, 5))

n, bins, patches = plt.hist(x, 100, range=(0, 10), density=False, log=False, facecolor='g', alpha=0.75)

plt.xlabel('sales')

plt.ylabel('shops')

plt.title('Histogram of Sales 0<x<10')

plt.grid(True)

plt.show()

def get_pareto(N, number_of_sales):

    all_sales = number_of_sales.sum()

    all_shops = len(number_of_sales)

    NSales = number_of_sales.sort_values(ascending=True).tail(n=N).sum()

    return N / all_shops, NSales / all_sales



number_of_sales = df['number_of_sales']

get_pareto(1000, number_of_sales), get_pareto(10000, number_of_sales), get_pareto(100000, number_of_sales)
df['number_of_reviews'].describe()
x = df['number_of_reviews']



plt.figure(figsize=(10, 5))

n, bins, patches = plt.hist(x, 30, density=False, log=True, facecolor='g', alpha=0.75)

plt.xlabel('sales')

plt.ylabel('reviews')

plt.title('Histogram of Reviews')

plt.grid(True)

plt.show()

active_shops = df[ (df['number_of_sales'] > 50)]

# df['reviews_per_sales'] = df['number_of_reviews'] / df['number_of_sales']

x = active_shops['number_of_reviews'] / active_shops['number_of_sales']



plt.figure(figsize=(10, 5))

n, bins, patches = plt.hist(x, range=(0,1), log=False, facecolor='g', alpha=0.75)

plt.xlabel('reviews / sales')

plt.ylabel('shops')

plt.title('Histogram of Reviews / Sales')

plt.grid(True)

plt.show()
x.describe()
corrmat = df[ ['number_of_sales', 'number_of_reviews']].corr()

corrmat
df['average_review_score'].describe()
x = df[ (df['number_of_sales'] > 20)]['average_review_score']



plt.figure(figsize=(20, 5))

plt.subplot(121)

n, bins, patches = plt.hist(x, log=False, facecolor='g', alpha=0.75)

plt.xlabel('avg review score')

plt.ylabel('shops')

plt.title('Histogram of Avg review score')

plt.grid(True)



plt.subplot(122)

n, bins, patches = plt.hist(x, log=True, facecolor='g', alpha=0.75)

plt.xlabel('avg review score')

plt.ylabel('log shops')

plt.title('Log Histogram of Avg review score')

plt.grid(True)







plt.show()
df['number_of_items'].describe(percentiles=[.25, .5, .75, .9, .95])
x = df['number_of_items']



plt.figure(figsize=(20, 5))

plt.subplot(131)

n, bins, patches = plt.hist(x, 30, density=False, log=True, facecolor='g', alpha=0.75)

plt.xlabel('shops')

plt.ylabel('items')

plt.title('Histogram of Items')

plt.grid(True)





plt.subplot(132)

n, bins, patches = plt.hist(x, 10, range = (0, 10), density=False, log=False, facecolor='g', alpha=0.75)

plt.xlabel('shops')

plt.ylabel('items')

plt.title('Histogram of Items')

plt.grid(True)



plt.subplot(133)

n, bins, patches = plt.hist(x, 20, range = (100, 500), density=False, log=False, facecolor='g', alpha=0.75)

plt.xlabel('shops')

plt.ylabel('items')

plt.title('Histogram of Items')

plt.grid(True)



plt.show()
corrmat = df[ ['number_of_sales', 'number_of_items']].corr()

corrmat
df['sales_per_year'] = df['number_of_sales'] / (2019 + 1 - df['seller_join_date'])

df['reviews_per_year'] = df['number_of_reviews'] / (2019 + 1 - df['seller_join_date'])

df.head()
sns.set(style="ticks", color_codes=True)

cols = ['seller_join_date', 'number_of_sales', 'number_of_reviews', 'average_review_score', 'number_of_items', 'sales_per_year']

sns.pairplot(df[cols])
# cols = ['seller_join_date', 'number_of_sales', 'number_of_reviews', 'average_review_score', 'number_of_items', 'sales_per_year', 'reviews_per_year']

cols = ['seller_join_date', 'number_of_sales', 'number_of_reviews', 'average_review_score', 'number_of_items']



corrmat = df[cols].corr()

f, ax = plt.subplots(figsize=(6,6))

sns.heatmap(corrmat, annot=True, square=True, fmt='.2f',   annot_kws={'fontsize':12 });

plt.title('Pearson Correlation of Features')

plt.show()

sales_by_country = locations[['number_of_sales', 'seller_country']].groupby(by='seller_country').sum().sort_values(by='number_of_sales', ascending=False)

sales_by_country = sales_by_country[ sales_by_country['number_of_sales'] > 0 ]

sales_by_country['number_of_sales_log'] = np.log(sales_by_country['number_of_sales'])

sales_by_country[['number_of_sales']].head(n=10).plot.bar()

# countries that sell (c)

sales_by_country_geo = countries_geocodes.set_index('name').join(sales_by_country, how='left').reset_index()

geoplot.choropleth(sales_by_country_geo, hue='number_of_sales_log', cmap='Greens', k = None,legend=True, figsize=(20, 10))
all_locations_df = pd.read_csv("../input/cleaning-etsy-shops-raw-data/all_locations.csv")



sales_by_location = df[ ['seller_location', 'number_of_sales', 'seller_country', 'seller_town']].groupby(by=['seller_location', 'seller_country', 'seller_town']).sum().sort_values(by='number_of_sales', ascending=False)

sales_by_location = sales_by_location[ sales_by_location['number_of_sales'] > 0 ]

sales_by_location['number_of_sales_log'] = np.log(sales_by_location['number_of_sales'])

sales_by_location.head()



sales_by_location_geo = sales_by_location.reset_index().set_index('seller_location').join(all_locations_df.set_index('seller_location'), how='inner').reset_index()

sales_by_location_geo.head()

fig2 = px.scatter_geo(sales_by_location_geo, lat = 'lat', lon = 'lng',

                     size='number_of_sales',text='seller_location', hover_name='seller_location',

                     projection='natural earth', title='Sales by Location')

fig2.show()
sales_by_location_geo.sort_values(by='number_of_sales', ascending=False).head(n=10).plot.bar(x='seller_location', y='number_of_sales')
fig2 = px.scatter_geo(sales_by_location_geo, lat = 'lat', lon = 'lng',

                     size='number_of_sales_log',text='seller_location', hover_name='seller_location',

                     projection='natural earth', title='Sales by Location - log scale')

fig2.show()
import plotly.express as px



filtered = sales_by_location_geo[(sales_by_location_geo['seller_country'] != 'United States')]



fig2 = px.scatter_geo(filtered, lat = 'lat', lon = 'lng',

                     size='number_of_sales',text='seller_location', hover_name='seller_location',

                     projection='natural earth', title = 'Sales by location, filtered')

fig2.show()
filtered.sort_values(by='number_of_sales', ascending=False).head(n=10).plot.bar(x='seller_location', y='number_of_sales')