%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.ticker as tick

import pandas as pd

import seaborn as sns



from random import sample
# TODO: update directory to '../input/' format when publishing as a kaggle kernel

CITY_DATA_PATH = '../input/City_time_series.csv'

CITY_CROSSWALK_PATH = '../input/cities_crosswalk.csv'

MONTHS_PER_YEAR = 12

STATE_DATA_PATH = '../input/State_time_series.csv'
df = pd.read_csv(STATE_DATA_PATH, parse_dates=['Date'])
df.tail(3)
df[df.RegionName == 'California'].plot(x='Date', y='ZRI_AllHomes');
df = pd.read_csv(CITY_DATA_PATH, parse_dates=['Date'], usecols=['Date', 'RegionName', 'ZRI_AllHomes'])

df.tail(3)
df.info(max_cols=0)
cities_cw = pd.read_csv(CITY_CROSSWALK_PATH, index_col='Unique_City_ID')
cities_cw.head(3)
king_county_cities = cities_cw[(cities_cw['County'] == 'King') & (cities_cw['State'] == 'WA')].index.values.tolist()

print(sample(king_county_cities, 3))
print(len(king_county_cities))
reader = pd.read_csv(CITY_DATA_PATH, chunksize=10**5, parse_dates=['Date'])

useful_chunks = []

for chunk in reader:

    useful_chunks.append(chunk[chunk.RegionName.isin(king_county_cities)].copy())

df = pd.concat(useful_chunks, ignore_index=True)

del useful_chunks
df.tail(3)
df.info(max_cols=0)
most_recent_date = df['Date'].max()



valid_cities = set(df[

    (df['Date'] == most_recent_date) &

    ~df['ZHVI_AllHomes'].isnull() &

    ~df['MedianListingPrice_AllHomes'].isnull()

                      ]['RegionName'].values)

df = df[df['RegionName'].isin(valid_cities)].copy()
highest_cost_cities = df[['RegionName', 'ZHVI_AllHomes']].groupby('RegionName').max().sort_values(

    by=['ZHVI_AllHomes'], ascending=False)[:9].index.values.tolist()
df['MaxPrice'] = df[['RegionName', 'ZHVI_AllHomes']].groupby('RegionName').transform(max)

df = df[df['RegionName'].isin(highest_cost_cities)].sort_values(

        by=['MaxPrice', 'RegionName', 'Date'], ascending=False)

del df['MaxPrice']
core_metrics = ['ZHVI_AllHomes', 'MedianListingPrice_AllHomes']

df = df.melt(id_vars=['Date', 'RegionName'], value_vars=core_metrics)

df.head(3)
colors = {'ZHVI_AllHomes': 'green', 'MedianListingPrice_AllHomes': 'blue'}

facetplot = sns.FacetGrid(df, col='RegionName', hue='variable', col_wrap=3, palette=colors)

facetplot.set_xticklabels(rotation=45)

def y_fmt(y, x):

    return '${:,}'.format(int(y))

for axis in facetplot.axes:

    axis.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

facetplot.map(plt.plot, "Date", "value", marker=".", linewidth=1, markersize=5).add_legend();