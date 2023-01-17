import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import os
from collections import Counter
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import geopandas as gpd
from shapely.geometry import Point

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/FAO.csv',  encoding='ISO-8859-1')
print(df.shape)
df.head()
countries = Counter(df.Area)
print("Number of Countries : {}".format(len(countries)))
fig, ax=plt.subplots(1,1, figsize=(18,8));
df.Area.hist(ax=ax, xrot=60);
## Getting the variations in food and feed
Counter(df.Element)
## food feed mean
ff_mean = df.groupby('Element').mean()
ff_med = df.groupby('Element').median()
## drop unused columns
ffmean_by_date = ff_mean.drop(['Area Code','Item Code','Element Code','latitude','longitude'],axis=1).T
ffmed_by_date = ff_med.drop(['Area Code','Item Code','Element Code','latitude','longitude'],axis=1).T
## re-index the years to date time
nidx = []
for i in ffmean_by_date.index:
    nidx.append(pd.to_datetime(i[1:]))
ffmean_by_date.index = nidx
ffmed_by_date.index = nidx
fig, ax=plt.subplots(1,2,figsize=(16,8))
ffmed_by_date.plot(ax=ax[0]);
ax[0].set_ylabel('Median of Total (1000s of Tonnes)');
ffmean_by_date.plot(ax=ax[1]);
ax[1].set_ylabel('Mean of Total (1000s of Tonnes)');
## sum the items by count and drop unneccesary columns
sum_ff_bycnt = df.groupby('Area').sum().drop(['Item Code','Element Code','Area Code','latitude','longitude'],axis=1)
sum_ff_bycnt.head()
## get the mean for the latitudes and longitudes for later for each area (should be pretty similar)
mean_lat_lon_bycnt = df.groupby('Area').mean()[['latitude','longitude']]
mean_lat_lon_bycnt.head()
## Take the mean of the sums year over year
year_item_mean = sum_ff_bycnt.mean(axis=1)
print(year_item_mean.sort_values(ascending=False).head(5))
print(year_item_mean.sort_values(ascending=False).tail(5))
fig, ax=plt.subplots(1,1,figsize=(14,8))
year_item_mean.sort_values(ascending=False).iloc[:30].plot(kind='bar', ax=ax, rot=90);
plt.ylabel('1000s of Tonnes (Food & Feed)');

## get top 5 from index
cnt = year_item_mean.sort_values(ascending=False).index[:5].values
# print(cnt)
top5 = sum_ff_bycnt.T
top5.index = nidx
fig, ax=plt.subplots(1,1,figsize=(14,8))
for c in cnt:
    top5[c].plot(ax=ax, legend=True);
plt.title('Top 5 Countries - Sum of all Items Year Over Year');
plt.ylabel('Food+Feed : 1000s Tonnes');
## get bottom 5 from index
cnt = year_item_mean.sort_values(ascending=False).index[-5:].values
# print(cnt)
bot5 = sum_ff_bycnt.T
bot5.index = nidx
fig, ax=plt.subplots(1,1,figsize=(14,8))
for c in cnt:
    bot5[c].plot(ax=ax, legend=True);
plt.title('Bottom 5 Countries - Sum of all Items Year Over Year');
plt.ylabel('Food+Feed : 1000s Tonnes');
## pul the mean lats and longs into the sums DF
sum_ff_bycnt['lat'] = mean_lat_lon_bycnt['latitude']
sum_ff_bycnt['lon'] = mean_lat_lon_bycnt['longitude']

## using panadas geometry
geometry = [Point(xy) for xy in zip(sum_ff_bycnt.lon, sum_ff_bycnt.lat)]
crs = {'init': 'epsg:4326'}
gitems = gpd.GeoDataFrame(sum_ff_bycnt, crs=crs, geometry=geometry)

## world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
## Just a quick and dirty /100 normalization to make the plot below more feasible.
sum_ff_bycnt['Year_Mean'] = year_item_mean/100 #np.min(year_item_mean)
sum_ff_bycnt.head()
fig, ax=plt.subplots(1,1,figsize=(18,9))
base = world.plot(color='white', edgecolor='black',ax=ax);
# gitems.plot(ax=ax)
src = ax.scatter(sum_ff_bycnt.lon, sum_ff_bycnt.lat, marker='o', 
           s = sum_ff_bycnt.Year_Mean, c=sum_ff_bycnt.Year_Mean,
           cmap=plt.get_cmap('jet'), alpha=0.4)
plt.colorbar(src);
plt.title('Year over Year Sum of Average Food+Feed  :  1000s of Tonnes / 100');
weird_points = sum_ff_bycnt[ ((sum_ff_bycnt['lat'] < 0) & (sum_ff_bycnt['lat'] > -25)) & (sum_ff_bycnt['lon'] < -130) ]
weird_points
items = Counter(df.Item)
print('Number of Items : {}'.format(len(items)), '\n', items)
## sum the items by count and drop unneccesary columns
sum_items = df.groupby('Item').sum().drop(['Item Code','Element Code','Area Code','latitude','longitude'],axis=1)
sum_items.head()
year_item_mean2 = sum_items.mean(axis=1)
print(year_item_mean2.sort_values(ascending=False).head(5))
print(year_item_mean2.sort_values(ascending=False).tail(5))
fig, ax=plt.subplots(1,1,figsize=(14,8))
year_item_mean2.sort_values(ascending=False).iloc[:30].plot(kind='bar', ax=ax, rot=90);
plt.title('Mean Year over Year of Total Items - Non Normalized');
## get top 5 from index
cnt2 = year_item_mean2.sort_values(ascending=False).index[:5].values
# print(cnt)
top5i = sum_items.T
top5i.index = nidx
fig, ax=plt.subplots(1,1,figsize=(14,8))
for c in cnt2:
    top5i[c].plot(ax=ax, legend=True);
plt.title('Top 5 Items - Sum of all Items Year Over Year');
plt.ylabel('Number of Items');
us_ff = df[df.Area == 'United States of America'][df.Item == 'Cereals - Excluding Beer']
us_ff.head()
# print(us_ff.columns)
## convert to a time series
us_ff_ts = us_ff.drop(['Area Abbreviation', 'Area Code', 'Area', 'Item Code', 'Item',
                       'Element Code', 'Element','Unit', 'latitude', 'longitude', ], axis=1).T
us_ff_ts.index = nidx
us_ff_ts.columns = ['Feed','Food']

cm_ff = df[df.Area == 'China, mainland'][df.Item == 'Cereals - Excluding Beer']
cm_ff_ts = cm_ff.drop(['Area Abbreviation', 'Area Code', 'Area', 'Item Code', 'Item',
                       'Element Code', 'Element','Unit', 'latitude', 'longitude', ], axis=1).T
cm_ff_ts.index = nidx
cm_ff_ts.columns = ['Feed','Food']

us_ff_ts.head()
fig, ax=plt.subplots(1,2,figsize=(18,8), sharey=True)
us_ff_ts.plot(style=['b-','r-'],ax=ax[0]);
cm_ff_ts.plot(style=['b-','r-'],ax=ax[1]);
plt.ylabel('1000s Tonnes');
ax[0].set_title('US Food and Feed - Cereals');
ax[1].set_title('China Food and Feed - Cereals');
import statsmodels.api as sma
import statsmodels.graphics as smg
import sklearn.metrics as skm
## First take a look at the log level of the items to 
## see what possible serial correlation exists
_ = smg.tsaplots.plot_acf(np.log(cm_ff_ts.Feed))
_ = smg.tsaplots.plot_pacf(np.log(cm_ff_ts.Feed))
## modeling the first difference at the log level
mod = sma.tsa.SARIMAX(endog=np.log(cm_ff_ts.Feed), order = (1,1,0), simple_differencing=False)
fit = mod.fit()
print(fit.summary())
_ = fit.plot_diagnostics(figsize=(12,8))
def pred_forc(fit):
    ## make sure simple_differencing in SARIMAX = False
    pred = fit.predict()
    pred.iloc[0] = np.log(cm_ff_ts.Feed.iloc[0]) #fill initial point since pred makes it 0
    pred_lvl = np.exp(pred)
    return pred_lvl
pred_lvl = pred_forc(fit)
fig, ax=plt.subplots(1,1,figsize=(12,8))
pred_lvl.plot(ax=ax, legend=True, label='Prediction');
cm_ff_ts.Feed.plot(ax=ax, legend=True);
plt.title('China Mainland - Feed');
## Average Percent Error 
ape = (cm_ff_ts.Feed.values-pred_lvl.values).sum()/cm_ff_ts.Feed.values.sum()
plt.xlabel('Average Percent Error = {:.3f}%'.format(ape*100));
# fit.forecast(steps=20).plot()
