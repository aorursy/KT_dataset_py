# Import all of the needed libraries 
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
%matplotlib inline
#read the data and see the data
data = pd.read_csv("../input/avocado.csv",index_col=0,parse_dates=['Date'])
data.head()
#Are there any missing data datapoints or do any cols need to be manipulated/dropped/transformed?
data.info()
#Does the data looks evenly distributed?
print(data['type'].value_counts())
print(data.groupby('region')['Total Volume'].count())
data['year'].value_counts()
us_total = data.loc[data['region'].isin(['TotalUS']), :]
#select all regions except for the listed ones
us_cities = data.loc[~data['region'].isin(['TotalUS', 'GreatLakes', 'Southeast', 'Midsouth', 'Northeast',
                                           'SouthCentral', 'California','West', 'WestTexNewMexico', 'NorthernNewEngland']), :]
#select all of the regions
us_regions = data.loc[data['region'].isin(['GreatLakes', 'Southeast', 'Midsouth', 
                                           'Northeast', 'SouthCentral','West',
                                           'WestTexNewMexico', 'NorthernNewEngland']), :]
print(us_total.groupby('type')['AveragePrice'].mean())
print(us_cities.groupby('type')['AveragePrice'].mean())
print(us_total['Total Volume'].sum())
print(us_cities['Total Volume'].sum())
tot_vol = us_cities[us_cities['type']=='conventional'].groupby('region')['Total Volume', 'AveragePrice'].mean().sort_values(by='AveragePrice')
tot_vol.plot(kind='barh', figsize=(12,18), logx=True)
plt.figure(figsize=(12,12))
#fig, (ax1, ax2) = plt.subplots(1, 2)
tot_vol = us_cities[us_cities['type']=='conventional'].groupby('region')['Total Volume', 'AveragePrice'].mean().sort_values(by='AveragePrice')
tot_vol.plot(kind='scatter',x='Total Volume', y='AveragePrice',figsize=(10,6), logx=True)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_figheight(8)
fig.set_figwidth(14)
us_total[us_total['type']=='conventional'].groupby('Date')['Total Volume'].sum().plot(kind='line', ax=ax1, subplots=True)
us_total[us_total['type']=='conventional'].groupby('Date')['AveragePrice'].mean().plot(kind='line', ax=ax2, subplots=True)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_figheight(10)
fig.set_figwidth(14)
us_total.groupby(us_total['Date'].dt.month)['Total Volume'].sum().plot(kind='bar', ax=ax1, ylim=(308992617, 685625110), subplots=True)
us_total.groupby(us_total['Date'].dt.month)['AveragePrice'].mean().plot(kind='bar', ax=ax2, ylim=(1, 1.6), subplots=True)
import warnings
warnings.filterwarnings('ignore')
(
    ggplot(data=us_regions)
      + aes(y='AveragePrice', x='Total Volume')
      + aes(color='region', shape='region')
      + geom_point(alpha=0.5)
      + scale_x_log10()
      + coord_fixed(ratio=3/4)
      + facet_wrap('~type', nrow=2, ncol=1)
      + theme_classic()
)
import warnings
warnings.filterwarnings('ignore')
(
    ggplot(data=us_regions)
      + aes(y='AveragePrice', x='Total Volume')
      + aes(color='region', shape='region')
      + geom_point(alpha=0.5)
      + scale_x_log10()
      + coord_fixed(ratio=3/4)
      + facet_wrap('~type', nrow=2, ncol=1)
      + theme_classic()
)
%matplotlib notebook