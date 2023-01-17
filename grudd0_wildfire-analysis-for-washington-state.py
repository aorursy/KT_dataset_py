# import modules

%matplotlib inline

import pandas as pd

import sqlite3

import matplotlib.pyplot as plt

import datetime

import numpy as np

import datetime

import matplotlib.ticker as ticker

import matplotlib.dates as mdates

from mpl_toolkits.basemap import Basemap

import warnings

warnings.filterwarnings('ignore')
# read data from SQL database

input = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')

df = pd.read_sql_query("SELECT * FROM 'Fires'", input)

# convert DISCOVERY_DATE and CONT_DATE columes from Julian dates

epoch = pd.to_datetime(0, unit='s').to_julian_date()

df.DISCOVERY_DATE = pd.to_datetime(df.DISCOVERY_DATE - epoch, unit='D')

df.CONT_DATE = pd.to_datetime(df.CONT_DATE - epoch, unit='D')

df.index = pd.to_datetime(df.DISCOVERY_DATE)

df_wa = df[df.STATE == 'WA']

df_wa.info()
# analysis for yearly burn area

y=df_wa.FIRE_SIZE.resample('AS').sum().fillna(0)

ax = y.plot(kind='bar',figsize=(10,6))

# set xaxis major labels

ticklabels = ['']*len(y.index)

ticklabels[::1] = [item.strftime('%Y') for item in y.index[::1]]

ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

plt.gcf().autofmt_xdate()

plt.xlabel('Year')

plt.ylabel('Acres Burned');

plt.title('Acres Burned by Year');
# Extract the data we're interested in

lat = df_wa['LATITUDE'].values

lon = df_wa['LONGITUDE'].values

fsize = df_wa['FIRE_SIZE'].values

# Draw the map background

fig = plt.figure(figsize=(17, 10))

m = Basemap(projection='mill',llcrnrlon=-124. ,llcrnrlat=45.3,urcrnrlon=-117 ,urcrnrlat=49.1, resolution = 'h', epsg = 4269)

# do not know how to  download the background image in this kaggel kernel, so I had to 

# comment out the command to get the kernal to run.

#m.arcgisimage(service='World_Physical_Map', xpixels = 5000, verbose= False)

m.drawcoastlines(color='blue')

m.drawcountries(color='blue')

m.drawstates(color='blue')

# scatter plot 

m.scatter(lon, lat, latlon=True,

          c=np.log10(fsize), s=fsize*.01,

          cmap='Set1', alpha=0.5)

# create colorbar and legend

plt.colorbar(label=r'$\log_{10}({\rm Size Acres})$',fraction=0.02, pad=0.04)

plt.clim(3, 7)
# get value counts for cause

cause = df_wa.STAT_CAUSE_DESCR.value_counts()

# plot pie chart for cause distribution

fig,ax = plt.subplots(figsize=(10,10))

ax.pie(x=cause,labels=cause.index,rotatelabels=False, autopct='%.2f%%');

plt.title('Fire Cause Distribution');
# group cause colume in 2 year segments

df_wa_cause = df_wa.groupby(pd.Grouper(key='DISCOVERY_DATE', freq='2AS'))['STAT_CAUSE_DESCR'].value_counts()

ticklabels = ['1992 - 1993','1994 - 1995','1996 - 1997','1998 - 1999','2000 - 2001','2002 - 2003','2004 - 2005',

'2006 - 2007','2008 - 2009','2010 - 2011','2012 - 2013','2014 - 2015']

df_wa_cause

# Fire Cause Distribution 2 Year Windows

df_wa_cause_us = df_wa_cause.unstack()

ax = df_wa_cause_us.plot(kind='bar',x=df_wa_cause_us.index,stacked=True,figsize=(10,6))

plt.title('Fire Cause Distribution 2 Year Window')

plt.xlabel('2 Year Window')

plt.ylabel('Number Fires')

ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))

ax.yaxis.grid(False,'minor') # turn off minor tic grid lines

ax.yaxis.grid(True,'major') # turn on major tic grid lines;

plt.gcf().autofmt_xdate()
fig=plt.figure()

fig.set_figheight(10)

fig.set_figwidth(15)

plt.subplots_adjust(hspace=0.5)

plt.subplot(211)

plt.title('Lightning Caused')

plt.xlabel('Fire Size')

plt.grid()

plt.ylabel('Number Wildfires')

plt.hist(df_wa[df_wa['STAT_CAUSE_DESCR'] == 'Equipment Use']['FIRE_SIZE'],bins=20,bottom=.1)

plt.semilogy()

plt.subplot(212)

plt.title('Equipment Use Caused')

plt.xlabel('Fire Size')

plt.ylabel('Number Wildfires')

plt.grid()

plt.hist(df_wa[df_wa['STAT_CAUSE_DESCR'] == 'Lightning']['FIRE_SIZE'],bins=20,bottom=.1)

plt.semilogy();
# build cost dataframe for fiscal years

cost_array = np.array([[2002,33],[2003,25],[2004,29],[2005,18],[2006,22],[2007,47],[2008,25],[2009,30],[2010,26],[2011,16],[2012,13],[2013,47],[2014,31],[2015,89],[2016,146]])

df_cost = pd.DataFrame(cost_array)

df_cost.columns = ['year','cost']

df_cost.index = pd.date_range(start='2002-01-01', end='2016-01-01', freq='AS')

df_cost = df_cost[df_cost.year < 2016]['cost']

df_cost
# convert wildfire database index to fiscal years

df_wa_fy = df_wa.assign(Season=(df_wa.DISCOVERY_DATE - pd.offsets.MonthBegin(7)).dt.year + 1)

df_wa_fy.index = pd.to_datetime(df_wa_fy.Season,format="%Y")
# calculate acres burned for fiscal years from 2002 to 2015

df_burn_fy = df_wa_fy[(df_wa_fy.Season >= 2002)  & (df_wa_fy.Season <= 2015)]['FIRE_SIZE'].resample('AS').sum()

df_burn_fy
# total fires for fiscal years from 2002 to 2015

df_count_fy = df_wa_fy[(df_wa_fy.Season >= 2002) & (df_wa_fy.Season <= 2015)]['FIRE_SIZE'].resample('AS').count()

df_count_fy
# total fires > 50000 acres for fiscal years from 2002 to 2015

df_large_fy = df_wa_fy[(df_wa_fy.Season >= 2002) & (df_wa_fy.Season <= 2015)]

df_large_fy = df_large_fy[df_large_fy.FIRE_SIZE > 50000]

df_large_fy = df_large_fy['FIRE_SIZE'].resample('AS').count()

range = pd.date_range(start='2002-01-01', end='2015-01-01', freq='AS')

df_large_fy = df_large_fy.reindex(index=range,fill_value=0)

df_large_fy
# create dataframe for analysis

df_reg = pd.DataFrame({'cost':df_cost,'area':df_burn_fy,'count':df_count_fy,'large':df_large_fy})

df_reg
# import linear regression from sci-kit learn module

from sklearn.linear_model import LinearRegression

X = df_reg.drop('cost', axis=1)

# this creates a linear regression object

lm = LinearRegression()

lm.fit(X, df_reg.cost)
pd.DataFrame(list(zip(X.columns,lm.coef_)),columns=['features','estimatedCoefficients'])
plt.scatter(df_reg.cost,lm.predict(X))

plt.xlabel( 'Measured Cost')

plt.ylabel('Predicted Cost')
lm.score(X,df_reg.cost)