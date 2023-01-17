import numpy as np # linear algebra

import pandas as pd # data processing

import datetime as dt # date and time processing functions

import itertools

import matplotlib.pyplot as plt # basic plotting 

import matplotlib.dates as mdates # date processing in matplotlib

from matplotlib.offsetbox import AnchoredText

plt.style.use('ggplot') # use ggplot style

%matplotlib inline 



# read in the data from the provided csv file

df = pd.read_csv('../input/seaice.csv')



# drop the 'Source Data' column as it obscures more useful columns and doesn't tell us much

df.drop('Source Data', axis = 1, inplace = True)



# convert the provided 3 column date format to datetime format and set it as the index

df['Date'] = pd.to_datetime(df[['Year','Month','Day']])

df.index = df['Date'].values



# split according to hemisphere, as we are expecting different trends for each

north = df[df['hemisphere'] == 'north']

south = df[df['hemisphere'] == 'south']
plt.figure(figsize=(9,3))

plt.plot(north.index,north['Extent'], label='Northern Hemisphere')

plt.plot(south.index,south['Extent'], label='Southern Hemisphere')



# add plot legend and titles

plt.legend(bbox_to_anchor=(0., -.362, 1., .102), loc=3, ncol=2, 

           mode="expand", borderaxespad=0.)

plt.ylabel('Sea ice extent (10^6 sq km)')

plt.xlabel('Date')

plt.title('Daily sea-ice extent');
# resample raw data into annual averages

northyear = north['01-01-1979':'31-12-2016'].resample('12M').mean()

southyear = south['01-01-1979':'31-12-2016'].resample('12M').mean()



# remove the initial and final item as they aer averaged incorrectly (also indexes seem bad)

northyear = northyear[1:-1]

southyear = southyear[1:-1]



plt.figure(figsize=(9,3))

plt.plot(northyear.Year,northyear['Extent'], marker = '.', label='Northern Hemisphere')

plt.plot(southyear.Year,southyear['Extent'], marker = '.', label='Southern Hemisphere')



# add plot legend and titles

plt.legend(bbox_to_anchor=(0., -.362, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

plt.ylabel('Sea ice extent (10^6 sq km)')

plt.xlabel('Date')

plt.title('Annual average sea-ice extent')

plt.xlim(1977, 2016)
# define date range to plot between

start = 1978

end = dt.datetime.now().year + 1



# define plot

f, axarr = plt.subplots(2, sharex=True, figsize=(9,6))





# organise plot axes (set x axis to months only and cycle colours according to gradient)

month_fmt = mdates.DateFormatter('%b')

axarr[0].xaxis.set_major_formatter(month_fmt)

axarr[0].set_prop_cycle(plt.cycler('color', 

                                   plt.cm.winter(np.linspace(0, 1, len(range(start, end))))))

axarr[1].set_prop_cycle(plt.cycler('color', 

                                   plt.cm.winter(np.linspace(0, 1, len(range(start, end))))))



# add plot legend and titles

axarr[0].set_ylabel('Sea ice extent (10^6 sq km)')

axarr[1].set_ylabel('Sea ice extent (10^6 sq km)')

axarr[1].set_xlabel('Month')

axarr[0].set_title('Annual change in sea-ice extent');

axarr[0].add_artist(AnchoredText('Northern Hemisphere', loc=3))

axarr[1].add_artist(AnchoredText('Southern Hemisphere', loc=2))



# loop for every year between the start year and current

for year in range(start, end):

    # create new dataframe for each year, 

    # and set the year to 1972 so all are plotted on the same axis

    nyeardf = north[['Extent', 'Day', 'Month']][north['Year'] == year]

    nyeardf['Year'] = 1972

    nyeardf['Date'] = pd.to_datetime(nyeardf[['Year','Month','Day']])

    nyeardf.index = nyeardf['Date'].values

    

    syeardf = south[['Extent', 'Day', 'Month']][south['Year'] == year]

    syeardf['Year'] = 1972

    syeardf['Date'] = pd.to_datetime(syeardf[['Year','Month','Day']])

    syeardf.index = syeardf['Date'].values

    

    # plot each year individually

    axarr[0].plot(nyeardf.index,nyeardf['Extent'], label = year)

    axarr[1].plot(syeardf.index,syeardf['Extent'])