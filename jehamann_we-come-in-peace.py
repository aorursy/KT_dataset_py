import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#fill some country values as US
sightings = pd.read_csv('../input/ufo-sightings-around-the-world/ufo_sighting_data.csv')
us_states= sightings[sightings.country == 'us']['state/province'].unique()
for row in sightings.itertuples():
    if (row._3 in us_states) and (type(row.country) == float):
        sightings.loc[row.Index,'country'] = 'us'

#set the geolocation latitude coordinate in proper float type
sightings.loc[43782,'latitude']='33.200088' # corrects a typo
sightings.latitude = pd.to_numeric(sightings.latitude)
sightings.loc[27822,'length_of_encounter_seconds']='2' # corrects a typo
sightings.loc[35692,'length_of_encounter_seconds']='8' # corrects a typo
sightings.loc[58591,'length_of_encounter_seconds']='0.5' # corrects a typo

sightings.length_of_encounter_seconds = pd.to_numeric(sightings.length_of_encounter_seconds)

#set the Date_time column as pd.datetime type
#correct some wrong time entries
for row in sightings.itertuples():
    if '24:00' in row.Date_time:
        sightings.loc[row.Index,'Date_time'] = sightings.iloc[row.Index].Date_time.replace('24:00', '23:59')
        
sightings['Date_time'] = pd.to_datetime(sightings['Date_time'], format='%m/%d/%Y %H:%M')

#set the df index Date_time so we have a proper time series for analysis
pd.to_datetime(sightings.index)
sightings.index = sightings.Date_time
#sightings.info()
by_year = sightings.resample('A').count()
ax = by_year.Date_time.plot()
ax.set_ylabel('Number of Sightings')
ax.set_xlabel('Year')
ax.set_xlim(['1940','2016'])
by_month_north = sightings[sightings.latitude > 0].resample('M').count()
by_month_south = sightings[sightings.latitude < 0].resample('M').count()
by_month = sightings.resample('M').count()

#function to extract monthly statistics for all years. stat = mean, std, min, max, etc
def get_month_prob(df, stat):
        years = np.arange(1943, 2015)
        month_name = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']
        all_months = pd.DataFrame(index=range(12))

        for year in years: #here we get the amount of sightings per month for each year normalized by the total sightings of the given year
            all_months[str(year)] = (df[str(year)].Date_time / df[str(year)].Date_time.sum()).reset_index(drop=True)
        all_months.index = month_name
        all_months_stats = all_months.T.describe() #we now get the statistics for each month for all the years from 1943 to 2014
        monthly= all_months_stats.loc[stat] #we now estract the mean value of the weighted sightings per month for all years
        return monthly

south = get_month_prob(by_month_south, 'mean')
north = get_month_prob(by_month_north, 'mean')
whole = get_month_prob(by_month, 'mean')

ax = whole.plot(kind='line',color='k' )
south.plot.bar(position=0,color='b', alpha=0.5, ax = ax, width=0.3)
north.plot.bar(position=1,color='r', alpha=0.5, ax = ax, width=0.3)

ax.set_xticks(np.arange(12))
ax.set_xticklabels(south.index)
ax.legend(['Whole world', 'South Hemisphere', ' North Hemisphere'])
ax.set_ylabel('Sighting probability')

total_sightings = by_month['1943':].Date_time.sum()
total_sightings_north = by_month_north['1943':].Date_time.sum()
print('Sightings in the north hemisphere are', str(round(total_sightings_north/total_sightings *100, 2)), '% of the total')
import datetime
import matplotlib.pyplot as plt

#get the days a time as columns
day_time = time=pd.DataFrame()
day_time['day'] = [sightings.index[x].day_name() for x in range(len(sightings))] # this extracts the day name from the date
day_time['time'] = [sightings.index[x].hour for x in range(len(sightings))] # this extracts the hour name from the date
grouped_day = day_time.groupby(by=['day'], as_index=False).count().reindex([1,5,6,4,0,2,3]).set_index('day') #reindex here sets the day names in weekly order (not alphabetical)

def get_yearly_medians(df):
    medians=[]   
    years = np.arange(1943, 2015)
    for year in years:
        year_medians = medians.append(df.loc[str(year)].length_of_encounter_seconds.median()) # get median in order to remove outliers
    return np.array(medians)
length_median = get_yearly_medians(sightings)

fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(131)
(grouped_day.time/grouped_day.time.sum()).plot.bar(x='day',y='time', rot=45)

ax2 = fig.add_subplot(132)
day_time.time.plot.hist(bins=24)
ax2.set_xlabel('Hour of the day')

ax3 = fig.add_subplot(133)
plt.hist(length_median, 20)
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Yearly frequency')

print('The average sighting length is', length_median.mean()/60, 'minutes') 
print('The median sighting length is', np.median(length_median)/60, 'minutes') 
ufo_shape = sightings.groupby(by='UFO_shape').count()
shapes = sightings.UFO_shape.unique()
ufo_shape = ufo_shape.Date_time / ufo_shape.Date_time.sum()
ax =ufo_shape.plot.bar(rot=65, figsize=(8,6))
ax.set_ylabel('Percentage')
#get the geolocation coordinate values of sightings
sightings_coord = sightings[['latitude', 'longitude']]

lon, lat = sightings_coord.longitude.values, sightings_coord.latitude.values
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
%matplotlib inline  
#m = Basemap(width=120000000,height=90000000,projection='lcc',
#            resolution='None',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=-40,lat_ts=20,resolution='c')
m1 = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=80,\
            llcrnrlon=-20,urcrnrlon=80,lat_ts=20,resolution='c')
x, y = m(lon, lat)
m.scatter(x,y, marker='o',color='y', alpha=0.5)
x1, y1 = m1(lon, lat)
m1.scatter(x1,y1, marker='o',color='y', alpha=0.5)

#m.fillcontinents(color='coral',lake_color='aqua')
#m.drawcoastlines()
#m.drawcountries()
#m.drawmapboundary(fill_color='aqua')
m.bluemarble()
m1.bluemarble()
plt.title("Sightings locations")
plt.show()

