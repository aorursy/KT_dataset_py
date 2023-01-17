#Import different packages



import os

import numpy as np

import pandas as pd

import seaborn as sns

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



plt.style.use("fivethirtyeight")

plt.rcParams['figure.figsize'] = (12, 8)
all_files=[]



for dirname, _, filenames in os.walk('/kaggle/input/airline-2019'):

    for filename in filenames:

        all_files.append(os.path.join(dirname, filename))

        

li = []



for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)



airline = pd.concat(li, axis=0, ignore_index=True)
#Delete temp files 

del li
# Clean city data in the dataset

citycode=pd.DataFrame(airline['ORIGIN'].unique())

citycode.columns=['ORIGIN']

city=citycode.merge(airline[['ORIGIN','ORIGIN_CITY_NAME']], how='left').drop_duplicates()

city[['Cityname', 'State']]=city['ORIGIN_CITY_NAME'].str.split(",", expand=True)

city=city.drop(columns=['ORIGIN_CITY_NAME'])

city=city.reset_index().drop(columns='index')

city.columns=['Code', 'Cityname', 'State']



#import airport GPS information

airportgps=pd.read_csv('/kaggle/input/airport/ICAO_airports.csv')



#remove NaN values and only keep IATA code and GPS coordinates

airportgps=airportgps[['iata_code','latitude_deg', 'longitude_deg']].dropna()

airportgps.columns=['Code', 'Lat', 'Lng']

city=city.merge(airportgps, how='left')



city.head()
#Since the city dataframe includes all information, the additional information can be deleted.

airline=airline.drop(columns=['ORIGIN_CITY_NAME', 'ORIGIN_STATE_NM', 'DEST_CITY_NAME', 'DEST_STATE_NM'])

airline.head(3)
airline.columns
airline.describe()
#Let's check the departure time, taxi out and wheels off relationship. 

#It seems the majority of time difference is 0. And some -40 number is due to formate.  

(airline['DEP_TIME']+airline['TAXI_OUT']-airline['WHEELS_OFF']).value_counts()
# Same applies to Wheels on and taxi_in time. 

(airline['WHEELS_ON']+airline['TAXI_IN']-airline['ARR_TIME']).value_counts()
#Based on the above analysis, these columns can be deleted

airline=airline.drop(columns=['WHEELS_OFF','WHEELS_ON', 'Unnamed: 25'])
airline['CANCELLED'].value_counts()
airline['CANCELLATION_CODE'].value_counts()
airline['CANCELLATION_CODE'].value_counts().sum()
airline=airline.drop(columns='CANCELLED')
airline.head(5)
#Create a new column with total delay

airline['Total_Delay']=airline.iloc[:,-5:].sum(axis=1)
airline.head(3)
delayed=airline[airline['Total_Delay']>0.5]

delayed['OP_CARRIER_AIRLINE_ID'].value_counts().plot(kind='bar')

plt.xlabel('Operating airline')

plt.ylabel('Total counts of delayed flights')

plt.title("Number of delayed flight for different operators", size=20)

plt.tight_layout()
#Is time an important factor for delay? 

flight_time=round(delayed['DEP_TIME']/100)

delayed['flight_time']=pd.Series(flight_time)

sns.countplot(delayed['flight_time'].dropna(), color='r')

plt.xlabel('Flight Time')

plt.ylabel('Total counts of delayed flights')

plt.xticks(rotation=90)

plt.title("Number of delayed flight at different hours", size=20)

plt.tight_layout()
airportdelay=delayed.groupby(['ORIGIN'])['Total_Delay'].agg(['count', 'mean']).reset_index().sort_values(by='count', ascending=False)[:15]



fig, ax1 = plt.subplots()



ax2 = ax1.twinx()

ax1.bar(airportdelay['ORIGIN'],airportdelay['count'])

ax2.plot(airportdelay['ORIGIN'],airportdelay['mean'], 'b-')



ax1.set_xlabel('Airport')

ax1.set_ylabel('Total number of delayed flight')

ax1.set_ylim([0,81000])

ax2.set_ylabel('Average delay (min)', color='b')

ax2.set_ylim([0,81])



plt.title('Top 15 aiports with the most delay and average delay in time')

plt.show()
#Which are the most delayed route? Top 50 routes

mostdelay=delayed.groupby(['ORIGIN','DEST'])['Total_Delay'].agg(['count', 

                'mean']).reset_index().sort_values(ascending=False, by='count')[:50]

longestdelay= delayed.groupby(['ORIGIN','DEST'])['Total_Delay'].agg(['count', 'mean']).reset_index().sort_values(ascending=False,

                by='count')[:50].sort_values(ascending=False, by='count')
mostdelay=mostdelay.merge(city, left_on='ORIGIN', right_on='Code', how='left')

mostdelay=mostdelay.drop(columns=['Code', 'Cityname', 'State'])

mostdelay=mostdelay.rename(columns={'Lat': 'Orgin_Lat', 'Lng':'Origina_Lng'})

mostdelay=mostdelay.merge(city, left_on='DEST', right_on='Code', how='left')

mostdelay=mostdelay.drop(columns=['Code', 'Cityname', 'State',])

mostdelay=mostdelay.rename(columns={'Lat': 'Dest_Lat', 'Lng':'Dest_Lng'})

mostdelay.head()
top20=mostdelay.iloc[:20]
majoraiport=mostdelay[['ORIGIN', 'Orgin_Lat', 'Origina_Lng']].drop_duplicates()
# create new figure, axes instances.

fig=plt.figure()

ax=fig.add_axes([0.1,0.1,0.8,0.8])



# setup mercator map projection.

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc', resolution='l', lat_1=33,lat_2=45,lon_0=-95,)



for startlat, startlng, endlat, endlng, delay in zip(top20['Orgin_Lat'],top20['Origina_Lng'], top20['Dest_Lat'], 

                                                     top20['Dest_Lng'], top20['count']):



    m.drawgreatcircle(startlng,startlat,endlng,endlat,linewidth=delay/500,color='red')





for lat, lng, label1 in zip(majoraiport['Orgin_Lat'],majoraiport['Origina_Lng'], majoraiport['ORIGIN']):

    x, y = m(lng, lat)

    plt.plot(x, y, 'ob', markersize=10)

    x1, y1 = m(lng+0.5, lat-0.5)

    plt.text(x1, y1, label1, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5));

    

m.drawcoastlines()

m.fillcontinents()

m.drawcoastlines()

m.drawcountries(linewidth=2)

m.drawstates()

m.fillcontinents(color='coral',lake_color='aqua', zorder = 1,alpha=0.4)

m.drawmapboundary(fill_color='aqua')

ax.set_title('Top 20 routes with the most delayed flight')







plt.show()