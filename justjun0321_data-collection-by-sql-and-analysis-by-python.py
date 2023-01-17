# import package with helper functions 
import bq_helper

accidents  = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents .list_tables()
accidents .head("vehicle_2015")
query = """SELECT COUNT(consecutive_number) as amount,day_of_crash
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY day_of_crash
            ORDER BY amount DESC
        """
accidents_by_day_in_month = accidents .query_to_pandas_safe(query)
accidents_by_day_in_month.head()
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.barplot(x=accidents_by_day_in_month.day_of_crash,y=accidents_by_day_in_month.amount)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of Accidents by day")
plt.show()
query = """SELECT COUNT(consecutive_number) as accidents,
                  EXTRACT(Hour FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(Hour FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
print(accidents_by_hour)
import seaborn as sns

plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.barplot(x=accidents_by_hour.f0_,y=accidents_by_hour.accidents)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of Accidents by Rank of Hour")
plt.show()
query = """SELECT COUNT(hit_and_run) AS Amount,registration_state_name AS STATE
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
                 
hit_n_run_by_state = accidents.query_to_pandas_safe(query)
hit_n_run_by_state.head()
plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.barplot(x=hit_n_run_by_state.Amount,y=hit_n_run_by_state.STATE)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of hit and run of each state")
plt.show()
query = """With a AS(
                    SELECT consecutive_number,number_of_drunk_drivers
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                    WHERE number_of_drunk_drivers > 0)
            SELECT v.registration_state_name AS STATE,
                   COUNT(v.hit_and_run) AS Hit_and_run_Amount,
                   SUM(a.number_of_drunk_drivers) as Drunk_Drivers
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v
            Join a
                ON v.consecutive_number = a.consecutive_number
            WHERE hit_and_run = "Yes"
            GROUP BY 1
            ORDER BY 3 DESC
        """
## You can use number (1,2,3) to stand for the columns you select

DrunkDriveEvent_by_state = accidents.query_to_pandas_safe(query)
DrunkDriveEvent_by_state.head()
plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.regplot(x="Hit_and_run_Amount", y="Drunk_Drivers", data=DrunkDriveEvent_by_state)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of hit and run vs drunk drivers")
plt.show()
Northeast_list = ['Connecticut','Maine','Massachusetts','New Hampshire','Rhode Island','Vermont','New Jersey','New York','Pennsylvania']
Midwest_list = ['Illinois','Indiana','Michigan','Ohio','Wisconsin','Iowa','Kansas','Minnesota','Missouri','Nebraska','North Dakota','South Dakota']
South_list = ['Delaware','Florida','Georgia','Maryland','North Carolina','South Carolina','Virginia','District of Columbia','West Virginia','Alabama','Kentucky','Mississippi','Tennesse','Arkansas','Louisiana','Oklahoma','Texas']
West_list = ['Arizona','Colorado','Idaho','Montana','Nevada','New Mexico','Utah','Wyoming','Alaska','California','Hawaii','Oregon','Washington']
'Arizona' in West_list
#DrunkDriveEvent_by_state['Region'] = 'Other'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Northeast_list,'Region']='Northeast'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Midwest_list,'Region']='Midwest'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Northeast_list,'Region']='South'
#DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE in Northeast_list,'Region']='West'
DrunkDriveEvent_by_state['Region'] = 'Other'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Connecticut','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Maine','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Massachusetts','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New Hampshire','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Rhode Island','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Vermont','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New Jersey','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New York','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Pennsylvania','Region']='Northeast'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Illinois','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Indiana','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Michigan','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Ohio','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Wisconsin','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Iowa','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Kansas','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Minnesota','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Missouri','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Nebraska','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'North Dakota','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'South Dakota','Region']='Midwest'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Delaware','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Florida','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Georgia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Maryland','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'North Carolina','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'South Carolina','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Virginia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'District of Columbia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'West Virginia','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Alabama','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Kentucky','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Mississippi','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Tennesse','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Arkansas','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Louisiana','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Oklahoma','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Texas','Region']='South'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Arizona','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Colorado','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Idaho','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Montana','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Nevada','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'New Mexico','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Utah','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Wyoming','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Alaska','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'California','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Hawaii','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Oregon','Region']='West'
DrunkDriveEvent_by_state.loc[DrunkDriveEvent_by_state.STATE == 'Washington','Region']='West'
plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.lmplot(x="Hit_and_run_Amount", y="Drunk_Drivers", data=DrunkDriveEvent_by_state,hue= "Region")

ax.set_xticklabels(rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of hit and run vs drunk drivers")
plt.show()
DrunkDriveEvent_by_state.head()
#from matplotlib.patches import Polygon

#Map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
#Map.readshapefile('st99_d00', name='states', drawbounds=True)

#state_names = []
#for shape_dict in map.states_info:
#    state_names.append(shape_dict['NAME'])
    
#ax = plt.gca()

#seg = Map.states[DrunkDriveEvent_by_state['STATE']]
#poly = Polygon(seg, facecolor='red',edgecolor='red')
#ax.add_patch(poly)

#plt.show()
query = """With a AS(
                    SELECT consecutive_number,number_of_drunk_drivers,latitude,
                    longitude,number_of_motor_vehicles_in_transport_mvit
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                    WHERE number_of_drunk_drivers > 0)
            SELECT ROUND(latitude,0) as latitude,
                   ROUND(longitude,0) as longtitude,
                   COUNT(v.hit_and_run) AS Hit_and_run_Amount,
                   SUM(a.number_of_drunk_drivers) as Drunk_Drivers
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` v
            Join a
                ON v.consecutive_number = a.consecutive_number
            WHERE hit_and_run = "Yes"
            GROUP BY 1,2
            ORDER BY 4 DESC
        """

DrunkDriveEvent_map = accidents.query_to_pandas_safe(query)
DrunkDriveEvent_map
DrunkDriveEvent_map.info()
DrunkDriveEvent_map.Drunk_Drivers = DrunkDriveEvent_map.Drunk_Drivers.astype(float)
DrunkDriveEvent_map.Hit_and_run_Amount = DrunkDriveEvent_map.Hit_and_run_Amount.astype(float)
import numpy as np
from mpl_toolkits.basemap import Basemap

Map = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130.,urcrnrlon=-60.,lat_ts=20,resolution='i')
Map.drawmapboundary(fill_color='paleturquoise')
Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()
used = set()

min_marker_size = 0.5
for i in range(0,179):
    x,y = Map(DrunkDriveEvent_map.longtitude[i], DrunkDriveEvent_map.latitude[i])
    msize = min_marker_size * DrunkDriveEvent_map.Drunk_Drivers[i]
    Map.plot(x, y, markersize=msize)
    
plt.show()
import numpy as np
from mpl_toolkits.basemap import Basemap

Map = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130.,urcrnrlon=-60.,lat_ts=20,resolution='i')
Map.drawmapboundary(fill_color='paleturquoise')
Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()
used = set()

x,y = Map(DrunkDriveEvent_map['longtitude'].values, DrunkDriveEvent_map['latitude'].values)
Map.plot(x, y, 'ro')
    
plt.show()
query = """SELECT COUNT(consecutive_number) as accidents,
                  EXTRACT(Hour FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(Hour FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)