# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from scipy import stats, integrate

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format
flight_data=pd.read_csv("../input/flight_data.csv")

flight_data.head()
df_summary = flight_data.describe()

df_summary
# 1. Date

flight_data['DATE'] = pd.to_datetime(flight_data[['year','month','day']], yearfirst=True)



# 2. Month name

month_dict={

    1:  '01- January',

    2:  '02- February',

    3:  '03- March',

    4:  '04- April',

    5:  '05- May',

    6:  '06- June',

    7:  '07- July',

    8:  '08- August',

    9:  '09- September',

    10: '10- October',

    11: '11- November',

    12: '12- December'

}

flight_data['MONTH_desc'] = flight_data['month'].apply(lambda m: month_dict[m])



flight_data.head()
rows,cols = flight_data.shape

print("Number of rows: ", rows)

print("Number of columns: ", cols)
flight_data.info()
# Copy of Flight_data after dropping null values

flt_data_copy= flight_data.dropna()

flt_data_copy.head()
# Number of scheduled departures planned by carriers in 2013



carrier_count = flight_data['carrier'].value_counts()

carrier_count
#Graphical representation of carriers scheduled flights in numbers and %



f,ax=plt.subplots(1,2,figsize=(20,10))

flight_data['carrier'].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=False)                                                  

ax[1].set_title('Flight % of Carrier')

ax[1].set_ylabel('')

sns.countplot('carrier',order = flight_data['carrier'].value_counts().index, data=flight_data,ax=ax[0])

ax[0].set_title('Frequency Distribution of Carriers')

ax[0].set_ylabel('Number of Flights')

plt.show()
#1 Overall pattern of departure time from NYC airports 



flt_data_copy.boxplot('dep_time','origin', rot = 40, figsize=(6,6))



plt.show()



print("The highest IQ range (900 to 1800) is for JFK, where 75% flights are falling under the departure time of 1800 and middle data point for departure is 1500 ")



# Number of scheduled departure from different origin



allflight_NYC=flt_data_copy['dest'].unique()    

allflightcount_NYC=len(allflight_NYC)      # Total number of destination is 104

print("The total number of destination flight from NYC is"+ "::" ,allflightcount_NYC)

print('\n')



flt_data_copy['origin'].value_counts()
origin_count = flt_data_copy['origin'].value_counts()

f,ax=plt.subplots(1,2,figsize=(20,10))

flt_data_copy['origin'].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=False)                                                  

ax[1].set_title('Percentage of Flt from Origin')

ax[1].set_ylabel('')

sns.countplot('origin',order = flt_data_copy['origin'].value_counts().index, data=flt_data_copy,ax=ax[0])

ax[0].set_title('Frequency Distribution of Flt from Origin')

ax[0].set_ylabel('Flight Count')

plt.show()
# Number of scheduled arrival at unique destination



Destination= flight_data['dest'].value_counts().sort_values(ascending=False).head(15)



Destination
#Top10 destination flight from NYC in bar graph



f,ax=plt.subplots(1,2,figsize=(20,10))

flight_data['dest'].value_counts().head(10).sort_values(ascending=False).plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=False)                                                

ax[1].set_title('Percentage of Flt from dest')

ax[1].set_ylabel('')

sns.countplot('dest',order = flight_data['dest'].sort_values(ascending=False).value_counts().head(10).index, data=flight_data,ax=ax[0])

ax[0].set_title('Frequency Distribution of Flt from dest')

ax[0].set_ylabel('Count by Flight_Destination ')

plt.show()
#Top10 destination flight from NYC 



dest_count=flight_data.groupby(['dest'],as_index=False).agg({'month':'count'})

max_dest_count=dest_count.sort_values(['month'], ascending=False)

print("The top_10 destination flight from NYC are")

top_dest_flight = max_dest_count.head(10)

plt.scatter(top_dest_flight.dest,top_dest_flight.month, color='red')

plt.legend

plt.grid(True, color='g', linewidth=1)

plt.show()

top_dest_flight
#  Maximum number of flights headed to unique destination from Origin.



dest_flight = flight_data.groupby('origin')['dest'].value_counts()



dest_flight.head(10)

#Total number of unique Airline headed to BOS from NYC



BOS_dest=flight_data[flight_data['dest']=='BOS']

Carrier_count=(BOS_dest['carrier']).unique()



print("Carrier fly to BOS", Carrier_count)

print('\n')



print("Total number of Carrier  headed to 'BOS' from NYC is",len(Carrier_count))

print('\n')



BOS_dest_count=BOS_dest['tailnum'].unique()

print("Total unique aircraft  headed to 'BOS' from NYC is",len(BOS_dest_count))

print('\n')
# Avg. Monthly Departure Delay for Carrier



Monthly_Avg_Delay= flt_data_copy.groupby(['carrier','MONTH_desc'], axis=0, as_index=True).agg({'dep_delay':'mean'})

Monthly_Avg_Delay.head(15)
# Monthly Average Departure Departure Delay



monthly_delay = flight_data.groupby(['month'], as_index=False).agg({'dep_delay': 'mean'})

monthly_delay['dep_delay']=np.round(monthly_delay['dep_delay'],0)

sns.catplot(x='month', y='dep_delay',data=monthly_delay, kind='bar')

plt.title("Monthly Average Departure Delay") 

plt.plot()

plt.show()

#Avg_arr_delay_sort by Carrier in 2013 (inclusive early arrival)



flt_data_copy.groupby('carrier').agg(np.size)                 

top_delay = flt_data_copy.groupby('carrier').agg({'arr_delay' :[np.size,np.mean]})

                                                

                                                                

top_delay.sort_values([('arr_delay', 'mean')], ascending=False).head(16)
#Average arrival delay by carrier in 2013   (inclusive early arrival)



f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('carrier','arr_delay', data=flt_data_copy,ax=ax[0], order=['F9', 'FL', 'EV', 'YV','OO','MQ','WN','B6',

                                                                                '9E','UA', 'US','VX','DL', 'AA','HA','AS'])

                                                                                 

ax[0].set_title('Average Arrival Delay by Carrier')

sns.boxplot('carrier','arr_delay', data=flt_data_copy,ax=ax[1], order=['F9', 'FL', 'EV', 'YV','OO','MQ','WN','B6',

                                                                                '9E','UA', 'US','VX','DL', 'AA','HA','AS'])

ax[1].set_title('Delay Distribution by Carrier')

plt.close(2)

plt.show()



print(['WN: Southwest Airlines', 'AA: American Airlines', 'MQ: American Eagle Airlines', 'UA: United Airlines',

       'OO: Skywest Airlines','DL: Delta Airlines','US: US Airways','YV: Mesa Airlines', 'HA: Hawaiian Airlines',

       'EV: Atlantic Southeast Airlines','FL: AirTran Airways','F9: Frontier Airlines','VX: Virgin America',

       'B6: JetBlue Airways','9E: Pinnacle Airlines','AS: Alaska Airlines'])
#Avg_arr_delay_ for destination in 2013

flt_data_copy.groupby('dest').agg(np.size)                 # size() is calculating the count

airport_delay = flt_data_copy.groupby('dest').agg({'arr_delay' :[np.size,np.mean]})

                                                                



airport_delay.sort_values([('arr_delay','mean')], ascending=False).head(10)
#Top_10_arr_delay_for Destination with sample size>1000



top_arr_delay = airport_delay['arr_delay']['size'] >= 1000       # only variable with sample size >= 1000 will be considered 



airport_delay[top_arr_delay].sort_values([('arr_delay', 'mean')], ascending=False)[:10]
# Departure_Delay status classifications



flt_data_copy.loc[flt_data_copy['dep_delay'] <= 0, 'Status'] = 0

flt_data_copy.loc[flt_data_copy['dep_delay'] >= 1, 'Status'] = 1

flt_data_copy.loc[flt_data_copy['dep_delay'] >= 15, 'Status'] = 2

flt_data_copy.loc[flt_data_copy['dep_delay'] >= 60, 'Status'] = 3



# Departure_Delay_status_graph



f,ax=plt.subplots(1,2,figsize=(20,10))

flt_data_copy['Status'].value_counts().plot.pie(explode=[0,0.1,0.1,0.1],autopct='%1.2f%%',ax=ax[1],shadow=False)

                                                       

ax[1].set_title('% value of Deprture Delay Status')

ax[1].set_ylabel('')

sns.countplot('Status',order = flt_data_copy['Status'].value_counts().index, data=flt_data_copy,ax=ax[0])

ax[0].set_title('Number of Departure Delay Status')

plt.show()

# Arrival_Delay status classification



flt_data_copy.loc[flt_data_copy['arr_delay'] <= 0, 'Status'] = 0

flt_data_copy.loc[flt_data_copy['arr_delay'] >= 1, 'Status'] = 1

flt_data_copy.loc[flt_data_copy['arr_delay'] >= 15, 'Status'] = 2

flt_data_copy.loc[flt_data_copy['arr_delay'] >= 60, 'Status'] = 3



## Arrival_delay_status_graph



f,ax=plt.subplots(1,2,figsize=(20,10))



flt_data_copy['Status'].value_counts().plot.pie(explode=[0,0.1,0.1,0.1],autopct='%1.2f%%',ax=ax[1],shadow=False)

ax[1].set_title('% Value of Arrival Delay Status')

ax[1].set_ylabel('')

sns.countplot('Status',order = flt_data_copy['Status'].value_counts().index, data=flt_data_copy,ax=ax[0])

ax[0].set_title('Number of Arrival Delay Status')

plt.show()

Delayedflights = flt_data_copy[(flt_data_copy.Status >= 1) &(flt_data_copy.Status <= 3)]
#Average arrival delay by carrier

f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('carrier','arr_delay', data=Delayedflights,ax=ax[0], order=['F9', 'FL', 'EV', 'YV','OO','MQ','WN','B6',

                                                                                '9E','UA', 'US','VX','DL', 'AA','HA','AS'])

                                                                                 

ax[0].set_title('Average Arrival Delay by Carrier')

sns.boxplot('carrier','arr_delay', data=Delayedflights,ax=ax[1], order=['F9', 'FL', 'EV', 'YV','OO','MQ','WN','B6',

                                                                                '9E','UA', 'US','VX','DL', 'AA','HA','AS'])

ax[1].set_title('Arrival Delay Distribution by Carrier')

plt.close(2)

plt.show()



print(['WN: Southwest Airlines', 'AA: American Airlines', 'MQ: American Eagle Airlines', 'UA: United Airlines',

       'OO: Skywest Airlines','DL: Delta Airlines','US: US Airways','YV: Mesa Airlines', 'HA: Hawaiian Airlines',

       'EV: Atlantic Southeast Airlines','FL: AirTran Airways','F9: Frontier Airlines','VX: Virgin America',

       'B6: JetBlue Airways','9E: Pinnacle Airlines','AS: Alaska Airlines'])



print('\n')

print('The top 5 US airline (American Airlines (AA), Southwest Airlines (WN), Delta Air Lines (DL), United Airlines (UA), Alaska Airlines (AS)generate an average delay of 37.2 minutes. Alaska Airlines, with an 34 minutes per flight, the second lowest of all the carriers.')

print('\n')

print('Carriers with higher average delay generation are Skywest Airlines(OO) with 60 minutes per flight,  Mesa Airlines (YV) with 50 minutes per flight, and Pinnacle Airlines (9E) with 49 minutes per flight. The error bar provide the insight that airlines with low number of flights having higher standard deviation distribution from the mean  (OO, HA, YV, F9, AS); so it seems like size matters.')

print('\n')

print('The boxplot shows, airlines with higher number of flights results having a higher chance of extreme waiting situation. American Eagle Airlines (MQ), American Airlines(AA), Delta Airline(DL) registered the maximum Carrier Delay for 2013 with an exception of Hawaiian Airlines (HA).')
#Average departure delay by carrier



f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('origin','dep_delay', data=Delayedflights,ax=ax[0], order=['EWR', 'JFK', 'LGA'])

                                                                                 

ax[0].set_title('Airport Average Departure Delay')

sns.boxplot('origin','dep_delay', data=Delayedflights,ax=ax[1], order=['EWR', 'JFK', 'LGA'])

ax[1].set_title('Airport Delay Distribution')

plt.close(2)

plt.show()





print('There seems to be a correlation between the number of flights operated and departure delay. A descending pattern can be seen from barplot for average departure delay per flight. Considering, the assumption that JFK being a busiest airport among 3 due to international flights; so the maximum departure delay for 2013 is registered by JFK.')
scheduled_departure = flight_data.count()['sched_dep_time'] 

actual_departure = flight_data.count()['dep_time']  

cancel_departure = scheduled_departure - actual_departure

ratio_oper = actual_departure / scheduled_departure * 100



ratio_cancel = 100 - ratio_oper

                                            

print("Sched_dep: ", scheduled_departure)

print("Operated: ", actual_departure)

print("Cancelled: ", cancel_departure)

print("\n")

print("Ratio operated flights over scheduled flights: %s" % ratio_oper)

print("Ratio of cancelled flights: %s" % ratio_cancel)
# The day and month having highest average delay departure by 



Delay_Day=flt_data_copy.groupby(['day','month'], as_index=False).agg({'dep_delay': 'mean'})

Delay_Day_max=Delay_Day['dep_delay'].max()

Delay_Day_info=Delay_Day[Delay_Day['dep_delay']==Delay_Day_max]

print("The day and month having highest delay by average for departures" '\n' ,Delay_Day_info)
# The day and month having highest number of flight delay



max_flightdelay_day=flt_data_copy[flt_data_copy['dep_delay'] > 0].groupby(['day','month'], as_index=False).agg({'flight': 'count'})

max_flightdelay_info = max_flightdelay_day[max_flightdelay_day['flight'].max() == max_flightdelay_day['flight']]

print("Day and month which have highest number of flight delay" '\n' ,max_flightdelay_info)
# histogram for arrival and departure delay



f,ax=plt.subplots(1,2,figsize=(20,8))

sns.distplot(Delayedflights['arr_delay'], ax=ax[0])



ax[0].set_title('Arrdelay_Status')



sns.distplot(Delayedflights['dep_delay'], ax=ax[1])

ax[1].set_title('Depdelay_Status')

plt.show()



print("Skewness_arr: %f" % Delayedflights['arr_delay'].skew())

print("Kurtosis_arr: %f" % Delayedflights['arr_delay'].kurt())

print("Skewness_dep: %f" % Delayedflights['dep_delay'].skew())

print("Kurtosis_dep: %f" % Delayedflights['dep_delay'].kurt())

print('\n')

print('It can be seen on the histogram and by the skewness and kurtosis indexes, that the skewness is >1 which reflect the data distribution is highly positively skewed and Kurtosis>3 shows the leptokurtic distribution, having longer and fatter tail with a central peak higher and sharper.') 

print('\n')

print('The histogram shows the delays are mostly located on the left side of the graph, with a long tail to the right. The majority of delays are short, and the longer delays, while unusual, are more heavy loaded in time.')
# Top5 destination where flight arrives early than expected arrival time



Ontime_Early_flights =flt_data_copy[(flt_data_copy.Status <= 0)]



top10flight=(Ontime_Early_flights.sort_values(['arr_delay'], ascending=True)).head(10)

top10flight_details=pd.DataFrame(top10flight, columns=['day','month','origin','dest','arr_delay','carrier','flight','tailnum','distance','flight_speed'])

print("Top 5 destinations where flights arrives early than expected arrival time",'\n')





sns.lmplot(x = 'dest', y='arr_delay', data = top10flight_details, fit_reg=False, hue="flight")



plt.grid(True, color='k')



plt.show()

# Best airport in terms of early departure from NYC



airport_info = pd.DataFrame(flt_data_copy,columns=['day','month','dep_delay','arr_delay','carrier','origin','dest','flight'])

airport_origin = airport_info[airport_info['dep_delay']<0]

best_airport = airport_origin.sort_values(['dep_delay']).groupby(['origin']).agg({'dep_delay':'mean'})

best_airport.plot(kind='bar',color='purple', title ="Best Airport in terms of early departure",figsize=(5,5),legend=True, fontsize=12)

plt.show()

plt.close()
# Line Graph for Avg_Monthly delay of a/c  and Number of minutes delayed by month (excluding staus 0)

f,ax=plt.subplots(1,2,figsize=(20,8))

Delayedflights[['month','arr_delay','dep_delay']].groupby(['month']).mean().plot(ax=ax[0],marker='*', linestyle='dashed',color ='g'+'r')

ax[0].set_title('Avg_Monthly delay ')

Delayedflights[['month','arr_delay','dep_delay']].groupby(['month']).sum().plot(ax=ax[1], marker='*', linestyle='dashed',color ='g'+'r')

ax[1].set_title('Number of minutes delayed by Month')

plt.show()
# Monthly_Delays of Ontime_Early_ Flights (only for status 0)



f,ax=plt.subplots(1,2,figsize=(20,8))

Ontime_Early_flights[['month','arr_delay','dep_delay']].groupby(['month']).mean().plot(ax=ax[0],marker='*', linestyle='dashed',color ='g'+'r')

ax[0].set_title('Avg_Monthly_ Minutes for Ontime_Early Flights for NYC airport ')





Ontime_Early_flights[['month','arr_delay','dep_delay']].groupby(['month']).sum().plot(ax=ax[1], marker='*', linestyle='dashed',color ='g'+'r')

ax[1].set_title('Cumulative Monthly Minutes for Ontime_Early Flights for NYC airport')

plt.grid(True, color='g',linewidth='0.2')

plt.show()
# Flight Speed Analysis 



flight_speed =flt_data_copy['distance'] / (flt_data_copy['air_time']/60)

flt_data_copy['flight_speed'] =flight_speed

flt_data_copy.sort_values(by='flight_speed', ascending=False).head(5)
# Top5 fastest flights details from NYC



speed_5=flt_data_copy.loc[:, ['flight', 'tailnum','distance','air_time','flight_speed']].sort_values(by='flight_speed',ascending=False,axis=0).head(5)

sns.lmplot(x = 'distance', y='flight_speed', data = speed_5, fit_reg=False, hue="flight")

plt.grid(True, color='g', linewidth=0.1)

plt.title("Top5 fastest flights from NYC")

plt.show()

speed_5
Carrier_hmap=flt_data_copy.drop(['dep_time','sched_dep_time','arr_time','sched_arr_time','hour','minute','DATE',

                              'time_hour','month','year','origin','dest','tailnum','Status',

                                 'distance','flight','air_time'], axis=1)

Carrier_hmap.head()
Carrier_hmap_OO = pd.pivot_table(Carrier_hmap,values='flight_speed', aggfunc='mean', index='carrier',columns='MONTH_desc')

Carrier_hmap_OO.head()



plt.figure(figsize=(15,10))

sns.heatmap(Carrier_hmap_OO,annot=True, fmt=".1f")

plt.show()

print('The peak season for air travel in USA is considered to be June to August  and lean season is mid of January to February. The airlines operate highest number of flights and carry maximum PAX load during the summer season and vis-à-vis during lean season. The data proves that the statement is true and most of the airlines having maximum departure between May to August and minimum between January to February. From, the heatmap, it is visible that during May to August most of the airlines tend to fly faster than normal flight speed, to cover maximum departure. Whereas, it is vis-à-vis during lean season.')
Airlines_info= sns.pairplot(flt_data_copy, height=3,

                            vars=['dep_delay','arr_delay','distance','air_time','flight_speed'], hue='carrier', palette="Paired")

plt.show(Airlines_info)
flt_corr= flt_data_copy.drop(['dep_time','sched_dep_time','arr_time','sched_arr_time','hour','minute',

                              'time_hour','DATE','MONTH_desc','year','flight'], axis=1)

corrmat = flt_corr.corr()

f, ax = plt.subplots(figsize=(12, 9))



sns.heatmap(corrmat, square=True, cmap="RdYlGn",linewidth=1,annot=True,fmt='.2f');

plt.show()







print('The coorelation matrix provide the result that the departure and arrival delay are positively and strongly coorelated. Similarly with air time and flight distance; So they tend to move in the same direction. Whereas  correlation between fightspeed, airtime, flightdistance are positive but not so strong enough to influence each other.')
print('END')