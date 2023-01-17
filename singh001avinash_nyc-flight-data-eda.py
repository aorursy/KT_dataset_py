import numpy as np
import pandas as pd
import pandas_profiling as pds
import matplotlib.pyplot as plt
import seaborn as sn
from bokeh.plotting import figure,show,output_notebook
from matplotlib import style
def convert (number):
        if ((len(str(number))>3)):
            return str(number)[0:2] + str(number)[2:]
        else:
            return "0" + str(number)[:1]+str(number)[1:]
import os
print(os.listdir("../input"))
flight_data=pd.read_csv("../input/flight_data.csv")
report= pds.ProfileReport(flight_data)
report.to_file("Flightdata_info_report.html")
dataset_f=pd.DataFrame(flight_data)
dataset_f= dataset_f.dropna()
print("The total number of rows and column in dataset after removing null value is", dataset_f.shape)
day_delay=dataset_f.groupby(['day','month'], as_index=False).agg({'dep_delay': 'mean'})
day_delay_max=day_delay['dep_delay'].max()
day_delay_info=day_delay[day_delay['dep_delay']==day_delay_max]
print(day_delay_info)
max_flightdealy_day=dataset_f[dataset_f['dep_delay'] > 0].groupby(['day','month'], as_index=False).agg({'flight': 'count'})
max_flightdealy_info = max_flightdealy_day[max_flightdealy_day['flight'].max() == max_flightdealy_day['flight']]
print("Day and month which have maximum number of flight delay" '\n' ,max_flightdealy_info)
month_delayinfo = dataset_f.groupby(['month'], as_index=False).agg({'dep_delay': 'mean'})
month_delayinfo['dep_delay']=np.round(month_delayinfo['dep_delay'],0)
sn.factorplot(x='month', y='dep_delay', data=month_delayinfo, kind='bar')
plt.plot()
plt.show()

airport_info_df=pd.DataFrame(dataset_f,columns=['day','month','dep_delay','arr_delay','carrier','origin','dest','flight'])
airport_info1=airport_info_df[airport_info_df['dep_delay']<1]
best_airport=airport_info1.sort_values(['dep_delay']).groupby(['origin']).agg({'dep_delay':'mean'})
best_airport.plot(kind='bar', title ="Best Airport in term of departure",figsize=(5,5),legend=True, fontsize=12)
plt.show()
plt.close()
%matplotlib inline

dataset_f['speed']=dataset_f['distance']/(dataset_f['air_time']/60)
fastest_flight_max=dataset_f.sort_values(['speed'], ascending=False)
fastest_flight_max['speed']=np.round(fastest_flight_max['speed'],1)
fastest_flight_max_top5=fastest_flight_max.head(5)
sn.lmplot(x = 'distance', y='speed', data = fastest_flight_max_top5,fit_reg=False, hue="tailnum")
plt.show()
fastest_flight_max_top5
carrier_info=dataset_f['carrier'].unique()


early_arrival=dataset_f[dataset_f['arr_delay']<=0]
top5carrier_info=(early_arrival.sort_values(['arr_delay'], ascending=True)).head(5)
top5carrier_info_details=pd.DataFrame(top5carrier_info, columns=['day','month','origin','dest','arr_delay','carrier','flight','tailnum','distance','speed'])
print("The details of top 5 flight info which arrives on destination",'\n')

plt.plot(top5carrier_info_details.dest,top5carrier_info_details.arr_delay,linestyle='--', color='red')
plt.grid(True, color='k')
plt.show()

carrier_arrival=early_arrival.sort_values(['arr_delay'], ascending=True).groupby(['month'],as_index=False).agg({'arr_delay':'mean'})
sn.factorplot(x='month', y='arr_delay', data=carrier_arrival)
plt.xticks(rotation=75)
plt.grid(True, color='b')
plt.legend

plt.show()

%matplotlib inline
NYC_allflight=dataset_f['dest'].unique()


NYC_allflightcount=len(NYC_allflight)
print("The total number unique destination flight  from NYC is"+ "::" ,NYC_allflightcount)
print('\n')

originwise_flight=dataset_f.groupby(['origin'])["dest"].count()
print("The following are the total count of destination flight from the following origion\n",originwise_flight)

NYC_allflight1=dataset_f.groupby(['dest'],as_index=False).agg({'month':'count'})
maximun_count_flight=NYC_allflight1.sort_values(['month'], ascending=False)
print("The top 5 Spot destination flight from NYC is")
top_destination_flight = maximun_count_flight.head(5)
plt.scatter(top_destination_flight.dest,top_destination_flight.month, color='red')
plt.legend
plt.grid(True, color='y')
plt.show()
top_destination_flight

ATL_dest=dataset_f[dataset_f['dest']=='ATL']
Airline_count=(ATL_dest['carrier']).unique()
print("The total number of Airline  headed to 'ATL' from NYC is",len(Airline_count))
print('\n')

flight_ATL_dest_count=ATL_dest['tailnum'].unique()
print("The total unique flight  headed to 'ATL' from NYC is",len(flight_ATL_dest_count))
print('\n')


