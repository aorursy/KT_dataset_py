import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling    
from collections import Counter
#my_local_path="C:/Users/xsparamesh/OneDrive - NESTLE/UPX files/attachment_Project_Datasets/Project Datasets/attachment_Project_1_NYC-Flight_data/Project_1_NYC-Flight data/"

flight_data = pd.read_csv('../input/flight_data.csv')
print(flight_data.info())
flight_data.iloc[:20,:]

by_month_depdelay=flight_data.groupby(['month','origin']).mean()

def depdelay_val(series):
    return series.fillna(series.mean())
flight_data.dep_delay=flight_data['dep_delay'].transform(depdelay_val)
flight_data.iloc[310920:310958,:]

Delay_months = flight_data.groupby('month').mean()
Delay_months

Delay_months1 = pd.pivot_table(flight_data, index = ['flight'], values=('dep_delay'), columns = ['month'],aggfunc= np.sum)
delay_month=Delay_months1.sum(axis=0)
delay_month

plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
Delay_months['dep_delay'].plot(kind='bar',color='blue',grid=display)
plt.title('Average departure delay')
plt.xlabel('Months')
plt.ylabel('Average delay in a month (in minutes)')
plt.subplot(2,2,2)
delay_month.plot(kind='bar',color='red',grid=display)
plt.title('Total departure delay')
plt.xlabel('Months')
plt.ylabel('Total delay in a month (in minutes)')

%matplotlib inline
Delay_by_origin = pd.pivot_table(flight_data, index = ['month'], values=('dep_delay'), columns = ['origin'],aggfunc= np.mean)
print(Delay_by_origin)
plt.figure(figsize=(20,20))
Delay_by_origin[['EWR', 'LGA', 'JFK']].plot().legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.xlabel('Months')
plt.ylabel('Average delay in a month (in minutes)')
plt.title('Delay by Origin')

plt.figure(figsize=(20,10))
flights_sum_delay = pd.pivot_table(flight_data,index=["carrier"], values=('dep_delay'), columns = ['month'],aggfunc= np.sum)
plt.subplot(2,2,1)
plt.title("Total delay")
sns.heatmap(flights_sum_delay, annot=False)
flights_mean_delay = pd.pivot_table(flight_data,index=["carrier"], values=('dep_delay'), columns = ['month'],aggfunc= np.mean)
#print(flights_mean_delay)
plt.subplot(2,2,2)
plt.title("Average delay")
sns.heatmap(flights_mean_delay, annot=True)



plt.figure(figsize=(20,30))
sns.lmplot('dep_delay','month',data=flight_data, fit_reg = False, hue='origin')
result = flight_data.sort_values(['distance'])
#result1 = pd.DataFrame(result['air_time'].fillna(method='bfill'))
filled_flight_data = result
filled_flight_data

fill_airtime = pd.DataFrame(filled_flight_data['air_time'].interpolate(method='linear'))
#fill_airtime.to_csv(my_local_path+'fill_airtime.csv')
filled_flight_data['air_time']=fill_airtime['air_time'].values

flight_time_hours = filled_flight_data.air_time / 60
filled_flight_data['air_time_hrs']=pd.Series(flight_time_hours)
filled_flight_data
speed = filled_flight_data.distance / filled_flight_data.air_time_hrs
speed.round(0)
filled_flight_data['flight_speed']=pd.Series(speed.round(0))
speed_analysis = filled_flight_data
speed_analysis

speed_analysis[['flight_speed']]=speed_analysis[['flight_speed']].fillna('160')
#speed_analysis.to_csv(my_local_path+'speed_analysis1.csv')
speed_analysis_filled=speed_analysis
#speed_analysis['flight_speed'].describe()
speed_analysis_filled

carrier_speed_df = speed_analysis_filled.groupby(['carrier']).mean()
carrier_speed_df
carrier_speed = carrier_speed_df.distance / carrier_speed_df.air_time_hrs
t = carrier_speed.mean(axis=0)
carrier_speed_df['carrier_speed']=pd.Series(carrier_speed.round(0))
carrier_speed_df
#carrier_speed_series = carrier_speed['average_speed'].mean
#print(carrier_speed_series)
#average_speed_all_carriers = carrier_speed_series(axis=0)
#print(average_speed_all_carriers)
print('Average speed ',t)
plt.figure(figsize=(10,5))
carrier_speed_df['carrier_speed'].plot(kind='bar',color='blue',grid=display,legend='best')
plt.axhline(y=t,color = 'r')
plt.ylabel('Average speed in Miles per hour')
plt.title('Speed analysis(2013)')
#plt.axvline(y=t, linewidth=2, color = 'g')

speed_flight = carrier_speed > t
speed_flight

flight_dest = speed_analysis_filled['dest'].value_counts()
plt.figure(figsize=(25,10))
flight_dest.plot(kind='bar')
#Esns.barplot(flight_dest)
plt.xlabel('Destination')
plt.ylabel('Frequency')
plt.title('Number of flights headed towards a destination in 2013')
print(flight_dest.max() ,'flights headed to ORD ')
print(flight_dest.min() ,'flight headed to LEX ')
speed_analysis.sort_values(['time_hour'])

def arr_delay_val(series3):
    return series3.fillna(series3.mean())
speed_analysis.arr_delay=speed_analysis['arr_delay'].transform(arr_delay_val)

%matplotlib inline
arr_Delay_by_origin = pd.pivot_table(speed_analysis, index = ['month'], values=('arr_delay'), columns = ['origin'],aggfunc= np.mean)
#print(arr_Delay_by_origin)

arr_Delay_by_origin[['EWR', 'LGA', 'JFK']].plot().legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.xlabel('Months')
plt.ylabel('Average arrival delay in a month (in minutes)')
plt.title('Arrival Delay by Origin')

Delay_by_origin[['EWR', 'LGA', 'JFK']].plot().legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.xlabel('Months')
plt.ylabel('Average departure delay in a month (in minutes)')
plt.title('Departure Delay by Origin')

mean_arrival_delay = speed_analysis['arr_delay'].mean()
arrival_sd = speed_analysis['arr_delay'].std()
arrival_max=speed_analysis['arr_delay'].max()
arrival_min=speed_analysis['arr_delay'].min()
print("Mean arrival delay  ",mean_arrival_delay)
print("Standard deviation  ",arrival_sd)
print("Maximum delay   ",arrival_max)
print("Manimum delay   ",arrival_min)
plt.hist(speed_analysis.arr_delay,range=(-100,250), bins = 70)
mean_arrival_delay = speed_analysis['arr_delay'].mean()
plt.axvline(x=mean_arrival_delay, linewidth=2, color = 'r')
plt.xlabel('arrival delay in minutes')
plt.ylabel('frequency')
plt.title('On Time arrival analysis')
plt.show()

speed_analysis_count4 = pd.Series(speed_analysis.arr_delay)
print('total number of flights  ',speed_analysis_count4.count())

speed_analysis_count = pd.Series(speed_analysis.arr_delay)
speed_analysis_count = speed_analysis_count[speed_analysis_count < 0]
print('\nnumber of flights with negative arrival delay  ', speed_analysis_count.count())
per_neg_delay = (speed_analysis_count.count())*(100/speed_analysis_count4.count())
print('percentage of flight having negative delay  ', per_neg_delay)

speed_analysis_count0 = pd.Series(speed_analysis.arr_delay)
speed_analysis_count0 = speed_analysis_count0[speed_analysis_count0 > 0]
print('\nnumber of flights with positive arrival delay  ', speed_analysis_count0.count())
per_pos_delay = (speed_analysis_count0.count())*(100/speed_analysis_count4.count())
print('percentage of flight having positive delay  ', per_pos_delay)

speed_analysis_count3 = pd.Series(speed_analysis.arr_delay)
speed_analysis_count3 = speed_analysis_count3[speed_analysis_count3 == 0]
print('\nnumber of flights which are right on time  ', speed_analysis_count3.count())
per_on_time = (speed_analysis_count3.count())*(100/speed_analysis_count4.count())
print('percentage of flight having delay less than average  ', per_on_time)

speed_analysis_count1 = pd.Series(speed_analysis.arr_delay)
speed_analysis_count1 = speed_analysis_count1[speed_analysis_count1 < mean_arrival_delay]
print('\nnumber of flights with arrival delay less than average delay  ', speed_analysis_count1.count())
per_neg_avg_delay = (speed_analysis_count1.count())*(100/speed_analysis_count4.count())
print('percentage of flight having delay less than average  ', per_neg_avg_delay)

speed_analysis_count2 = pd.Series(speed_analysis.arr_delay)
speed_analysis_count2 = speed_analysis_count2[speed_analysis_count2 > mean_arrival_delay]
print('\nnumber of flights with arrival delay greater than average delay  ', speed_analysis_count2.count())
per_pos_avg_delay = (speed_analysis_count2.count())*(100/speed_analysis_count4.count())
print('percentage of flight having delay less than average  ', per_pos_avg_delay)

