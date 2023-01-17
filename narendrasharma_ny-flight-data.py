import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk('/kaggle/input/flight_data.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

## To Make Change in File Just Local Path needs to be changed to applicable Path

flight_data=pd.read_csv('/kaggle/input/flight_data.csv')

flight_data
from datetime import datetime

flight_data['time_hour']= pd.to_datetime(flight_data['time_hour']) 

## Inserting Columns Day_week based upon day taken from time_hour variable. 

flight_data['flight_date'] = pd.to_datetime(flight_data[['year','month','day']], yearfirst=True)

flight_data['Day_Week']=flight_data.flight_date.dt.day_name()

#### Removing and adding the Column as Month Name ########################

flight_data['month1']=flight_data['flight_date'].dt.month_name()

##################Adding Column Avg Speed to Data Frame######################

flight_data['Avgspped']=flight_data['distance'] / (flight_data['air_time']/60)

flight_data['Avgspped']=flight_data.Avgspped.round(2)

flight_data_org=flight_data.copy()

flight_data_org.shape
# List of Records where dept_time is null and Departure delay is also null and Arrival time and arrival _delay is also null.There are 8255 such records 

### ALl These Records are taken in Different Cancelled

flight_Cancelled=flight_data[(flight_data["dep_time"].isnull())&(flight_data["dep_delay"].isnull())&(flight_data["arr_time"].isnull())&(flight_data["arr_delay"].isnull())]

### Taking index of Columns and Removing the values from Orginal data Frame

flight_data.drop(index=flight_Cancelled.index,inplace=True)
# Data Frame Created to Update the Values for Arrival Time

test1=pd.DataFrame(flight_data[flight_data["arr_time"].isnull()])

# Updating the Arrival time for Null as scheduled arrival time plus delay.

test1['arr_time'].fillna(test1['sched_arr_time']+test1['dep_delay'],inplace=True)
### Function Written to Validate the time in test data set to correct value 22:86 to be 23:26 and 2420 to 0020 as military standard time

def Validate_time(hours):

    no=hours

    minutes=no%100

    if(minutes>59):

         hours=(no - minutes)/100

         hours=hours*100

         hours+=100

         if(hours>=2400):hours=hours-2400

         hours=hours+(minutes-60)      

    else:

        if(hours>=2400):hours=hours-2400

    return int(hours)
## Applying the validate time function for arr_time with axis=1 for each row 

test1['arr_time'] = test1.apply(lambda row : Validate_time(row['arr_time']), axis = 1) 

#flight_data[flight_data["arr_time"].isnull()]
## Updating the Data using the arr_time updated values with values from test data frame 

flight_data['arr_time'].fillna(value=test1['arr_time'],inplace=True)

## Updating the Arrival delay from temp datafrome where values are updated.

flight_data['arr_delay'].fillna(value=flight_data['arr_time'] - flight_data['sched_arr_time'],inplace=True)

### Updating airtime as 65% of difference between arrival and departure time.

flight_data['air_time'].fillna(value=round((flight_data['arr_time'] - flight_data['dep_time'])*.65),inplace=True)
###########Function Created to Drop Data Frame to Free up Memory #############

def Drop_dataframe(df):

    df.drop(df.index, inplace=True)

    df.drop(columns=df.columns,inplace=True)

    return (df.shape)
############Departure Delays#################

flight_data_new0=flight_data[flight_data['dep_delay']!=0]

flight_data_new1=flight_data[flight_data['dep_delay']>0]

flight_data_new2=flight_data[flight_data['dep_delay']==0]

flight_data_new3=flight_data[flight_data['dep_delay']<0]



test0=pd.DataFrame(flight_data_new0[['carrier','month1','dep_delay']].groupby(['carrier','month1']).mean())

test1=pd.DataFrame(flight_data_new1[['carrier','month1','dep_delay']].groupby(['carrier','month1']).count())

test2=pd.DataFrame(flight_data_new2[['carrier','month1','dep_delay']].groupby(['carrier','month1']).count())

test3=pd.DataFrame(flight_data_new3[['carrier','month1','dep_delay']].groupby(['carrier','month1']).count())



Temp1=pd.merge(test0,test1,how='outer',on=['carrier','month1'])

Temp2=pd.merge(test2,test3,how='outer',on=['carrier','month1'])

Temp3=pd.merge(Temp1,Temp2,how='outer',on=['carrier','month1'])



#############Arrvial Delays##########################################################################

flight_data_new4=flight_data[flight_data['arr_delay']!=0]

flight_data_new5=flight_data[flight_data['arr_delay']>0]

flight_data_new6=flight_data[flight_data['arr_delay']==0]

flight_data_new7=flight_data[flight_data['arr_delay']<0]



test4=pd.DataFrame(flight_data_new4[['carrier','month1','arr_delay']].groupby(['carrier','month1']).mean())

test5=pd.DataFrame(flight_data_new5[['carrier','month1','arr_delay']].groupby(['carrier','month1']).count())

test6=pd.DataFrame(flight_data_new6[['carrier','month1','arr_delay']].groupby(['carrier','month1']).count())

test7=pd.DataFrame(flight_data_new7[['carrier','month1','arr_delay']].groupby(['carrier','month1']).count())



Temp4=pd.merge(test4,test5,how='outer',on=['carrier','month1'])

Temp5=pd.merge(test6,test7,how='outer',on=['carrier','month1'])

Temp6=pd.merge(Temp4,Temp5,how='outer',on=['carrier','month1'])

##############Merging dataframe in one############################################################

sum_flight=pd.merge(Temp3,Temp6,how='outer',on=['carrier','month1'])

#Temp5.rename(columns = {'dep_delay_x':'dep_delay','dep_delay_y':'dep_intime','dep_delay':'dep_ahead_time'}, inplace = True)

#########Renaming Columns for Meaningfull Name

sum_flight.rename(columns = {'arr_delay_y_x':'arr_delay','arr_delay_x_y':'arr_intime','arr_delay_y_y':'arr_ahead_time','arr_delay_x_x':'avg_arr_delay'}, inplace = True)

sum_flight.rename(columns = {'dep_delay_x_x':'avg_dep_delay','dep_delay_y_x':'dep_delay','dep_delay_x_y':'dep_intime','dep_delay_y_y':'dep_ahead_time'}, inplace = True)

##########Rounding Values for Better Display

sum_flight['avg_dep_delay']=sum_flight.avg_dep_delay.round(2)

sum_flight['avg_arr_delay']=sum_flight.avg_arr_delay.round(2)

sum_flight.head()
##########Cleaning the Data frames ###############

#################Cleaning the Data frames which are not required later#############################

Drop_dataframe(flight_data_new0)

Drop_dataframe(flight_data_new1)

Drop_dataframe(flight_data_new2)

Drop_dataframe(flight_data_new3)

Drop_dataframe(flight_data_new4)

Drop_dataframe(flight_data_new5)

Drop_dataframe(flight_data_new6)

Drop_dataframe(flight_data_new7)

Drop_dataframe(test0)

Drop_dataframe(test1)

Drop_dataframe(test2)

Drop_dataframe(test3)

Drop_dataframe(test4)

Drop_dataframe(test5)

Drop_dataframe(test6)

Drop_dataframe(test7)

Drop_dataframe(Temp1)

Drop_dataframe(Temp2)

Drop_dataframe(Temp3)

Drop_dataframe(Temp4)

Drop_dataframe(Temp5)

Drop_dataframe(Temp6)

print("")
#### Handling Null Values as join was outer for Arrival time

sum_flight.fillna(0,inplace=True)

sum_flight.reset_index(level=1, drop=False,inplace=True)

sum_flight.reset_index(level=0, drop=False,inplace=True)

sum_flight.rename(columns = {'month1':'month'}, inplace = True)



########Converting Data Types to Index 

sum_flight['dep_delay'] = sum_flight['dep_delay'].astype(int)

sum_flight['dep_intime'] = sum_flight['dep_intime'].astype(int)

sum_flight['dep_ahead_time'] = sum_flight['dep_ahead_time'].astype(int)

sum_flight['arr_delay'] = sum_flight['arr_delay'].astype(int)

sum_flight['arr_intime'] = sum_flight['arr_intime'].astype(int)

sum_flight['arr_ahead_time'] = sum_flight['arr_ahead_time'].astype(int)
############Departure Delays Day Wise #################

flight_data_new0=flight_data[flight_data['dep_delay']!=0]

flight_data_new1=flight_data[flight_data['dep_delay']>0]

flight_data_new2=flight_data[flight_data['dep_delay']==0]

flight_data_new3=flight_data[flight_data['dep_delay']<0]



test0=pd.DataFrame(flight_data_new0[['carrier','Day_Week','dep_delay']].groupby(['carrier','Day_Week']).mean())

test1=pd.DataFrame(flight_data_new1[['carrier','Day_Week','dep_delay']].groupby(['carrier','Day_Week']).count())

test2=pd.DataFrame(flight_data_new2[['carrier','Day_Week','dep_delay']].groupby(['carrier','Day_Week']).count())

test3=pd.DataFrame(flight_data_new3[['carrier','Day_Week','dep_delay']].groupby(['carrier','Day_Week']).count())



Temp1=pd.merge(test0,test1,how='outer',on=['carrier','Day_Week'])

Temp2=pd.merge(test2,test3,how='outer',on=['carrier','Day_Week'])

Temp3=pd.merge(Temp1,Temp2,how='outer',on=['carrier','Day_Week'])



#############Arrvial Delays Day Wise ##########################################################################

flight_data_new4=flight_data[flight_data['arr_delay']!=0]

flight_data_new5=flight_data[flight_data['arr_delay']>0]

flight_data_new6=flight_data[flight_data['arr_delay']==0]

flight_data_new7=flight_data[flight_data['arr_delay']<0]



test4=pd.DataFrame(flight_data_new4[['carrier','Day_Week','arr_delay']].groupby(['carrier','Day_Week']).mean())

test5=pd.DataFrame(flight_data_new5[['carrier','Day_Week','arr_delay']].groupby(['carrier','Day_Week']).count())

test6=pd.DataFrame(flight_data_new6[['carrier','Day_Week','arr_delay']].groupby(['carrier','Day_Week']).count())

test7=pd.DataFrame(flight_data_new7[['carrier','Day_Week','arr_delay']].groupby(['carrier','Day_Week']).count())

Temp4=pd.merge(test4,test5,how='outer',on=['carrier','Day_Week'])

Temp5=pd.merge(test6,test7,how='outer',on=['carrier','Day_Week'])

Temp6=pd.merge(Temp4,Temp5,how='outer',on=['carrier','Day_Week'])



wsum_flight=pd.merge(Temp3,Temp6,how='outer',on=['carrier','Day_Week'])

wsum_flight.rename(columns = {'dep_delay_x_x':'wek_avg_dep_delay','dep_delay_y_x':'wek_dep_delay','dep_delay_x_y':'wek_dep_intime','dep_delay_y_y':'wek_dep_ahead_time'}, inplace = True)

wsum_flight.rename(columns = {'arr_delay_x_x':'wek_avg_arr_delay','arr_delay_y_x':'wek_arr_delay','arr_delay_y_y':'wek_arr_ahead_time','arr_delay_x_y':'wek_arr_intime'}, inplace = True)

### Updating Null Values 

wsum_flight.fillna(0,inplace=True)

wsum_flight['wek_avg_dep_delay']=wsum_flight.wek_avg_dep_delay.round(2)

wsum_flight['wek_avg_arr_delay']=wsum_flight.wek_avg_arr_delay.round(2)



######Converting Columns to Int######################

wsum_flight['wek_dep_delay'] = wsum_flight['wek_dep_delay'].astype(int)

wsum_flight['wek_dep_intime'] = wsum_flight['wek_dep_intime'].astype(int)

wsum_flight['wek_arr_delay'] = wsum_flight['wek_arr_delay'].astype(int)

wsum_flight['wek_arr_intime'] = wsum_flight['wek_arr_intime'].astype(int)

wsum_flight['wek_arr_ahead_time'] = wsum_flight['wek_arr_ahead_time'].astype(int)

wsum_flight.head()
#### Updating the Index Columns to We have Weekday as column and not as index

wsum_flight.reset_index(level=1, drop=False,inplace=True)

wsum_flight.reset_index(level=0, drop=False,inplace=True)
#################Cleaning the Data frames which are not required later#############################

Drop_dataframe(flight_data_new0)

Drop_dataframe(flight_data_new1)

Drop_dataframe(flight_data_new2)

Drop_dataframe(flight_data_new3)

Drop_dataframe(flight_data_new4)

Drop_dataframe(flight_data_new5)

Drop_dataframe(flight_data_new6)

Drop_dataframe(flight_data_new7)

Drop_dataframe(test0)

Drop_dataframe(test1)

Drop_dataframe(test2)

Drop_dataframe(test3)

Drop_dataframe(test4)

Drop_dataframe(test5)

Drop_dataframe(test6)

Drop_dataframe(test7)

Drop_dataframe(Temp1)

Drop_dataframe(Temp2)

Drop_dataframe(Temp3)

Drop_dataframe(Temp4)

Drop_dataframe(Temp5)

Drop_dataframe(Temp6)

print("")
###########Updating the Flight Cancelled Dataframe inspiration was Virat for the Code ################

flight_Cancelled['dep_time'].fillna(0.00, inplace=True)

flight_Cancelled['dep_delay'].fillna(0.00, inplace=True)

flight_Cancelled['arr_time'].fillna(0.00, inplace=True)

flight_Cancelled['arr_delay'].fillna(0.00, inplace=True)

flight_Cancelled['air_time'].fillna(.00, inplace=True)

#flight_Cancelled.loc[(flight_Cancelled['dep_time'] == 0.00) & (flight_Cancelled['dep_delay'] == 0.00) & (flight_Cancelled['arr_time'] == 0.00) & (flight_Cancelled['arr_delay'] == 0.00) & (flight_Cancelled['air_time'] == 0.00),'flight_sattus'] = 'Cancelled'

flight_Cancelled['flight_sattus']='Cancelled'

flight_Cancelled['tailnum'].fillna('Unknown', inplace=True)

flight_Cancelled.isnull().sum()

#########Creating Dataframe for Updating the Cancelled Summary Starting with adding column

############ Two Data Frames are Defined One for Monthly Cancellation and Another for Day Wise Cancellation

test1=pd.DataFrame(flight_Cancelled.groupby(['carrier','month1','flight_sattus'],observed=False).count())

test2=pd.DataFrame(flight_Cancelled.groupby(['carrier','Day_Week','flight_sattus'],observed=False).count())

test1.rename(columns = {'year':'Cancelled_Flight'},inplace=True)

test2.rename(columns = {'year':'Cancelled_Flight'},inplace=True)



#######################=Droping of Unwanted Columns ####################

test1=test1.drop(['month', 'day', 'dep_time', 'sched_dep_time', 'dep_delay','arr_time', 'sched_arr_time', 'arr_delay', 

'flight', 'tailnum', 'origin', 'dest', 'air_time', 'distance', 'flight_date', 'Avgspped','hour', 'minute','time_hour', 'Day_Week'],axis=1)



test2=test2.drop(['month', 'day', 'dep_time', 'sched_dep_time', 'dep_delay','arr_time', 'sched_arr_time', 'arr_delay', 

'flight', 'tailnum', 'origin', 'dest', 'air_time', 'distance','flight_date', 'Avgspped', 'hour', 'minute','time_hour','month1'],axis=1)



###############Droping Indexes and setting them as normal columns

test1.reset_index(level=2, drop=False,inplace=True)

test1.reset_index(level=1, drop=False,inplace=True)

test1.reset_index(level=0, drop=False,inplace=True)



test2.reset_index(level=2, drop=False,inplace=True)

test2.reset_index(level=1, drop=False,inplace=True)

test2.reset_index(level=0, drop=False,inplace=True)



######################Dropping Columns which are not required####################################

test1.drop('flight_sattus',axis=1,inplace=True)

test2.drop('flight_sattus',axis=1,inplace=True)



#############renaming Columns===================

test1.rename(columns = {'month1':'month'},inplace=True)
### Merging Data Frames to Get Consolidated Output . 

#Temp7=pd.merge(sum_flight,cancell_sum1,how='outer',on=['carrier','month'])

sum_flight=pd.merge(sum_flight,test1,how='outer',on=['carrier','month'])

#Temp8= pd.merge(wsum_flight,cancell_sum2,how='outer',on=['carrier','Day_Week'])

wsum_flight=pd.merge(wsum_flight,test2,how='outer',on=['carrier','Day_Week'])
########(185, 11) (112, 11) Validate Sum_flight has 191 and 11columns and wsum_flights has 112 rows and 11 Columns

print(sum_flight.shape,wsum_flight.shape)

wsum_flight.head()
sum_flight.head()
###########Updating Null Values##############

sum_flight['Cancelled_Flight'].fillna(0,inplace=True)

wsum_flight['Cancelled_Flight'].fillna(0,inplace=True)

sum_flight.fillna(0,inplace=True)
###########Inserting New Columns to Get the Total Count of Flights based upon condition validated.

sum_flight['Total_Flights']=sum_flight['arr_delay'] + sum_flight['arr_intime']+ sum_flight['arr_ahead_time']+sum_flight['Cancelled_Flight']

wsum_flight['Total_Flights']=wsum_flight['wek_dep_delay'] + wsum_flight['wek_dep_intime']+ wsum_flight['wek_dep_ahead_time']+wsum_flight['Cancelled_Flight']
#cancell_sum1=pd.DataFrame(flight_Cancelled1.groupby(['carrier','month1','flight_sattus'],observed=False).count())

airport_analysis=pd.DataFrame(flight_data[['month1','carrier','flight','tailnum','dep_delay','arr_delay','origin','dest']])
###########Function created to Update the Status of Row Based upon input condition for arrival and Departure Status.

def Update_Status(row):

    if row == 0 :val= 'Ontime'

    elif row< 0 :val= 'Ahdtime'

    elif row> 0 :val= 'Delay'

    else: val=""

    return val
## Inserting New Column Dep Status and Arrival Status and Updating Values inside it.

airport_analysis['Dep_Status']=airport_analysis.apply(lambda row : Update_Status(row['dep_delay']), axis = 1) 

airport_analysis['Arr_Status']=airport_analysis.apply(lambda row : Update_Status(row['arr_delay']), axis = 1) 

airport_analysis.rename(columns = {'month1':'month'},inplace=True)
#### Data Frames are Crated to Store the Departure Summary with Few Columns

dep_sum=pd.DataFrame(airport_analysis.groupby(['month','carrier','origin','Dep_Status',],observed=True).count())

dep_sum=dep_sum.drop(['tailnum', 'dep_delay','Arr_Status', 'arr_delay','dest'],axis=1)

#dep_sum.reset_index(level=4, drop=False,inplace=True)

dep_sum.reset_index(level=3, drop=False,inplace=True)

dep_sum.reset_index(level=2, drop=False,inplace=True)

dep_sum.reset_index(level=1, drop=False,inplace=True)

dep_sum.reset_index(level=0, drop=False,inplace=True)

dep_sum.rename(columns = {'flight':'Total_Dep'},inplace=True)

dep_sum.shape
#############Arrival status Flights  ########################################

arr_sum=pd.DataFrame(airport_analysis.groupby(['month','carrier','dest','Arr_Status'],observed=True).count())

arr_sum=arr_sum.drop(['tailnum', 'dep_delay','arr_delay','Dep_Status','origin'],axis=1)

arr_sum.reset_index(level=3, drop=False,inplace=True)

arr_sum.reset_index(level=2, drop=False,inplace=True)

arr_sum.reset_index(level=1, drop=False,inplace=True)

arr_sum.reset_index(level=0, drop=False,inplace=True)

arr_sum.rename(columns = {'flight':'Total_Arr'},inplace=True)
############Maximum No of Flights Headed for Some particular Destination.==>#############

#flight_data_org

airport_dep_sum=pd.DataFrame(airport_analysis.groupby(['origin','dest']).count())

airport_dep_sum.reset_index(level=1, drop=False,inplace=True)

airport_dep_sum.reset_index(level=0, drop=False,inplace=True)

airport_dep_sum=airport_dep_sum.drop(['carrier','flight','tailnum','dep_delay','arr_delay','Dep_Status','Arr_Status'],axis=1)

airport_dep_sum.rename(columns = {'month':'Count'},inplace=True)

print(airport_dep_sum.shape)

airport_dep_sum.head()

########This Query Gives the Exact Count of Data based 

airport_dep_sum.sort_values('Count',ascending=False).head()
## Importing important functions.

from scipy import stats, integrate

import matplotlib.pyplot as plt

import numpy as np

from scipy import stats, integrate

import seaborn as sns

plt.rcParams['font.size'] = 14

%matplotlib inline

image_path=r"E:\UPX\Project\attachment_Project_Datasets\Project Datasets\attachment_Project_1_NYC-Flight_data\Pictures/"
##########Departure Delays for Various Airlines. 

sns.scatterplot(x="carrier", y="dep_delay", data=sum_flight)

plt.title('Departure Delays for Various Airlines ')
#######################Plots to Show Average Arrival Delay in Arrival and Departure for Different Airlines################

flg,ax=plt.subplots(1,2,figsize=(14,8))

ax[1].set_title('Average Monthly Arrival Delay for Carriers')

ax[0].set_title('Average Monthly Departure Delay for Carriers')

sns.boxplot(x="carrier", y="avg_arr_delay", data=sum_flight, palette='rainbow',ax=ax[1])

sns.boxplot(x="carrier", y="avg_dep_delay", data=sum_flight, palette='rainbow',ax=ax[0])

plt.show()
plt.tight_layout()

g=sns.catplot(x="carrier", y="wek_avg_dep_delay", data=wsum_flight,height=8, kind="bar",hue='Day_Week',legend=True)

g.fig.suptitle('Departure Delay Day wise for Different Carrier')

plt.plot()
plt.tight_layout()

g=sns.catplot(x="carrier", y="wek_avg_arr_delay", data=wsum_flight,height=8, kind="bar",hue='Day_Week',legend=True)

g.fig.suptitle('Arrival Delay Day wise for Different Carrier')

plt.plot()
######## Non Punctual Flights or the One Not Departed on Time

plt.title('Flights Which have not Departed on Time')

sns.lineplot(x="carrier", y="dep_delay",  data=sum_flight)

### Temp test1 to create to visualise departure summary report for 3 airports

test1=pd.DataFrame(dep_sum[['origin','Dep_Status','Total_Dep']].groupby(['origin','Dep_Status']).sum())

test1.reset_index(level=1, drop=False,inplace=True)

test1.reset_index(level=0, drop=False,inplace=True)

plt.title("Departure Summary Status for three Airport")

sns.lineplot(x='origin',y='Total_Dep',data=test1,hue='Dep_Status')

test2=test1[test1['Dep_Status']=='Ontime']
##### Plots Showing Deleyed Departed and Delayed Arrived Flights

flg,ax=plt.subplots(1,2,figsize=(12,6))

ax[0].set_title('Flight Departured with Delay ')

ax[1].set_title('Flight Arrived Late ')

sns.distplot(sum_flight.dep_delay,kde=True,ax=ax[0])

sns.distplot(sum_flight.arr_delay,kde=True,ax=ax[1])
## Company Wise Arrival Delay 

fig=plt.figure()

#plt.title('Average Arrival Delay for Airlines ')

sns.catplot(x="carrier", y="avg_arr_delay", data=sum_flight, kind="bar")

#sns.catplot(x="carrier", y="avg_dep_delay", data=sum_flight, kind="bar")
##### Plots Showing the Average Departure Delays for Airlines across different Month

#fig = plt.figure()

#plt.title('Average Departure Delay Month wise for Airlines')

sns.catplot(x="carrier", y="avg_dep_delay",hue='month' ,data=sum_flight)

plt.title("On Time Departure for Various Airports")

sns.countplot(x='Total_Dep',data=test2 ,hue='origin')
############Showing Total Flights vs Cancelled Flights 

#est1=pd.DataFrame(sum_flight[['Total_Flights','dep_delay','arr_delay','Cancelled_Flight','month']].groupby('month').sum())

test1=pd.DataFrame(sum_flight[['dep_delay','arr_delay','Cancelled_Flight','carrier']].groupby(['carrier']).sum())

plt.figure(figsize=(12,6))

plt.title("Cancelled and Arrived and Departed Late Flights for Months in 2013")

plt.ylabel("Airlines")

sns.heatmap(test1,cmap="YlGnBu",annot=True,fmt='g',linewidths=.06,cbar=False)

#plt.savefig(image_path+"Heat Map of Departure Delays and Cancelled Flights.png")
########### Ploting Total Flights in Bar and Pie Charts. 

flg,ax=plt.subplots(1,2,figsize=(12,6))

label1=flight_data['carrier'].unique().tolist()

plt.pie(flight_data['carrier'].value_counts(),autopct='%1.2f%%',shadow=False,labels=label1)                                               

ax[1].set_title('Flight as Percent of Carrier')

sns.countplot('carrier',order = flight_data['carrier'].value_counts().index, data=flight_data,ax=ax[0])

ax[0].set_title('Total Flight Count Airline Wise')

ax[0].set_ylabel('Number of Flights')

#plt.show()
#########FLight Correlation Between Arrival and Departure for Differeng Airlines.

sns.relplot(x='dep_intime',y='arr_intime',data=sum_flight,hue='carrier')

plt.title('Flights Which have Departed on Time and Arrived on Time')
#### Ploting Linear Regerrion plot between Flight Arrived and Departed for Different Airlines

sns.lmplot(x='dep_delay',y='arr_delay',data=sum_flight,fit_reg=False,hue='carrier',aspect=1.5,height=6)

###### Time Arrival % Analysis################

plt.title("Airlines Arrival and Delay Arrival Status")

sns_plot=sns.swarmplot(x='carrier',y='Total_Arr',data=arr_sum,hue='Arr_Status')

#sns_plot.savefig(image_path+"Arrival_Delays.png")

#plt.savefig(image_path+"Arrival_Delays.png")
plt.title("Airlines Departure and their Delay Status")

sns.swarmplot(x='carrier',y='Total_Dep',data=dep_sum,hue='Dep_Status')
plt.title("Departure Status for Airlines")

sns.lineplot(x='carrier',y='Total_Dep',data=dep_sum,hue='Dep_Status')

#plt.savefig(image_path+"Departure Status for Airlines.png")
plt.title("Arrival Status for Airlines")

sns.lineplot(x='carrier',y='Total_Arr',data=arr_sum,hue='Arr_Status')

#plt.savefig(image_path+"Arrival_Status_for_Airline.png")
plt.figure(figsize=(13,4))

g = sns.scatterplot(x='month',y='Cancelled_Flight',hue='carrier',data=sum_flight)

g.legend(loc='center right', bbox_to_anchor=(1.13, 0.4), ncol=1)

plt.title("Cancelled Flight Summary for Various Months")

#plt.savefig(image_path+"Cancelled Flight Summary monthwise .png")

plt.show()
plt.figure(figsize=(13,4))

g = sns.scatterplot(x='Day_Week',y='Cancelled_Flight',hue='carrier',data=wsum_flight)

g.legend(loc='center right', bbox_to_anchor=(1.13, 0.4), ncol=1)

plt.title("Cancelled Flight Summary for Various Days")

#plt.savefig(image_path+"Cancelled Flight Summary daywise.png")

plt.show()
sns.set_style("dark")

test1=airport_dep_sum.sort_values(by='Count',ascending=False).head(40)

plt.title("List of Maximum Destination for Three Airport Top 40 Records")

sns.swarmplot(x='dest',y='Count',hue='origin',data=test1,size=6)
###########Speed Analysis of Aircraft #############

#test1=pd.DataFrame(flight_speed[['carrier','Avgspped']].groupby('carrier').mean())

#test2=pd.DataFrame(flight_speed[['tailnum','Avgspped']].groupby('tailnum').mean())

test1=pd.DataFrame(flight_data[['carrier','Avgspped']].groupby('carrier').mean())

test2=pd.DataFrame(flight_data[['tailnum','Avgspped']].groupby('tailnum').mean())

test1['Avgspped']=test1.Avgspped.round(2)

test2['Avgspped']=test2.Avgspped.round(2)

test3=test2.head(10)

test4=test2.tail(10)
sns.lineplot(x=test1.index,y='Avgspped',data=test1,color='blue', linewidth=2.5)

#sns.lineplot(x=test2.index,y='Avgspped',data=test2,color='pink', linewidth=2.5)

plt.title("Average Speed of Companies ")

plt.figure(figsize=(10,4))

plt.title("Top 10 Aircraft in terms of Average Speed ")

sns.lineplot(x=test3.index,y='Avgspped',data=test3,color='red', linewidth=2.5,legend='full')
plt.figure(figsize=(10,4))

plt.title("Bottom 10 Aircraft in terms of Average Speed ")

sns.lineplot(x=test4.index,y='Avgspped',data=test4,color='blue', linewidth=2.5,legend='full')
#################Cleaning the Data frames which are not required later#############################

Drop_dataframe(flight_data_new0)

Drop_dataframe(flight_data_new1)

Drop_dataframe(flight_data_new2)

Drop_dataframe(flight_data_new3)

Drop_dataframe(flight_data_new4)

Drop_dataframe(flight_data_new5)

Drop_dataframe(flight_data_new6)

Drop_dataframe(flight_data_new7)

Drop_dataframe(test0)

Drop_dataframe(test1)

Drop_dataframe(test2)

Drop_dataframe(test3)

Drop_dataframe(test4)

Drop_dataframe(test5)

Drop_dataframe(test6)

Drop_dataframe(test7)

Drop_dataframe(Temp1)

Drop_dataframe(Temp2)

Drop_dataframe(Temp3)

Drop_dataframe(Temp4)

Drop_dataframe(Temp5)

Drop_dataframe(Temp6)