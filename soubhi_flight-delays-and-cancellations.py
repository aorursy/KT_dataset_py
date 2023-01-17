import numpy as np

import csv

import operator

import pandas as pd

import matplotlib.pyplot as plt

import pylab

%matplotlib inline

pylab.rcParams['figure.figsize'] = (16,  10)
airlines_data = pd.read_csv('../input/airlines.csv')

airports_data = pd.read_csv('../input/airports.csv')

flights_data = pd.read_csv('../input/flights.csv')

#Not able to load all the data because of the limited resources. The final result will not be the

#same for the whole dataset

flights_data=flights_data[1:90000]
airlines_data.head()
airports_data.head()
flights_data.head()
flights_data.columns.values

flights_data=flights_data[['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER',

       'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',

       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY',

       'SCHEDULED_TIME','DISTANCE', 'SCHEDULED_ARRIVAL',

       'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED',

       'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY',

       'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]
flights_data["AIRLINE_NAME"]=flights_data.apply(lambda x: airlines_data.loc[airlines_data['IATA_CODE'] == x["AIRLINE"],"AIRLINE"].values[0],axis=1)
flights_data[["AIRLINE_NAME","AIRLINE","ORIGIN_AIRPORT"]].head()
flights_data["ORIGIN_AIRPORT_NAME"]=flights_data.apply(lambda x: airports_data.loc[airports_data['IATA_CODE'] == x["ORIGIN_AIRPORT"],"AIRPORT"].values[0],axis=1)
flights_data[["AIRLINE_NAME","ORIGIN_AIRPORT","ORIGIN_AIRPORT_NAME"]].head()
#General Info

number_of_delayed = flights_data["DEPARTURE_DELAY"].apply(lambda s: 1 if s!=0 else 0);

print("Total number of flights: "+str(len(flights_data)))

print("Number of cancelled flights: "+str(sum(flights_data["CANCELLED"])))

print("Number of delayed flights: "+str(sum(number_of_delayed)))

print("Number of diverted flights: "+str(sum(flights_data["DIVERTED"])))





print("Number of not cancelled flights: "+str(len(flights_data)-sum(flights_data["CANCELLED"])))

print("Number of not delayed flights: "+str(len(flights_data)-sum(number_of_delayed)))

# print("The number of missing data: "+str(flights_data['DEPARTURE_TIME'].isnull().sum()));

print("Percentage of cancelled flights: "+str((sum(flights_data["CANCELLED"])*1.0/len(flights_data))*100)+"%")

print("Percentage of delayed flights: "+str((sum(number_of_delayed)*1.0/len(flights_data))*100)+"%")
flights_data["ON_TIME"]=flights_data["ARRIVAL_DELAY"].apply(lambda row: 1 if row==0 else 0)

print(len(flights_data["AIRLINE_DELAY"]))

print("ON_TIME: "+str(flights_data["ON_TIME"].sum()))

missing_data_info={};

for column in flights_data.columns:

    missing_data_info[column]=flights_data[column].isnull().sum()

missing_data_info_sorted = sorted(missing_data_info.items(), key=operator.itemgetter(1))

missing_data_info_sorted
flights_data[["DEPARTURE_DELAY","ARRIVAL_DELAY"]].plot.box()
def get_airline_information(column_name,airline_dataframe,flight_dataframe):

    return airline_dataframe.apply(lambda x: flight_dataframe.loc[x["IATA_CODE"]==flight_dataframe["AIRLINE"],column_name].values[0] if len(flight_dataframe.loc[x["IATA_CODE"]==flight_dataframe["AIRLINE"],column_name])>0 else 0,axis=1)
cancelled_flights = flights_data

grouped_cancelled_flights=cancelled_flights[["AIRLINE","AIRLINE_NAME","CANCELLED","ON_TIME"]].groupby(['AIRLINE','AIRLINE_NAME']).sum().reset_index()

grouped_cancelled_flights["FLIGHTS_COUNT"]=cancelled_flights[["AIRLINE","AIRLINE_NAME","ON_TIME"]].groupby(['AIRLINE','AIRLINE_NAME']).count().reset_index()["ON_TIME"]

grouped_cancelled_flights["CANCELLED_PERCENTAGE"]=grouped_cancelled_flights["CANCELLED"]*1.0/grouped_cancelled_flights["FLIGHTS_COUNT"]*100

grouped_cancelled_flights["ON_TIME_PERCENTAGE"]=grouped_cancelled_flights["ON_TIME"]*1.0/grouped_cancelled_flights["FLIGHTS_COUNT"]*100

grouped_cancelled_flights[["AIRLINE","AIRLINE_NAME","FLIGHTS_COUNT","CANCELLED","ON_TIME","CANCELLED_PERCENTAGE","ON_TIME_PERCENTAGE"]].sort_values(by=['CANCELLED_PERCENTAGE'],ascending=[False])
airlines_data["FLIGHTS_COUNT"]=get_airline_information("FLIGHTS_COUNT",airlines_data,grouped_cancelled_flights)

airlines_data["ON_TIME"]=get_airline_information("ON_TIME",airlines_data,grouped_cancelled_flights)

airlines_data["ON_TIME_PERCENTAGE"]=get_airline_information("ON_TIME_PERCENTAGE",airlines_data,grouped_cancelled_flights)

airlines_data.sort_values(by="ON_TIME_PERCENTAGE",ascending=False)
airlines_data["ON_TIME"].plot.pie(labels=airlines_data["AIRLINE"],autopct='%.2f', fontsize=20, figsize=(10, 10),colors=['r','g','b','w','y'])
airlines_data.sort_values(by=["ON_TIME_PERCENTAGE"],ascending=False).plot(x="AIRLINE",y='ON_TIME_PERCENTAGE',kind='bar', figsize=(10, 10),colors=['r','g','b','w','y'])
#Delay by Airlines

positive_delayed_flight=flights_data

positive_delayed_flight=positive_delayed_flight[positive_delayed_flight['DEPARTURE_DELAY']>=0]

positive_delayed_flight_grouped=positive_delayed_flight[["AIRLINE","AIRLINE_NAME","DEPARTURE_DELAY"]].groupby(["AIRLINE",'AIRLINE_NAME']).mean().reset_index()
airlines_data["MEAN_DEPARTURE_DELAY"]=get_airline_information("DEPARTURE_DELAY",airlines_data,positive_delayed_flight_grouped)

airlines_data[["AIRLINE","ON_TIME_PERCENTAGE","MEAN_DEPARTURE_DELAY"]].sort_values(by="MEAN_DEPARTURE_DELAY",ascending=True).head()
#Mean delay for each airlines

airlines_data.sort_values(by=["MEAN_DEPARTURE_DELAY"],ascending=False).plot(x="AIRLINE",y="MEAN_DEPARTURE_DELAY",kind='bar')
#Ahead flights by Airlines

ahead_flight=flights_data

ahead_flight=ahead_flight[ahead_flight['DEPARTURE_DELAY']<=0]

ahead_flight['DEPARTURE_DELAY']=ahead_flight['DEPARTURE_DELAY'].abs()

ahead_flight_grouped=ahead_flight[["AIRLINE","AIRLINE_NAME","DEPARTURE_DELAY"]].groupby(['AIRLINE','AIRLINE_NAME']).mean().reset_index()

ahead_flight_grouped.sort_values(by=["DEPARTURE_DELAY"],ascending=False)
airlines_data["MEAN_DEPARTURE_AHEAD"]=get_airline_information("DEPARTURE_DELAY",airlines_data,ahead_flight_grouped)

airlines_data[["AIRLINE","ON_TIME_PERCENTAGE","MEAN_DEPARTURE_DELAY"]].sort_values(by="MEAN_DEPARTURE_DELAY",ascending=True).head()
airlines_data.sort_values(by=["MEAN_DEPARTURE_AHEAD"],ascending=False).plot(x="AIRLINE",y="MEAN_DEPARTURE_AHEAD",kind='bar')
airlines_data[["AIRLINE","ON_TIME_PERCENTAGE","MEAN_DEPARTURE_DELAY","MEAN_DEPARTURE_AHEAD"]].sort_values(by=["MEAN_DEPARTURE_AHEAD"],ascending=False)
airlines_data["CANCELLED_PERCENTAGE"]=get_airline_information("CANCELLED_PERCENTAGE",airlines_data,grouped_cancelled_flights)

airlines_data.sort_values(by=["CANCELLED_PERCENTAGE"],ascending=False).plot(x="AIRLINE",y="CANCELLED_PERCENTAGE",kind='bar')
airlines_data[["AIRLINE","CANCELLED_PERCENTAGE"]].sort_values(by=["CANCELLED_PERCENTAGE"],ascending=True)
#Percentage by AIRLINES for diverted flights

diverted_flights = flights_data#.drop(flights_data[flights_data["CANCELLED"] != 1].index)

diverted_flights=diverted_flights[["AIRLINE","AIRLINE_NAME","DIVERTED"]].groupby(['AIRLINE','AIRLINE_NAME']).sum().reset_index()

diverted_flights.sort_values(by=["DIVERTED"],ascending=True).head(3)
airlines_data["DIVERTED_FLIGHTS"]=get_airline_information("DIVERTED",airlines_data,diverted_flights)

airlines_data.sort_values(by=["DIVERTED_FLIGHTS"],ascending=False).plot(x="AIRLINE",y="DIVERTED_FLIGHTS",kind='bar')
#CANCELLATION_REASON PERCENTAGE

cancellation_reasons_flights = flights_data

cancellation_reasons_flights=cancellation_reasons_flights[["CANCELLATION_REASON","CANCELLED"]].groupby(['CANCELLATION_REASON']).sum().reset_index()

cancellation_reasons_flights["CANCELLATION_REASON_PERCENTAGE"]=cancellation_reasons_flights["CANCELLED"]/sum(flights_data["CANCELLED"])

print("A - Carrier; B - Weather; C - National Air System; D - Security")

cancellation_reasons_flights
#CANCELLATION_REASON FOR AIRLINES

cancellation_reasons_flights = flights_data

cancellation_reasons_flights=cancellation_reasons_flights[["CANCELLED","AIRLINE","AIRLINE_NAME","CANCELLATION_REASON"]].groupby(['AIRLINE','AIRLINE_NAME','CANCELLATION_REASON']).sum().reset_index()

print("A - Carrier; B - Weather; C - National Air System; D - Security")

cancellation_reasons_flights.sort_values(by=['CANCELLED'],ascending=[False])
def create_airlines_cancellation_table(reason_code,airlines_dataframe,cancellation_reasons_dataframe):

    tmp_cancellation_reasons=cancellation_reasons_dataframe[cancellation_reasons_dataframe["CANCELLATION_REASON"]==reason_code]

    return airlines_dataframe.apply(lambda x: tmp_cancellation_reasons.loc[x["IATA_CODE"]==tmp_cancellation_reasons["AIRLINE"],"CANCELLED"].values[0] if len(tmp_cancellation_reasons.loc[x["IATA_CODE"]==tmp_cancellation_reasons["AIRLINE"],"CANCELLED"])>0 else 0,axis=1)



    
airlines_cancellation_reasons=airlines_data;

airlines_cancellation_reasons["CARRIER"]=create_airlines_cancellation_table("A",airlines_cancellation_reasons,cancellation_reasons_flights)

airlines_cancellation_reasons["WEATHER"]=create_airlines_cancellation_table("B",airlines_cancellation_reasons,cancellation_reasons_flights)

airlines_cancellation_reasons["AIR_SYS"]=create_airlines_cancellation_table("C",airlines_cancellation_reasons,cancellation_reasons_flights)

airlines_cancellation_reasons["SECURITY"]=create_airlines_cancellation_table("D",airlines_cancellation_reasons,cancellation_reasons_flights)

airlines_cancellation_reasons
# Setting the positions and width for the bars

pos = list(range(len(airlines_cancellation_reasons['AIRLINE']))) 

width = 0.25 

# Plotting the bars

fig, ax = plt.subplots(figsize=(35,10))



plt.bar(pos, 

        airlines_cancellation_reasons['CARRIER'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#EE3224', 

        # with label the first value in first_name

        label=airlines_cancellation_reasons['CARRIER'][0])



plt.bar([p + width for p in pos], 

        airlines_cancellation_reasons['WEATHER'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#4286f4', 

        # with label the first value in first_name

        label=airlines_cancellation_reasons['WEATHER'][0])





plt.bar([p + width*2 for p in pos], 

        airlines_cancellation_reasons['AIR_SYS'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#FFC222', 

        # with label the first value in first_name

        label=airlines_cancellation_reasons['AIR_SYS'][0])



plt.bar([p + width*3 for p in pos], 

        airlines_cancellation_reasons['SECURITY'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#80f441', 

        # with label the first value in first_name

        label=airlines_cancellation_reasons['SECURITY'][0])



# Set the y axis label

ax.set_ylabel('Count')



# Set the chart's title

ax.set_title('Airlines cancelled reasons counts')

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(airlines_cancellation_reasons['AIRLINE'])



plt.xlim(min(pos)-width, max(pos)+width*5)

plt.ylim([0, max(airlines_cancellation_reasons['SECURITY'] + airlines_cancellation_reasons['AIR_SYS'] + airlines_cancellation_reasons['WEATHER']+airlines_cancellation_reasons["CARRIER"])] )



# Adding the legend and showing the plot

plt.legend(['CARRIER',"WEATHER", 'AIR_SYS', 'SECURITY'], loc='upper left')

plt.grid()

plt.show()
#make the time 4-digits to all departure records

flights_data['DEPARTURE_TIME']=flights_data['DEPARTURE_TIME'].fillna(0)

flights_data['DEPARTURE_TIME']=flights_data['DEPARTURE_TIME'].astype(int)

flights_data['SCHEDULED_DEPARTURE']=flights_data['SCHEDULED_DEPARTURE'].apply(lambda x: "0"+str(x) if (int(x)<999 and int(x)>99) else "00"+str(x) if int(x)<100 else int(x))

flights_data['DEPARTURE_TIME']=flights_data['DEPARTURE_TIME'].apply(lambda x: "0"+str(x) if (int(x)<999 and int(x)>99) else "00"+str(x) if int(x)<100 else int(x))



# #combine time with data and formate it

flights_data['SCHEDULED_DEPARTURE_DATE']=flights_data[['SCHEDULED_DEPARTURE','YEAR','MONTH','DAY']].apply(lambda x: str(x['YEAR'])+"-"+str(x['MONTH'])+"-"+str(x['DAY'])+"-"+str(x['SCHEDULED_DEPARTURE']),axis=1)

flights_data['SCHEDULED_DEPARTURE_DATE']=pd.to_datetime(flights_data['SCHEDULED_DEPARTURE_DATE'], format='%Y-%m-%d-%H%M', errors='coerce')



flights_data['DEPARTURE_DATE']=flights_data[['DEPARTURE_TIME','YEAR','MONTH','DAY']].apply(lambda x: str(x['YEAR'])+"-"+str(x['MONTH'])+"-"+str(x['DAY'])+"-"+str(x['DEPARTURE_TIME']),axis=1)

flights_data['DEPARTURE_DATE']=pd.to_datetime(flights_data['DEPARTURE_DATE'], format='%Y-%m-%d-%H%M', errors='coerce')
flights_data['DEPARTURE_DATE'].head()
flights_data.head()
#make the time 4-digits to all arrival records

flights_data['ARRIVAL_TIME']=flights_data['ARRIVAL_TIME'].fillna(0)

flights_data['ARRIVAL_TIME']=flights_data['ARRIVAL_TIME'].astype(int)

flights_data['SCHEDULED_ARRIVAL']=flights_data['SCHEDULED_ARRIVAL'].apply(lambda x: "0"+str(x) if (int(x)<999 and int(x)>99) else "00"+str(x) if int(x)<100 else x)

flights_data['ARRIVAL_TIME']=flights_data['ARRIVAL_TIME'].apply(lambda x: "0"+str(x) if (int(x)<999 and int(x)>99) else "00"+str(x) if int(x)<100 else x)



#combine time with data and formate it

flights_data['SCHEDULED_ARRIVAL_DATE']=flights_data[['SCHEDULED_ARRIVAL','YEAR','MONTH','DAY']].apply(lambda x: str(x['YEAR'])+"-"+str(x['MONTH'])+"-"+str(x['DAY'])+"-"+str(x['SCHEDULED_ARRIVAL']),axis=1)

flights_data['SCHEDULED_ARRIVAL_DATE']=pd.to_datetime(flights_data['SCHEDULED_ARRIVAL_DATE'], format='%Y-%m-%d-%H%M', errors='coerce')



flights_data['ARRIVAL_DATE']=flights_data[['ARRIVAL_TIME','YEAR','MONTH','DAY']].apply(lambda x: str(x['YEAR'])+"-"+str(x['MONTH'])+"-"+str(x['DAY'])+"-"+str(x['ARRIVAL_TIME']),axis=1)

flights_data['ARRIVAL_DATE']=pd.to_datetime(flights_data['ARRIVAL_DATE'], format='%Y-%m-%d-%H%M', errors='coerce')

flights_data["ARRIVAL_TIME"].head()
flights_data["FLIGHT_TIME"]=flights_data['ARRIVAL_DATE']-flights_data['DEPARTURE_DATE']
flights_data[["ARRIVAL_DATE","DEPARTURE_DATE",'FLIGHT_TIME']].head()
flights_data["FLIGHT_TIME_IN_MINUTES"]=flights_data['FLIGHT_TIME'].apply(lambda x: int(x.seconds/60) if x.seconds>0 else 0)
flights_data['SPEED']=flights_data.apply(lambda x: x["DISTANCE"]/x['FLIGHT_TIME_IN_MINUTES'] if x['FLIGHT_TIME_IN_MINUTES']>0 else 0,axis=1)
flights_data[['SPEED','DISTANCE','FLIGHT_TIME_IN_MINUTES','ARRIVAL_DATE','DEPARTURE_DATE']].sort_values(by=["SPEED"],ascending=False).head()
#Speed by AIRLINES

flights=flights_data[["AIRLINE","SPEED"]].groupby(['AIRLINE']).mean().reset_index()
airlines_data["MEAN_SPEED"]=get_airline_information("SPEED",airlines_data,flights)

airlines_data[["AIRLINE","MEAN_SPEED"]].sort_values(by=["MEAN_SPEED"],ascending=False).head(3)
#Airlines by Speed

# plot = flights.sort_values(by=["SPEED"],ascending=False).plot(x="AIRLINE_NAME",y="SPEED",kind='bar')

airlines_data.sort_values(by=["MEAN_SPEED"],ascending=False).plot(x="AIRLINE",y="MEAN_SPEED",kind='bar')

airlines_data["RANKING"]=0

tmp=airlines_data.sort_values(by=["ON_TIME_PERCENTAGE"],ascending=True).reset_index(drop=True)

tmp["RANKING"]=tmp.apply(lambda x: (x["RANKING"]+x.name),axis=1)



tmp=tmp.sort_values(by=["MEAN_SPEED"],ascending=True).reset_index(drop=True)

tmp["RANKING"]=tmp.apply(lambda x: (x["RANKING"]+x.name),axis=1)



tmp=tmp.sort_values(by=["MEAN_DEPARTURE_DELAY"],ascending=False).reset_index(drop=True)

tmp["RANKING"]=tmp.apply(lambda x: (x["RANKING"]+x.name),axis=1)



tmp=tmp.sort_values(by=["MEAN_DEPARTURE_AHEAD"],ascending=False).reset_index(drop=True)

tmp["RANKING"]=tmp.apply(lambda x: (x["RANKING"]+x.name),axis=1)



tmp=tmp.sort_values(by=["CANCELLED_PERCENTAGE"],ascending=False).reset_index(drop=True)

tmp["RANKING"]=tmp.apply(lambda x: (x["RANKING"]+x.name),axis=1)



tmp=tmp.sort_values(by=["DIVERTED_FLIGHTS"],ascending=False).reset_index(drop=True)

tmp["RANKING"]=tmp.apply(lambda x: (x["RANKING"]+x.name),axis=1)



(tmp.sort_values(by=["RANKING"],ascending=False))[["AIRLINE","RANKING"]]
#Percentage by ORIGIN_AIRPORT

cancelled_flights_by_origin_airpot = flights_data

grouped_cancelled_flights_by_origin_airpot=cancelled_flights_by_origin_airpot[["ORIGIN_AIRPORT","CANCELLED","ON_TIME"]].groupby(['ORIGIN_AIRPORT']).sum().reset_index()

grouped_cancelled_flights_by_origin_airpot["FLIGHTS_COUNT"]=cancelled_flights_by_origin_airpot[["ORIGIN_AIRPORT","ON_TIME"]].groupby(['ORIGIN_AIRPORT']).count().reset_index()["ON_TIME"]

grouped_cancelled_flights_by_origin_airpot["CANCELLED_PERCENTAGE"]=grouped_cancelled_flights_by_origin_airpot["CANCELLED"]*1.0/grouped_cancelled_flights_by_origin_airpot["FLIGHTS_COUNT"]*100

grouped_cancelled_flights_by_origin_airpot["ON_TIME_PERCENTAGE"]=grouped_cancelled_flights_by_origin_airpot["ON_TIME"]*1.0/grouped_cancelled_flights_by_origin_airpot["FLIGHTS_COUNT"]*100

grouped_cancelled_flights_by_origin_airpot[["ORIGIN_AIRPORT","FLIGHTS_COUNT","CANCELLED","ON_TIME","CANCELLED_PERCENTAGE","ON_TIME_PERCENTAGE"]].sort_values(by=['ON_TIME_PERCENTAGE'],ascending=[False])

plt.figure();

# print(len(grouped_cancelled_flights_by_origin_airpot["ORIGIN_AIRPORT"]))

plot = grouped_cancelled_flights_by_origin_airpot.sort_values(by=["ON_TIME_PERCENTAGE"],ascending=False).plot(x="ORIGIN_AIRPORT",y="ON_TIME_PERCENTAGE",kind='bar',figsize=(100,30))