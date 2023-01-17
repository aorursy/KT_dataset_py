

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualization  

import seaborn as sns # Visualization 

bike =pd.read_csv("../input/2019 - 01.csv") # Reading one of the csv files
bike.shape # rows , columns
bike.head(3) # first three rows
Maletotal = (bike.member_gender=="Male").sum() #male sum

Femaletotal = (bike.member_gender=="Female").sum() # female sum

Othertotal = ((bike.member_gender!="Male") & (bike.member_gender!="Female" )).sum() # dataset contains null values so we will refer to them as other.

Maletotal,Femaletotal,Othertotal #results
plt.figure(figsize=(4,8))

plt.bar(['MALE'],[Maletotal],color='g',label='MALE',width=0.5)

plt.bar(['FEMALE'],[Femaletotal],color='r',label='FEMALE',width=0.5)

plt.bar(['OTHER'],[Othertotal],color='grey',label='OTHER',width=0.5)

plt.title("HOW MANY MALES/FEMALE USED A BIKE THIS MONTH(by uses)")

plt.legend()

plt.show()
Customerstotal = (bike.user_type=="Customer").sum() # customer sum

subscriberstotal = (bike.user_type=="Subscriber").sum() # subs sum

Customerstotal , subscriberstotal # results
plt.figure(figsize=(4,8))

plt.bar(['Customer'],[Customerstotal],color='g',label='Customers',width=0.5)

plt.bar(['Subscriber'],[subscriberstotal],color='r',label='Subscribers',width=0.5)

plt.title("Customers vs Subscribers (by uses)")

plt.legend()

plt.show()
notadults = (bike.member_birth_year>=2001).sum() # under 18 sum

adults = ((bike.member_birth_year<=2001) & (bike.member_birth_year>1959)).sum() # adult sum

elder = (bike.member_birth_year<=1959).sum() # elder sum

notadults , adults , elder # results
plt.figure(figsize=(6,8))

plt.bar(['Under 18'],[notadults],color='g',label='Under 18',width=0.5)

plt.bar(['Adult'],[adults],color='r',label='Adult(18 - 60)',width=0.5)

plt.bar(['Elder'],[elder],color='grey',label='Elder(over 60)',width=0.5)

plt.title("Age  groups (by uses)")

plt.legend()

plt.show()
allstations = pd.DataFrame(columns=["Stations"]) # creating new dataframe

allstations = pd.concat([bike['start_station_id'], bike['end_station_id']])# "concating" the two columns
uniqueid = np.unique(allstations) # Returning all the unique values from the allstation dataframe that we created earlier.

stationsusage = pd.DataFrame(columns=["station_id","Total_Uses"]) # new dataframe

for row in uniqueid:

    temp1 = bike[bike.start_station_id == row]# As start station

    temp2 = bike[bike.end_station_id == row] # As end station

    tempsum = len(temp1)+len(temp2) # sum

    stationsusage = stationsusage.append({'station_id':row,'Total_Uses':tempsum}, ignore_index=True) # add row on our dataframe
Top = stationsusage.nlargest(15, 'Total_Uses')# Top 15 stations by uses 

plt.figure(figsize=(16,8))

sns.barplot(y=Top.Total_Uses,x=Top.station_id)

plt.title("Top 15 stations (by uses)")
Top # Top 15 used stations
uniqueid = np.unique(bike.bike_id)  # Returning all the unique values from the bike_id column.

Duration = pd.DataFrame(columns=["id","Total_duration"])# New dataframe

for row in uniqueid:

    temp = bike[bike.bike_id == row] # Returns every row that matches our bike id 

    tempsum = sum(temp.trip_duration_sec) # Returns the sum of the "trip_duration_sec" column for the rows we found earlier 

    Duration = Duration.append({'id':row,'Total_duration':tempsum}, ignore_index=True) # Add new row to our dataframe
Duration.dtypes# old dtype
Duration = Duration.astype('int64') # change object to int64
Duration.dtypes # new types
Duration.nlargest(15, 'Total_duration') # top 15

Top = Duration.nlargest(15, 'Total_duration') # new dataframe

plt.figure(figsize=(16,8))

sns.barplot(y=Top.Total_duration,x=Top.id) # plotting our new dataframe

plt.title("Top 15 bikes (by uses)")
Top