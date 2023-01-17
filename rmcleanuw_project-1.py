# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime



!pip install sodapy #Install the sodapy library so I can access the open data via API.

from sodapy import Socrata





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("seattleopendata")



# Unauthenticated client only works with public data sets. Note 'None'

# in place of application token, and no username or password:

client = Socrata("data.seattle.gov", secret_value_0) #Initialize authenticated access to the seattle data portal.

client.timeout=240 #set the timeout value higher.



#Setup variables for the databases to make code more readable.

transactions_2019 = "qktt-2bsy"

transactions_2020 = "wtpb-jp8d"



query19 = client.get(transactions_2019, limit=1) #Get 1 record from the 2019 dataset.

print("Done receiving 2019 data.")



# Convert to pandas DataFrame

df2019 = pd.DataFrame.from_records(query19)

df2019
query20 = client.get(transactions_2020, limit=1) #Get 1 record from the 2019 dataset.

print("Done receiving 2020 data.")



# Convert to pandas DataFrame

df2020 = pd.DataFrame.from_records(query20)

df2020
#Find out the number of records we expect to find. Store this value as expected_record_count

#results_query = client.get(transactions_2019, where="occupancydatetime between '2019-01-01T00:00:00' and '2019-04-30T23:59:59'", select="count(*)")

#results_query

#expected_record_count = results_query[0]['count']

#expected_record_count

#expected_record_count_2019 = expected_record_count



#This is the number. I am now baking the number into the code as it shouldn't change, and so this part doesn't have to run again if I close the Kaggle book.

expected_record_count_2019 = 94296797
#Find out the number of records we expect to find. Store this value as expected_record_count

#results_query = client.get(transactions_2020, where="occupancydatetime between '2020-01-01T00:00:00' and '2020-04-30T23:59:59'", select="count(*)")

#expected_record_count_2020 = results_query[0]['count']

#expected_record_count_2020



#This is the number. I am now baking the number into the code as it shouldn't change, and so this part doesn't have to run again if I close the Kaggle book.

expected_record_count_2020 = 69957859
#start = 0 #Start at 0

#page = 1500000 #Grab 1.5M records at a time.

#results = [] #Start with an empty results list.





#while True:

    # Fetch the set of records starting at 'start'

#    results.extend( client.get(transactions_2019, where="occupancydatetime between '2019-01-01T00:00:00' and '2019-04-30T23:59:59'", select="occupancydatetime,paidoccupancy,paidparkingarea", offset=start, limit=page))

    # Move up the starting record

#    start = start + page

    # Print a status message.

#    print("Working chunk starting at %d of %d." %(start,expected_record_count_2020))

    # If we have fetched all of the records, bail out

#    if (len(results) >= expected_record_count_2019):

#       break

# Convert the list into a data frame

#df2019 = pd.DataFrame.from_records(results)

#df2019
#Create a date range for 2020-01-01 to 2020-01-30

#Method found on https://stackoverflow.com/questions/17576615/pandas-date-range-from-datetimeindex-to-date-format

rng = pd.date_range('2020-01-01', '2020-04-30', freq='D')

range2019 = rng.date

print(len(range2019))



counter = 0

day_max = str(0)

#while counter < len(range2019)-1:

#    today = counter

#    tomorrow = counter+1

#    print("Today is "+ str(range2019[today]) + " and tomorrow is " + str(range2019[tomorrow]))

#    results_query = client.get(transactions_2020, select="count(*)", where="occupancydatetime between " + "'" + str(range2019[today]) +"' and '" + str(range2019[tomorrow]) +"'")

#    counter += 1

#    day_count = results_query[0]['count']

#    print(day_count)

#    if int(day_count) > int(day_max):

#        print("New record "+ day_count)

#        day_max = day_count



#    if int(day_count) > 2000000:

#        print("Over 2 million records on this day")

#    

#print("All done and the highest day is" + day_count)

#This was used to get the 2019 data.



#First setup a date series from Jan 1 to May 1. I have to go 1 day beyond the last date I want data for (April 30) because of the way the search query works.

#rng_series = pd.date_range('2019-01-01', '2019-05-01', freq='D')

#rng = rng_series.date #Convert the date-time series into a date series.

#today = 0 #Set a counter

#day_index = 1 #Set another counter

#collector = [] #Set a list.



#Now I will setup a loop to step through all the dates in my series and query the Seattle open data API. I will ask the data server to sum all the paid parking transactions for each day and group by the parking area. I will then add the date to this data and append it to a collector. This saves a lot of memory since I am asking the Seattle Open Data server API to send over only what I need, and since I'm only working on a little chunk at a time.



#while today < len(rng)-1: #Stop when we are last from the end, because I don't want to ever have a query STARTING with May 1.

#    print("Working through date " +str(rng[today])) #Just to help me.

#    results = client.get(transactions_2019, select="paidparkingarea,sum(paidoccupancy)", group="paidparkingarea", where="occupancydatetime between " + "'" + str(rng[today]) +"' and '" + str(rng[today + 1]) +"'")

#    for item in results: #This is where we iterate through each day's results and add in the date to the dictionaries comtained in the list.

#        item.update( {"date": rng[today]}) #Add the full date to each dictionary entry.

#        item.update( {"year": "2019"}) #Add the year value (Just adding it this 'dumb' way.)

#        item.update( {"yearday": day_index}) #Add the day of the year.

#        collector.append(item)

#    today +=1

#    day_index +=1



#Now save off to a file.



#import pickle



#with open("2019_doy.txt", "wb") as fp:   #Pickling

#    pickle.dump(collector, fp)

#This was used to get the 2020 data. It is pretty much the same code. I will comment where changes were made.



#First setup a date series from Jan 1 to May 1. I have to go 1 day beyond the last date I want data for (April 30) because of the way the search query works.

#rng_series = pd.date_range('2020-01-01', '2020-05-01', freq='D') #2020

#rng = rng_series.date #Convert the date-time series into a date series.

#today = 0 #Set a counter

#day_index = 2 #Set another counter. I started this one at 2 so that the day-of-year would match up better with the day of week. January 1 2020 was the first WEDNESDAY or the year, equivalent to day "2" of 2019.

#collector = [] #Set a list.



#Now I will setup a loop to step through all the dates in my series and query the Seattle open data API. I will ask the data server to sum all the paid parking transactions for each day and group by the parking area. I will then add the date to this data and append it to a collector. This saves a lot of memory since I am asking the Seattle Open Data server API to send over only what I need, and since I'm only working on a little chunk at a time.



#while today < len(rng)-1: #Stop when we are last from the end, because I don't want to ever have a query STARTING with May 1.

#    print("Working through date " +str(rng[today])) #Just to help me.

#    results = client.get(transactions_2020, select="paidparkingarea,sum(paidoccupancy)", group="paidparkingarea", where="occupancydatetime between " + "'" + str(rng[today]) +"' and '" + str(rng[today + 1]) +"'")

#    for item in results: #This is where we iterate through each day's results and add in the date to the dictionaries comtained in the list.

#        item.update( {"date": rng[today]}) #Add the full date to each dictionary entry.

#        item.update( {"year": "2020"}) #Add the year value (Just adding it this 'dumb' way.)

#        item.update( {"yearday": day_index}) #Add the day of the year.

#        collector.append(item)

#    today +=1

#    day_index +=1



#Now save off to a file.



#import pickle



#with open("2020_doy2.txt", "wb") as fp:   #Pickling

#    pickle.dump(collector, fp)



import pickle



#Open the files.

with open("/kaggle/input/parking/2019_doy.txt", "rb") as fp:   # Unpickling

    a = pickle.load(fp)



with open("/kaggle/input/parking/2020_doy2.txt", "rb") as fp:   # Unpickling

    b = pickle.load(fp)



#Make dataframes for pandas.

df1 = pd.DataFrame.from_records(a)

df2 = pd.DataFrame.from_records(b)
df1 #Let's see that gorgeous 2019 dataframe. 
df2 #Let's see that gorgeous 2020 dataframe. 
#Merge the dataframes to 1.

df = pd.merge(df1, df2, how='outer')

df
#Now I am going to use the pivot table function to reshape this data so it is organized by date on the rows and the column is the parking areas.

pivot = df.pivot_table(index="yearday", columns=["paidparkingarea","year"], values="sum_paidoccupancy", aggfunc="sum")

pivot
#Now let's plot it.

import matplotlib.pyplot as plt

import datetime as dt



plt.style.use('ggplot')

#pivot2019 =  pivot2019.astype(float)

#pivot2020 =  pivot2020.astype(float)

#pivot2019.plot()

#pivot2020.plot()



pivotfloat = pivot.astype(float)

pivotfloat.plot()
pivotfloat.plot(style='o')
cols = pivot.iloc[:,0:].columns #THANK YOU RAFAL - Grabs the names of all the columns.

#print(cols) #print the column names.

pivotfloat[cols].plot(figsize=(20,60), style='.', subplots=True, layout=(11,4))
pivot_2 = df.pivot_table(index=["yearday","year"], columns=["paidparkingarea"], values="sum_paidoccupancy", aggfunc="sum")

pivot_2.fillna(0)

pivotfloat2 = pivot_2.astype(float).fillna(0)

pivotfloat2
cols2 = pivot_2.iloc[:,0:].columns #THANK YOU RAFAL - Grabs the names of all the columns.

print(cols2) #print the column names.

pivotfloat2.plot(figsize=(20,60), subplots=True, layout=(11,4))



#fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(12,4)) #sets-up subplotting



#for index, col in enumerate(cols2):

#    axs[index].plot(df[[col,"Ballard"]])

#    axs[index].set_title('Title:'+str(col))
