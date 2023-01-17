import pandas as pd 
import numpy as numpy 
import matplotlib.pyplot as plt
import datetime
import calendar 
  
def findDay(date): 
    born = datetime.datetime.strptime(date, '%d %m %Y').weekday() 
    return (calendar.day_name[born]) 
fitbit = pd.read_csv('../input/one-year-of-fitbit-chargehr-data/One_Year_of_FitBitChargeHR_Data.csv') # fitbit data
fitbit.shape
fitbit.head()
# convert the data type and set Date as index
fitbit['Date'] = pd.to_datetime(fitbit['Date'])
fitbit = fitbit.set_index('Date')
# Plot the time series of floors per day
ax = fitbit['floors'].plot(color ='blue',figsize=(20,4))

ax.set_xlabel('Date')
ax.set_ylabel('Floors per day')

plt.figure()
plt.show()
### Minutes sitting 
# Plot the time series of floors per day
ax = fitbit['Minutes_sitting'].plot(color ='blue',figsize=(20,4))

ax.set_xlabel('Date')
ax.set_ylabel('Minutes sitting per day')

plt.figure()
plt.show()
### Minutes sitting 
# Plot the time series of floors per day
ax = fitbit['Minutes_of_intense_activity'].plot(color ='blue',figsize=(20,4))

ax.set_xlabel('Date')
ax.set_ylabel('Minutes_of_intense_activity per day')

plt.figure()
plt.show()
### In the same way other variables can be visualized
### ........
# create lists for weekends and business days
activity_wd = []
activity_bd = []

# reset index for obtaining the row Date
fitbit = fitbit.reset_index()
# loop through the rows
counter = 0
for index, row in fitbit.iterrows(): 
    
    date = str(row['Date'])
    msi = row['Minutes_of_intense_activity']
    
    date = date.split(' ')
    
    date = date[0].replace('-', ',')
    
    year = date.split(',')[0]
    
    month = date.split(',')[1]
  
    day = date.split(',')[2]
    
    day_string = str(day)+' '+str(month)+' '+str(year)
    print(day_string)
 
    day_of_week = findDay(day_string)
    if day_of_week == 'Saturday' or day_of_week == 'Sunday': 
        activity_wd.append(msi)
    else: 
        activity_bd.append(msi)
# Plot the time series of floors per day

plt.figure(figsize=(20,4))
plt.plot(activity_bd)
plt.show()

plt.figure(figsize=(20,4))
plt.plot(activity_wd)
plt.show()
# extend the analysis with future work ..........



