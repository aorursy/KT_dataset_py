import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
path = "../input/Uber Request Data.csv"

df = pd.read_csv(path)
df.head()
df.shape
df.info()
#printing null columns and totoal no. of null values

null_cols = df.isnull().sum()

print(null_cols[null_cols>0])



print("Total number of null values: ", df.isnull().sum().sum())



#Driver with maximum trips

driver_max = df['Driver id'].value_counts()

print("The driver id: ", driver_max.index[0], " has done maximum number of trips", driver_max.iloc[0]);
df.columns
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(6,6))

sns.countplot(x=df['Pickup point'])

sns.set_style("dark")

plt.title("Frequency of Requests")

plt.xlabel("Pick up point")

plt.ylabel("Frequency")

plt.show()
#Status Frequency

print(df['Status'].value_counts())
plt.figure(figsize=(6,6))

sns.countplot(x=df['Status'])

sns.set_style("dark")

plt.title("Status Frequency")

plt.xlabel("Status")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(x='Pickup point', hue='Status', data=df)

sns.set_style("dark")

plt.title("Status Frequency")

plt.xlabel("Pickup point")

plt.show()
df['Request timestamp'].head()


#Converting the request timestampt to a similar format

df['Request timestamp'] = pd.to_datetime(df['Request timestamp'], dayfirst=True)  #  format='%d/%m/%Y %H:%M'  or '%d-%m-%Y %H:%M:%S'
#Cheching the minimum and maximum date between the requested time-stamps

print("First Request: ",df['Request timestamp'].min())

print("Last Request: ",df['Request timestamp'].max())
#Extracting hour from Request timestamp column and converting it into hour column

df['hour'] = df['Request timestamp'].apply(lambda x: x.hour)



#Extracting minute from Request timestamp column and converting it into minute column

df['minute'] = df['Request timestamp'].apply(lambda x: x.minute)



#Extracting day from Request timestamp column and converting it into day column

df['day'] = df['Request timestamp'].apply(lambda x: x.day)
df.head()
#Plotting Uber Requests by day

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='day', data=df)

plt.title("Uber Requests by Day", fontsize=20)

plt.xlabel("Day")

plt.ylabel("Request Count")

plt.show()
#Plotting Uber Requests by day

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='day', hue='Status', data=df)

plt.title("Uber Requests by Day", fontsize=20)

plt.xlabel("Day")

plt.ylabel("Request Count")

plt.legend(loc='upper right')

plt.show()
#Plotting Uber Requests by Hour

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='hour', data=df)

plt.title("Uber Requests by Hour", fontsize=20)

plt.xlabel("Hour")

plt.ylabel("Request Count")

plt.show()
#Plotting Uber Requests by hour

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='hour', hue='Status', data=df)

plt.title("Uber Requests by Hour", fontsize=20)

plt.xlabel("Hour")

plt.ylabel("Request Count")

plt.legend(loc='upper right')

plt.show()
#Splitting hours of day into different time slots



df.loc[df['hour'].between(0,3, inclusive=True), 'Time Slot'] = 'Late Night'



df.loc[df['hour'].between(4,7, inclusive=True), 'Time Slot'] = 'Early Morning'



df.loc[df['hour'].between(8,11, inclusive=True), 'Time Slot'] = 'Morning'



df.loc[df['hour'].between(12,15, inclusive=True), 'Time Slot'] = 'Noon'



df.loc[df['hour'].between(16,19, inclusive=True), 'Time Slot'] = 'Evening'



df.loc[df['hour'].between(19,23, inclusive=True), 'Time Slot'] = 'Night'
#Plotting Uber Requests by Time Slots

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='Time Slot', data=df, order=['Early Morning', 'Morning', 'Noon', 'Evening', 'Night', 'Late Night'])

plt.title("Requests by  Differnt Time Slots", fontsize=20)

plt.show()
#Plotting Uber Requests by Time Slots

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='Time Slot', hue='Status', data=df, order=['Early Morning', 'Morning', 'Noon', 'Evening', 'Night', 'Late Night'])

plt.title("Uber Requests by Time Slots", fontsize=20)

plt.ylabel("Request Count")

plt.legend(loc='upper right')

plt.show()
df.head()
dic_mapping_supply = {'Status': {'Trip Completed': 'Supply', 'Cancelled': 'Demand', 'No Cars Available': 'Demand'}}

print(dic_mapping_supply)
supply = df.replace(dic_mapping_supply)

df['supply'] = supply['Status']

df.head()
#Plotting Demand

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='hour', data=df)

plt.title("Demand", fontsize=20)

plt.xlabel("Hour")

plt.ylabel("Request Count")

plt.show()
#Plotting Supply/Demand Graph

plt.figure(figsize=(15,6))

sns.set_style("darkgrid")

sns.countplot(x='hour', hue='supply', data=df)

plt.title("Supply/Demand", fontsize=20)

plt.xlabel("Hour")

plt.ylabel("Count")

plt.show()