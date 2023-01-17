# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
# use Seaborn styles
sns.set() 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
emergency = pd.read_csv("../input/911.csv")
emergency.head()
# Any results you write to the current directory are saved as output.
total = emergency.isnull().sum().sort_values(ascending=False)
total
#Two columns have null values - zip and twp
percent = (emergency.isnull().sum()/emergency.isnull().count()).sort_values(ascending = False)
#print(percent)
pd1 = pd.concat([total,percent],axis =1 ,keys=['Total','Percent'])
pd2 = pd1[pd1['Total']>0]
print(pd2)
#zip has 0.12 percent of null values
#twp has very small percent of null values
#to reset the index
#emergency.reset_index(inplace=True)
title1 = emergency.pivot_table(values = 'e',index=['title'],aggfunc='count')

#display the top 10 title
print(title1.sort_values('e',ascending=False).head(10))

#groupby on the basis of title and then plot a pie chart
emergency.groupby(['title'])['title'].size().sort_values(ascending=False).head(7).plot(kind='pie',autopct='%1.1f%%')

#'Traffic: VEHICLE ACCIDENT' title has the highest 911 calls
#Extract only the titles with 'Traffic' and display the top 5 titles
print(title1[title1.index.str.contains('^Traffic')].sort_values('e',ascending=False).head(5))
#Extracting the type from the titles and storing it in a new column
emergency['type'] = emergency["title"].apply(lambda x: x.split(':')[0])
emergency.head()
#emergency.reset_index(inplace=True)
title2 = emergency.pivot_table(values = 'e',index=['type'],aggfunc='count').sort_values('e',ascending=False)
print(title2)
title2.plot(kind='bar')
plt.ylabel("count")

#EMS type has the highest count of 911 calls although the 
#'Traffic: VEHICLE ACCIDENT' title has the highest count of 911 calls
#pie-chart displaying the percent of different types
emergency.groupby(['type'])['type'].size().sort_values(ascending=True).plot(kind='pie',autopct='%1.1f%%')
#emergency.reset_index(inplace=True)
title3 = emergency.pivot_table(values = 'e',index=['type','title'],aggfunc='count')
title4 = title3[title3.e>5000]
print(title4)
title4.plot(kind='barh')
plt.ylabel("count")
#To extract the timestamp

emergency['timeStamp']=pd.to_datetime(emergency.timeStamp)
emergency['year']=emergency['timeStamp'].dt.year
emergency['month']=emergency['timeStamp'].dt.month
emergency['quarter']=emergency['timeStamp'].dt.quarter
emergency['hour']=emergency['timeStamp'].dt.hour
emergency.head()
#To display the 911 calls for each month and display it for each type
sns.countplot(emergency['month'],hue = emergency['type'],palette="Blues")

#At the start of the year and also at the end of the year, the 911 calls
#are high (Jan,Feb,March and December)
#To display the 911 calls for each quarter and display it for each type
sns.countplot(emergency['quarter'],hue = emergency['type'],palette="Blues")

#For quarter 1, the count of 911 calls for EMS type is highest
# For type - 'Fire', the calls are equally spread out for all the quarters
#Histogram plot with the variation in calls during the hours of the day
sns.kdeplot(emergency.hour,shade=True)
plt.title("Histogram of hour")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()

#At around 4 PM, the number of calls have peaked and are lower during the
#early hours of the day(3-5 AM)
#scatter plot
plt.plot(emergency.lat, emergency.lng, 'o', color='blue');
plt.title("Scatter Plot of Location of Calls")
plt.xlabel("Lattitude")
plt.ylabel("Longitutde")
plt.show()
#seeing the scatter plot we can see that most of the emergencies 
#are from around lat = 40 and lon = -75 i.e. from a particular area
#upon checking the data - town 'LOWER MERION' is around the clustered (lat,Lon)
#create a subset of only the EMS type
ems = emergency[emergency['type']=='EMS']
#ems.reset_index(inplace=True)
ems_subset = ems[['timeStamp','title','e']]
#merge the count of calls on the basis of week using resample.
#This can be done as this is a time series data

piv=pd.pivot_table(ems_subset, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)
piv1=piv.resample('W', how=[np.sum]).reset_index()  #'W' represents the week
piv1.head()
#Plot for 'EMS: ASSAULT VICTIM' on the basis of the timestamp

plt.xticks(rotation=90)
plt.plot_date(piv1['timeStamp'], piv1['EMS: ASSAULT VICTIM'],'k')
plt.title("Timestamp Analysis for 'EMS: ASSAULT VICTIM'")
plt.xlabel("Timestamp")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#Seeing the plot we can observe that during october of 2016, the calls for
#'EMS: ASSAULT VICTIM' were highest
#Also seeing the trend, the calls for 'EMS: ASSAULT VICTIM' increased
#as the year 2016 progressed. But it didn't follow the same trend in 2017
#and fluctuated throughout the year 2017
#comparison betwen 'EMS: ASSAULT VICTIM' and 'EMS: ALTERED MENTAL STATUS'

plt.subplot()
plt.xticks(rotation=90)
plt.plot_date(piv1['timeStamp'], piv1['EMS: ASSAULT VICTIM'],'k')
#plt.plot_date(piv1['timeStamp'], piv1['EMS: ASSAULT VICTIM'],'r.')


plt.plot_date(piv1['timeStamp'], piv1['EMS: ALTERED MENTAL STATUS'],'y')
#plt.plot_date(piv1['timeStamp'], piv1['EMS: ALTERED MENTAL STATUS'],'b.')
#plt.set_title("EMS: ASSAULT VICTIM vs  EMS: ALTERED MENTAL STATUS")

plt.title("Comparison betwen 'EMS: ASSAULT VICTIM' and 'EMS: ALTERED MENTAL STATUS'")
plt.xlabel("Timestamp")
plt.ylabel("Frequency")
plt.plot()

#from the plot we can observe that 'EMS: ALTERED MENTAL STATUS' has higher
#calls compared to the 'EMS: ASSAULT VICTIM'.
#Seeing the trend of 'EMS: ALTERED MENTAL STATUS', we can observe that there
#is no specific pattern to it.
#Only during the end of 2016, 'EMS: ASSAULT VICTIM' peaks and crosses the
#count of 'EMS: ALTERED MENTAL STATUS'
#Top 10 towns with the 911 emergecny calls
town = emergency.pivot_table(values = 'e',index=['twp'],aggfunc='count')
print(town.sort_values('e',ascending=False).head(10))
emergency.groupby(['twp'])['twp'].size().sort_values(ascending=False).head(7).plot(kind='pie',autopct='%1.1f%%')

#Seeing the table and pie-chart,we can observe that 'LOWER MERION' made
#the highest 911 calls
#As town 'LOWER MERION' has the highest 911 calls, we drill down to
#that data and check the data

twp = emergency[emergency['twp']=='LOWER MERION']

#countplot month wise for town = 'LOWER MERION'
sns.countplot(twp['month'],hue = twp['type'],palette="Blues")

#Seeing the countplot for the town = 'LOWER MERION', we can observe the
#same trend as was observed for the entire emergency data.
plt.figure(1)
sns.kdeplot(twp.hour,shade=True)
plt.title("Histogram of hour for town ='LOWER MERION'")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()

plt.figure(2)
twp_EMS = twp[twp['type']=='EMS']
sns.kdeplot(twp_EMS.hour,shade=True)
plt.title("Histogram of hour for town ='LOWER MERION'and type = 'EMS'")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()

#Here plot 1 is for town = 'LOWER MERION' and plot is for the drilled
#down data of plot 1 with type as 'EMS'

#Seeing both the plots we can observe that with further drill down of data
#of 'LOWER MERION' town, there is a shift in the peak in the hour of the
#day with highest 911 calls.
#For town = 'LOWER MERION', at around 3 PM the 911 calls peaked
#while only checking for EMS type calls, the peak shifts at around 12 PM
twp_Traffic = twp[twp['type']=='Traffic']
sns.kdeplot(twp_Traffic.hour,shade=True)
plt.title("Histogram of hour for town ='LOWER MERION' and Type = 'Traffic'")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()
#on the other hand for type as Traffic, the peak shifts towards right
#at around 4 PM

#This explains that for different types of 911 calls, the peak hours vary