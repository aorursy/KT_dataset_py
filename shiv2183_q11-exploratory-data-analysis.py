#Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#Changing Working directory and printing files in it
os.chdir("../input")
print(os.listdir("../input"))
#Reading Emergency-911 calls into dataframe df
df=pd.read_csv('911.csv')
#First Part-Cleaning Data

#1. Column 'title' consists of Emergency Category and Emergency Description.Splitting it into two columns 
#'title_emr'-Emergency Category AND 'title_emrdesc'-Emergency Description
df['title_emr'],df['title_emrdesc']=df['title'].str.split(':').str
#2 Column description consists of station, rest all other information is repeated
#Extracting 'Station+6 characters after station' in column 'desc' ans storing it in column 'station'
df['station']=df['desc'].str.extract(('(Station......)'), expand=True)
#Removing Word 'Station', character ':',';' from the column 'station' and storing the cleaned data in the
#column again
df['station']=df['station'].str.replace(('Station'),'')
df['station']=df['station'].str.replace((':'),'')
df['station']=df['station'].str.replace((';'),'')
#3 Column 'timeStamp' contains date and time
#Extracting date and time from column'timeStamp' and storing it in column 'date' and 'time' respectively
df['date']=df['timeStamp'].str.extract(('(....-..-..)'), expand=True)
df['time']=df['timeStamp'].str.extract(('(..:..:..)'), expand=True)
#Second Part-Droping Repetitive Data Columns
#Dropping columns 'desc','title' and 'timeStamp' which contains repetitive information
df=df.drop(columns=['desc','title','timeStamp'])
#Third Part-Checking for Repetitive Data
#1-Length of current dataset
len(df)
#Third Part-Checking for Repetitive Data
#2-Drop duplicates
df.drop_duplicates()
#3-Compare with earlier length value to know the number of rows which are dropped
len(df)
#Fourth Part-Removing NA values
#1-Checking for columns with NA values
(df.isna().sum()/len(df))*100
#Result shows 12% of zip,0.03% of twp and 35% of station values are NA
#We can drop NA values of zip and twp since these are less than 20%, station NA values are 35% too high to be dropped
#2-Drop NA values from 'zip' and 'twp' columns
df=df.dropna(subset=['zip', 'twp'])
#3-Checking for columns with NA values AGAIN
(df.isna().sum()/len(df))*100
#Fifth Part-Column Datatype correction
#Converting zip code in column zip to integer type 
df['zip']=df['zip'].astype('int')
#Cleaned up Data
df
#Ploting histogram by grouping it by emergency category and plotting the frequency
plot=df.groupby(['title_emr'])['e'].count().plot.bar(title='Categorization of 911 Calls', figsize=(5,5), fontsize=10)
#Setting X and Y axis Label
plot.set_xlabel('Type of Emergency')
plot.set_ylabel('Frequency')
#Setting Date as Datetime object
df['date']=pd.to_datetime(df['date'], format='%Y-%m-%d')
#Extracting YEAR and MONTH out of date and storing it in a column 'year' and 'month' respectively
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df.groupby(['year','month'])['e'].count()
#Creating Monthwise 911 calls received for each year 
Year1=df[df['year']==2015].groupby(['month'])['e'].count().to_frame()
Year2=df[df['year']==2016].groupby(['month'])['e'].count().to_frame()
Year3=df[df['year']==2017].groupby(['month'])['e'].count().to_frame()
Year4=df[df['year']==2018].groupby(['month'])['e'].count().to_frame()
#Plotting Each Year line plot
plt.plot(Year1.index, Year1.e)
plt.plot(Year2.index, Year2.e)
plt.plot(Year3.index, Year3.e)
plt.plot(Year4.index, Year4.e)
#Formatting the lineplot
plt.legend(['2015', '2016', '2017', '2018'], loc='lower right')
plt.xlabel('Month No.')
plt.ylabel('Frequency of Call')
plt.title('Yearwise 911 calls')
plt.show()
#Setting time as Datetime object
df['time']=pd.to_datetime(df['time'], format='%H:%M:%S')
#Extracting hour out of column 'time' and storing it in a column 'time'
df['hour']=df['time'].dt.hour
df.groupby(['month','hour'])['e'].count()
#Creating hourwise 911 calls received for each month
Month1=df[df['month']==1].groupby(['hour'])['e'].count().to_frame()
Month2=df[df['month']==2].groupby(['hour'])['e'].count().to_frame()
Month3=df[df['month']==3].groupby(['hour'])['e'].count().to_frame()
Month4=df[df['month']==4].groupby(['hour'])['e'].count().to_frame()
Month5=df[df['month']==5].groupby(['hour'])['e'].count().to_frame()
Month6=df[df['month']==6].groupby(['hour'])['e'].count().to_frame()
Month7=df[df['month']==7].groupby(['hour'])['e'].count().to_frame()
Month8=df[df['month']==8].groupby(['hour'])['e'].count().to_frame()
Month9=df[df['month']==9].groupby(['hour'])['e'].count().to_frame()
Month10=df[df['month']==10].groupby(['hour'])['e'].count().to_frame()
Month11=df[df['month']==11].groupby(['hour'])['e'].count().to_frame()
Month12=df[df['month']==12].groupby(['hour'])['e'].count().to_frame()

#Plotting Each Month line plot
plt.plot(Month1.index, Month1.e)
plt.plot(Month2.index, Month2.e)
plt.plot(Month3.index, Month3.e)
plt.plot(Month4.index, Month4.e)
plt.plot(Month5.index, Month5.e)
plt.plot(Month6.index, Month6.e)
plt.plot(Month7.index, Month7.e)
plt.plot(Month8.index, Month8.e)
plt.plot(Month9.index, Month9.e)
plt.plot(Month10.index, Month10.e)
plt.plot(Month11.index, Month11.e)
plt.plot(Month12.index, Month12.e)

#Formatting the lineplot
plt.legend(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], loc='best')
plt.xlabel('Hour.')
plt.ylabel('Frequency of Call')
plt.title('Monthwise 911 calls')
figure(figsize=(100,100))
plt.show()
a=df.groupby(['zip'])['e'].count().to_frame()
b=a[a.e<50]['e'].plot.bar(figsize=(15,15), title='Zip Codes with less than 50- 911 incidents')
b.set_xlabel('ZipCodes')
b.set_ylabel('Frequency')
#Sorting Zip Codes on basis of no of incidence and selecting top 10 zip codes
b=a.sort_values(by='e', axis=0, ascending=False).head(10)['e'].plot.bar(title='Top 10 Zip Codes with 911 Incidents')
b.set_xlabel('ZipCodes')
b.set_ylabel('Frequency')
a=df['title_emr']
b=df['twp']
pd.crosstab(a, b, rownames=['a'], colnames=['b']).apply(lambda r: r*100/r.sum(), axis=0)
a=df['title_emr']
b=df['title_emrdesc']
pd.crosstab(a, b, rownames=['a'], colnames=['b']).apply(lambda r: r*100/r.sum(), axis=0)
a=df.groupby(['station'])['e'].count().to_frame()
b=a.sort_values(by='e', axis=0, ascending=False).head(10)['e'].plot.bar(title='Top 10 station with 911 Incidents')
b.set_xlabel('Station')
b.set_ylabel('Frequency')