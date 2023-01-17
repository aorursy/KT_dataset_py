import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')

#importing data
data1=pd.read_csv("../input/Chicago_Crimes_2012_to_2017.csv")
data2=pd.read_csv("../input/Chicago_Crimes_2008_to_2011.csv",error_bad_lines=False)

#viewing data
data1.head()

data2.head()
#cheecking the data type  whether it is strng or intiger bcz in the next block I would have to write my code according to string/integer
data2.Year.dtype
#Acessing values only for the year 2010 and 2011 from data2 which contained values from the year 2008 onwards and checking the difference in their shape
data2_new=data2[(data2['Year']==2010)|(data2['Year']==2011)]
print(data2.shape)
print(data2_new.shape)
#assigning the values of data1 and data_new into crime and removing the duplicate values also checking the difference in their shape

crimes = pd.concat([data1, data2_new], ignore_index=False, axis=0)

del data1
del data2_new

print('Dataset ready..')

print('Dataset Shape before drop_duplicate : ', crimes.shape)
crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print('Dataset Shape after drop_duplicate: ', crimes.shape)

#checking the first 5 elements of the new Data set
crimes.head()
#checking the last 5 elements of the new Dataset
crimes.tail()
# convert dates to pandas datetime format and setting the index to be the date will help us a lot later on
crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')

crimes.index = pd.DatetimeIndex(crimes.Date)

crimes.tail()
              
   
#checking the types of columns
crimes.info()

#Exploration and visualization
#Qstn answered:How maany crimes per month between the year 2010-2017
plt.figure(figsize=(12,6))
crimes.resample('M').size().plot()
plt.title('Number of crimes per month (2010 - 2017)')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()
#now let's see if the sum of all the crime is decresing over the period of time
plt.figure(figsize=(12,6))
crimes.resample('D').size().rolling(365).sum().plot()
plt.title('Sum of all crimes from 2010 - 2017')
plt.xlabel('Days')
plt.ylabel('Number of crimes')
plt.show()

#below diag shows a decrease in the overall crime rate
#now let's seperate crime by it's type 
crimes_count_date = crimes.pivot_table('ID', aggfunc=np.size, columns='Primary Type',
                                       index=crimes.index.date, fill_value=0)
crimes_count_date.index = pd.DatetimeIndex(crimes_count_date.index)
plo = crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), 
                                                subplots=True, layout=(-1, 3), 
                                                sharex=False, sharey=False)

#if we were to only believe the previous graph we would have been wrong since some of the crimes have actually 
#incresed over the period of time
#Crimes like Concealed carry license violation,Deceptive practice,Human trafficing etc have show an increasing trend

days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
crimes.groupby([crimes.index.dayofweek]).size().plot(kind='barh')
plt.ylabel('Days of the week')
plt.yticks(np.arange(7), days)
plt.xlabel('Number of crimes')
plt.title('Number of crimes by day of the week')
plt.show()
#from the below diag we can see that maximum no of crime occur on Friday
#Now,lets look at crimes per month
crimes.groupby([crimes.index.month]).size().plot(kind='barh')
plt.ylabel('Months of the year')
plt.xlabel('Number of crimes')
plt.title('Number of crimes by month of the year')
plt.show()


#Now lets see which crimes occur more frequently
plt.figure(figsize=(10,10))
crimes.groupby([crimes['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by type')
plt.ylabel('Crime Type')
plt.xlabel('Number of crimes')
plt.show()
#Now plotting based on location
plt.figure(figsize=(8,30))
crimes.groupby([crimes['Location Description']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by Location')
plt.ylabel('Crime Location')
plt.xlabel('Number of crimes')
plt.show()