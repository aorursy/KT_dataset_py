#Problem---Convert the vector of string representing date and time to time series data.

#Solution---We will use pandas library



#Importing numpy and pandas library

import numpy as np

import pandas as pd



#creating strings date time

date_string =np.array(['03-04-2005 11:35 PM',

                       '23-05-2010 12:01 AM',

                       '04-09-2009 09:09 PM'])



#converting to date time series

[pd.to_datetime(date,format='%d-%m-%Y %I:%M %p') for date in date_string]
#If any error occurs and we dont want to raise an error we can use erroes='coerce'

#to set the error to NaT(Not a Time i.e. missing value)

[pd.to_datetime(date, format="%d-%m-%Y %I:%M %p", errors="coerce")

for date in date_string]

#Problem---Add or change the time zone of the information.

#Solution---We will use pandas's tz



#importing pandas library

import pandas as pd



#crating datetime

pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')
#Adding time zone to previously created time

#creating datetime

date=pd.Timestamp('2017-05-01 06:00:00')



#setting time zone

date_in_london = date.tz_localize('Europe/London')



#displaying datetime

date_in_london
#Converting to different time zone

date_in_london.tz_convert('Africa/Abidjan')
#we can apply pandas series object by tz_localize and tz_convert



#creating three dates

dates=pd.Series(pd.date_range('2/2/2002', periods =3, freq ='M'))



#Setting time zone

dates.dt.tz_localize('Africa/Abidjan')
#We can use pytz library also

#importing pytz library

from pytz import all_timezones



#Displaying 5 timezones

all_timezones[::-1]
#Problem---Select one or more date and time observation

#Solution---We will use boolean conditions to select



#importing pandas library

import pandas as pd



#creating dataframe

dataframe=pd.DataFrame()



#creating datetimes

dataframe['date']  = pd.date_range('1/1/2001',periods=100000,freq='H')



#Selecting observation between two datetimes

dataframe[(dataframe['date']>'2002-1-1 01:00:00') & 

         (dataframe['date']<='2002-1-1 04:00:00')]
#Or we can use slicing 

#setting index

dataframe =dataframe.set_index(dataframe['date'])



#Selecting observation between two datetimes

dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']
#Problem---Break the datetime feature into multiple features of date and time

#Solution---We will use pandas Series.dt 



#importing pandas library

import pandas as pd



#creating dataframe

dataframe=pd.DataFrame()



#creating datetime

dataframe['date']=pd.date_range('1/1/2001',periods=150,freq='W')



#Creating features for year ,months, day, hour, and minute

dataframe['year']=dataframe['date'].dt.year

dataframe['month']=dataframe['date'].dt.month

dataframe['day']=dataframe['date'].dt.day

dataframe['hour']=dataframe['date'].dt.hour

dataframe['minute']=dataframe['date'].dt.minute
#Displaying rows

dataframe.head(5)
#Problem---Calculate the time between two observations

#Solution---We will use pandas library



#importing pandas library

import pandas as pd



#creating dataframe

dataframe = pd.DataFrame()



#Creating two datetimes features

dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]

dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]



#calculating time between features by using minus(-) sign

dataframe['Left'] - dataframe['Arrived']
#If we want to remove the days output and only keep the numerical value



#calculating time between features

pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))
#Problem---Display the days of the week of the data

#Solution---We will use Series.dt



#importing pandas library

import pandas as pd



#creating dates

dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))



#Showing the days of the week

dates.dt.weekday





#displaying with name

dates.dt.day_name()
#Problem---Create a feature that is lagged time prediods

#Solution---We will use pandas's shift 



#importing pandas library

import pandas as pd



#creating data frame

dataframe=pd.DataFrame()



#creating data

dataframe['dates']=pd.date_range('1/1/2001',periods=5,freq="D")

dataframe['stock_prize']= [1.1,2.2,3.3,4.4,5.5]

          

#Lagging value by one row

dataframe['previous_days_stock_prize']=dataframe["stock_prize"].shift(1)



          

#Displaying dataframe

dataframe
#Problem---Calculate the statistic for a rolling time

#Solution--We will use pandas's rolling



#importing pandas library

import pandas as pd



#creating datetimes

time_index = pd.date_range("01/01/2010", periods=5, freq="M")



#creating dataframe from time_index

dataframe=pd.DataFrame(index=time_index)



#creating features

dataframe["Stock_Price"]=[1,2,3,4,5]



#Calculating Rolling mean()

dataframe.rolling(window=2).mean()
#Problem---Fill the missing values in time series data

#Solution---We will use interpolation technique



#importing numpy and pandas library

import numpy as np

import pandas as pd



#creating date

time_index = pd.date_range("01/01/2010", periods=5, freq="M")



#creating dataframe

dataframe=pd.DataFrame(index=time_index)



#creating feature with gap of missing values

dataframe["Sales"] = [1.0,2.0,np.nan,np.nan,5.0]



#Using interpolation to fill the gaps

dataframe.interpolate()
#Forwardfill ---filling with last known value

dataframe.ffill()
#BackwordFilling ---filling with the latest known value

dataframe.bfill()
dataframe.interpolate(method="quadratic")
dataframe.interpolate(limit=1, limit_direction="forward")