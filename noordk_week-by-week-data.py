#python using 2.7

#this karnel will return for you a table that has the average unit_sales per week.

#Moreover it will return the average of a certain week(for all years) for a specific store

#those tables can help you develop a moving average forecast by week.





import csv

import datetime

import pandas as pd



train = pd.read_csv('train.csv')

#converting to datetime

train['date']= pd.to_datetime(train.date)

#adding yerar and week 

train['week'] = train['date'].dt.week

train['year'] = train['date'].dt.year



#getting the mean for a week in each year

weekly_beh_yearly = train.groupby(['item_nbr','year','week','store_nbr'],as_index= False) ['unit_sales'].mean()

#getting the mean for a week along all years

weekly_beh_avr = train.groupby(['item_nbr', 'week','store_nbr'],as_index=False)['unit_sales'].mean()



#this way you will deal with data based on weeks instead of days.


