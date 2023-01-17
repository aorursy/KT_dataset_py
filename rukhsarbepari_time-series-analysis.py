import datetime as dt
d = dt.datetime.now()
print(dt.datetime.now())
dir(dt)
#Present date only
date = dt.date.today()
print(date)
#only time to be printed
print(d.now().time())
#Spliting the date
print(f'Year = {date.year}')
print(f'Month = {date.month}')
print(f'Day = {date.day}')
from datetime import time
a = time(11,29,45)
print(a)
print(type(a))
#split hours ,min and seconds
import pandas as pd

df = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df.head()
df.info()
df['DATE_TIME'][0]
#Convert the DATE_TIME column from string datatype to Datetime datatype
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df.info()
df
#Spliting DATE_TIME column INTO date and time columns
df['DATE'] = pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
df['TIME'] = pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
df.info()
