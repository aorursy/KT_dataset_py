!conda install --yes seaborn #install seaborn library
import numpy as np              #Import Library
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
!wget -O  weatherHistory.csv https://raw.githubusercontent.com/akshayvanjare8/Data-Analytics-using-Python-Internship-Projects/main/Meteorological%20Data.csv?token=ARHGAUOI24YTCQXQDVO7ZJ27P725Y
MD = pd.read_csv("weatherHistory.csv")
#to display the top 5 rows

MD.head(5)
#to display the bottom 5 rows

MD.tail(5)
#Checking for Null Values

MD.isnull().sum()
#Checking DataTypes of columns

MD.dtypes
MD.describe()
MD.info()
#Formatting Date

MD['Formatted Date'] = pd.to_datetime(MD['Formatted Date'], utc=True)
MD['Formatted Date']
#Set index as "Date

MD = MD.set_index('Formatted Date')
MD.head()
#after resampling

data_columns = ['Apparent Temperature (C)', 'Humidity']
df_monthly_mean = MD[data_columns].resample('MS').mean()
df_monthly_mean.head()
#Plotting Variation in Apparent Temperature and Humidity with time

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.figure(figsize=(14,6))
plt.title("Variation in Apparent Temperature and Humidity with time")
sns.lineplot(data=df_monthly_mean)
#retrieving the data of a particular month from every year, say April

df1 = df_monthly_mean[df_monthly_mean.index.month==4]
print(df1)

df1.dtypes
#Plotting each years Humidity and Temperature change

import matplotlib.dates as mdates
from datetime import datetime 

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Apparent Temperature (C)'], marker='o', linestyle='-',label='Apparent Temperature (C)')
ax.plot(df1.loc['2006-04-01':'2016-04-01', 'Humidity'], marker='o', linestyle='-',label='Humidity')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend(loc = 'center right')
ax.set_xlabel('Month of April')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
#!wget -O  weather-dataset.csv https://raw.githubusercontent.com/akshayvanjare8/Data-Analytics-using-Python-Internship-Projects/main/Meteorological%20Data.csv?token=ARHGAUOI24YTCQXQDVO7ZJ27P725Y
df = pd.read_csv("weatherHistory.csv")
df.head()
titles_required = ["Formatted Date", "Apparent Temperature (C)", "Humidity", "Daily Summary"]

df1 = df[titles_required]
df1['Formatted Date'] = pd.to_datetime(df1['Formatted Date'], utc = True)

df_2 = df1.set_index('Formatted Date')

df_2 = df_2.resample('MS').mean()
df_2.head()
plt.figure(figsize = (14 , 6))
plt.title("Varition in Apparent Temperature and Humadity with time")
plt.plot(df_2)
df_april = df_2[df_2.index.month == 4]
plt.figure(figsize = (14 , 6))
plt.plot(df_april)
sns.distplot(df.Humidity, color = 'red')
plt.figure(figsize = (14 , 5))
sns.barplot(x = 'Apparent Temperature (C)', y = 'Humidity', data = df_april)
plt.xticks(rotation = -45)
sns.relplot(data = df, x = "Apparent Temperature (C)", y = "Humidity", color = 'purple')
sns.jointplot(data = df, x = "Apparent Temperature (C)", y = "Humidity")
sns.pairplot(MD)