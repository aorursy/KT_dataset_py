# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%time

DaysRents_df = pd.read_csv(os.path.join('/kaggle/input/bike-sharing-dataset', 'day.csv'))

HourlyRents_df = pd.read_csv(os.path.join('/kaggle/input/bike-sharing-dataset', 'hour.csv'))
print("DaysRents_df: {}\nHourlyRents_df: {}".format(DaysRents_df.shape, HourlyRents_df.shape))
def missing_values(data):

    total = data.isnull().sum()

    percent = (total/data.isnull().count()*100)

    miss_column = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    miss_column['Types'] = types

    return(np.transpose(miss_column)) 
missing_values(DaysRents_df)
missing_values(HourlyRents_df)
def preview_head(data):

    return(data.head())
def describe_data(data):

    return(data.describe())
DaysRents_df.rename(columns={'instant':'id','dteday':'Date','yr':'Year','mnth':'Month','weathersit':'WeatherCondition','atemp':'FeelinTemp','hum':'Humidity','cnt':'TotalRentDay'},inplace=True)

HourlyRents_df.rename(columns={'instant':'id','dteday':'Date','yr':'Year','mnth':'Month','hr':'Hours','weathersit':'WeatherCondition','atemp':'FeelinTemp','hum':'Humidity','cnt':'TotalRentHourly'},inplace=True)
preview_head(DaysRents_df)
preview_head(HourlyRents_df)
DaysRents_df['Date']=pd.to_datetime(DaysRents_df.Date)

DaysRents_df['season']=DaysRents_df.season.astype('category')

DaysRents_df['Year']=DaysRents_df.Year.astype('category')

DaysRents_df['Month']=DaysRents_df.Month.astype('category')

DaysRents_df['holiday']=DaysRents_df.holiday.astype('category')

DaysRents_df['weekday']=DaysRents_df.weekday.astype('category')

DaysRents_df['workingday']=DaysRents_df.workingday.astype('category')

DaysRents_df['WeatherCondition']=DaysRents_df.WeatherCondition.astype('category')





HourlyRents_df['Date']=pd.to_datetime(HourlyRents_df.Date)

HourlyRents_df['season']=HourlyRents_df.season.astype('category')

HourlyRents_df['Year']=HourlyRents_df.Year.astype('category')

HourlyRents_df['Month']=HourlyRents_df.Month.astype('category')

HourlyRents_df['Hours']=HourlyRents_df.Hours.astype('category')

HourlyRents_df['holiday']=HourlyRents_df.holiday.astype('category')

HourlyRents_df['weekday']=HourlyRents_df.weekday.astype('category')

HourlyRents_df['workingday']=HourlyRents_df.workingday.astype('category')

HourlyRents_df['WeatherCondition']=HourlyRents_df.WeatherCondition.astype('category')
describe_data(DaysRents_df)
describe_data(HourlyRents_df)
import matplotlib.pyplot as plt

import seaborn as sns
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to Season

sns.barplot(x='Month',y='casual',data=DaysRents_df[['Month','casual','season']],hue='season',ax=ax)

ax.set_title('Seasonal(MonthlyDistributionCasualRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to Season

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=DaysRents_df[['Month','registered','season']],hue='season',ax=ax1)

ax1.set_title('Seasonal(MonthlyDistributionRegisteredRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to Season

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentDay',data=DaysRents_df[['Month','TotalRentDay','season']],hue='season',ax=ax2)

ax2.set_title('Seasonal(MonthlyDistributionTotalRidersDaily)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to Season

sns.barplot(x='Month',y='casual',data=HourlyRents_df[['Month','casual','season']],hue='season',ax=ax)

ax.set_title('Seasonal(MonthlyDistributionCasualRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to Season

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=HourlyRents_df[['Month','registered','season']],hue='season',ax=ax1)

ax1.set_title('Seasonal(MonthlyDistributionRegisteredRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to Season

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentHourly',data=HourlyRents_df[['Month','TotalRentHourly','season']],hue='season',ax=ax2)

ax2.set_title('Seasonal(MonthlyDistributionTotalRidersHourly)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to holiday

sns.barplot(x='Month',y='casual',data=DaysRents_df[['Month','casual','holiday']],hue='holiday',ax=ax)

ax.set_title('Holidays(MonthlyDistributionCasualRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to holiday

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=DaysRents_df[['Month','registered','holiday']],hue='holiday',ax=ax1)

ax1.set_title('Holidays(MonthlyDistributionRegisteredRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to holiday

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentDay',data=DaysRents_df[['Month','TotalRentDay','holiday']],hue='holiday',ax=ax2)

ax2.set_title('Holidays(MonthlyDistributionTotalRidersDaily)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to holiday

sns.barplot(x='Month',y='casual',data=HourlyRents_df[['Month','casual','holiday']],hue='holiday',ax=ax)

ax.set_title('Holidays(MonthlyDistributionCasualRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to holiday

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=HourlyRents_df[['Month','registered','holiday']],hue='holiday',ax=ax1)

ax1.set_title('Holidays(MonthlyDistributionRegisteredRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to holiday

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentHourly',data=HourlyRents_df[['Month','TotalRentHourly','holiday']],hue='holiday',ax=ax2)

ax2.set_title('Holidays(MonthlyDistributionTotalRidersHourly)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to weekday

sns.barplot(x='Month',y='casual',data=DaysRents_df[['Month','casual','weekday']],hue='weekday',ax=ax)

ax.set_title('Weekdays(MonthlyDistributionCasualRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to weekday

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=DaysRents_df[['Month','registered','weekday']],hue='weekday',ax=ax1)

ax1.set_title('Weekdays(MonthlyDistributionRegisteredRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to weekday

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentDay',data=DaysRents_df[['Month','TotalRentDay','weekday']],hue='weekday',ax=ax2)

ax2.set_title('Weekdays(MonthlyDistributionTotalRidersDaily)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to weekday

sns.barplot(x='Month',y='casual',data=HourlyRents_df[['Month','casual','weekday']],hue='weekday',ax=ax)

ax.set_title('Weekdays(MonthlyDistributionCasualRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to weekday

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=HourlyRents_df[['Month','registered','weekday']],hue='weekday',ax=ax1)

ax1.set_title('Weekdays(MonthlyDistributionRegisteredRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to weekday

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentHourly',data=HourlyRents_df[['Month','TotalRentHourly','weekday']],hue='weekday',ax=ax2)

ax2.set_title('Weekdays(MonthlyDistributionTotalRidersHourly)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to workingday

sns.barplot(x='Month',y='casual',data=DaysRents_df[['Month','casual','workingday']],hue='workingday',ax=ax)

ax.set_title('Workingdays(MonthlyDistributionCasualRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to workingday

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=DaysRents_df[['Month','registered','workingday']],hue='workingday',ax=ax1)

ax1.set_title('Workingdays(MonthlyDistributionRegisteredRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to workingday

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentDay',data=DaysRents_df[['Month','TotalRentDay','workingday']],hue='workingday',ax=ax2)

ax2.set_title('Workingdays(MonthlyDistributionTotalRidersDaily)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to workingday

sns.barplot(x='Month',y='casual',data=HourlyRents_df[['Month','casual','workingday']],hue='workingday',ax=ax)

ax.set_title('Workingdays(MonthlyDistributionCasualRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to workingday

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=HourlyRents_df[['Month','registered','workingday']],hue='workingday',ax=ax1)

ax1.set_title('Workingdays(MonthlyDistributionRegisteredRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to workingday

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentHourly',data=HourlyRents_df[['Month','TotalRentHourly','workingday']],hue='workingday',ax=ax2)

ax2.set_title('Workingdays(MonthlyDistributionTotalRidersHourly)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to weatherCondition

sns.barplot(x='Month',y='casual',data=DaysRents_df[['Month','casual','WeatherCondition']],hue='WeatherCondition',ax=ax)

ax.set_title('WeatherCondition(MonthlyDistributionCasualRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to weatherCondition

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=DaysRents_df[['Month','registered','WeatherCondition']],hue='WeatherCondition',ax=ax1)

ax1.set_title('WeatherCondition(MonthlyDistributionRegisteredRidersDaily)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to weatherCondition

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentDay',data=DaysRents_df[['Month','TotalRentDay','WeatherCondition']],hue='WeatherCondition',ax=ax2)

ax2.set_title('WeatherCondition(MonthlyDistributionTotalRidersDaily)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to weatherCondition

sns.barplot(x='Month',y='casual',data=HourlyRents_df[['Month','casual','WeatherCondition']],hue='WeatherCondition',ax=ax)

ax.set_title('WeatherCondition(MonthlyDistributionCasualRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to weatherCondition

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=HourlyRents_df[['Month','registered','WeatherCondition']],hue='WeatherCondition',ax=ax1)

ax1.set_title('WeatherCondition(MonthlyDistributionRegisteredRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to weatherCondition

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentHourly',data=HourlyRents_df[['Month','TotalRentHourly','WeatherCondition']],hue='WeatherCondition',ax=ax2)

ax2.set_title('WeatherCondition(MonthlyDistributionTotalRidersHourly)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar-plot for monthly distribution of CasualRiders in respect to weatherCondition

sns.barplot(x='Month',y='casual',data=HourlyRents_df[['Month','casual','Hours']],hue='Hours',ax=ax)

ax.set_title('WeatherCondition(MonthlyDistributionCasualRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of RegisteredRiders in respect to weatherCondition

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='registered',data=HourlyRents_df[['Month','registered','Hours']],hue='Hours',ax=ax1)

ax1.set_title('WeatherCondition(MonthlyDistributionRegisteredRidersHourly)')

plt.show()



#Bar-plot for monthly distribution of TotalRentals in a day in respect to weatherCondition

fig,ax2=plt.subplots(figsize=(15,8))

sns.barplot(x='Month',y='TotalRentHourly',data=HourlyRents_df[['Month','TotalRentHourly','Hours']],hue='Hours',ax=ax2)

ax2.set_title('WeatherCondition(MonthlyDistributionTotalRidersHourly)')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders

sns.violinplot(x='Year',y='casual',data=DaysRents_df[['Year','casual']])

ax.set_title('Yearly distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders

sns.violinplot(x='Year',y='registered',data=DaysRents_df[['Year','registered']])

ax.set_title('Yearly distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders

sns.violinplot(x='Year',y='TotalRentDay',data=DaysRents_df[['Year','TotalRentDay']])

ax.set_title('Yearly distribution of TotalRidersDays')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders

sns.violinplot(x='Year',y='casual',data=HourlyRents_df[['Year','casual']])

ax.set_title('Yearly distribution of CasualRidersHourly')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders

sns.violinplot(x='Year',y='registered',data=HourlyRents_df[['Year','registered']])

ax.set_title('Yearly distribution of RegisteredRidersHourly')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of riders

sns.violinplot(x='Year',y='TotalRentHourly',data=HourlyRents_df[['Year','TotalRentHourly']])

ax.set_title('Yearly distribution of TotalRidersHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders

sns.violinplot(x='season',y='casual',data=DaysRents_df[['season','casual']])

ax.set_title('Seasonal distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders

sns.violinplot(x='season',y='registered',data=DaysRents_df[['season','registered']])

ax.set_title('Seasonal distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders

sns.violinplot(x='season',y='TotalRentDay',data=DaysRents_df[['season','TotalRentDay']])

ax.set_title('Seasonal distribution of TotalRidersDays')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders

sns.violinplot(x='season',y='casual',data=HourlyRents_df[['season','casual']])

ax.set_title('Seasonal distribution of CasualRidersHourly')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders

sns.violinplot(x='season',y='registered',data=HourlyRents_df[['season','registered']])

ax.set_title('Seasonal distribution of RegisteredHourly')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for seasonal distribution of riders

sns.violinplot(x='season',y='TotalRentHourly',data=HourlyRents_df[['season','TotalRentHourly']])

ax.set_title('Seasonal distribution of TotalRidersHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders

sns.violinplot(x='holiday',y='casual',data=DaysRents_df[['holiday','casual']])

ax.set_title('Holiday distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders

sns.violinplot(x='holiday',y='registered',data=DaysRents_df[['holiday','registered']])

ax.set_title('Holiday distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders

sns.violinplot(x='holiday',y='TotalRentDay',data=DaysRents_df[['holiday','TotalRentDay']])

ax.set_title('Holiday distribution of TotalRidersDays')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders

sns.violinplot(x='holiday',y='casual',data=HourlyRents_df[['holiday','casual']])

ax.set_title('Holiday distribution of CasualRidersHourly')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders

sns.violinplot(x='holiday',y='registered',data=HourlyRents_df[['holiday','registered']])

ax.set_title('Holiday distribution of RegisteredRidersHourly')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for holiday distribution of riders

sns.violinplot(x='holiday',y='TotalRentHourly',data=HourlyRents_df[['holiday','TotalRentHourly']])

ax.set_title('Holiday distribution of TotalRidersHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='weekday',y='casual',data=DaysRents_df[['weekday','casual']])

ax.set_title('Weekday distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='weekday',y='registered',data=DaysRents_df[['weekday','registered']])

ax.set_title('Weekday distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='weekday',y='TotalRentDay',data=DaysRents_df[['weekday','TotalRentDay']])

ax.set_title('Weekday distribution of TotalRidersDays')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='weekday',y='casual',data=HourlyRents_df[['weekday','casual']])

ax.set_title('Weekday distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='weekday',y='registered',data=HourlyRents_df[['weekday','registered']])

ax.set_title('Weekday distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='weekday',y='TotalRentHourly',data=HourlyRents_df[['weekday','TotalRentHourly']])

ax.set_title('Weekday distribution of TotalRidersHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders

sns.violinplot(x='workingday',y='casual',data=DaysRents_df[['workingday','casual']])

ax.set_title('Workingday distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders

sns.violinplot(x='workingday',y='registered',data=DaysRents_df[['workingday','registered']])

ax.set_title('Workingday distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='workingday',y='TotalRentDay',data=DaysRents_df[['workingday','TotalRentDay']])

ax.set_title('Workingday distribution of TotalRidersDays')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders

sns.violinplot(x='workingday',y='casual',data=HourlyRents_df[['workingday','casual']])

ax.set_title('Workingday distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for workingdays distribution of riders

sns.violinplot(x='workingday',y='registered',data=HourlyRents_df[['workingday','registered']])

ax.set_title('Workingday distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for weekdays distribution of riders

sns.violinplot(x='workingday',y='TotalRentHourly',data=HourlyRents_df[['workingday','TotalRentHourly']])

ax.set_title('Workingday distribution of TotalRidersHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='WeatherCondition',y='casual',data=DaysRents_df[['WeatherCondition','casual']])

ax.set_title('WeatherCondition distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='WeatherCondition',y='registered',data=DaysRents_df[['WeatherCondition','registered']])

ax.set_title('WeatherCondition distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='WeatherCondition',y='TotalRentDay',data=DaysRents_df[['WeatherCondition','TotalRentDay']])

ax.set_title('WeatherCondition distribution of TotalRidersDays')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='WeatherCondition',y='casual',data=HourlyRents_df[['WeatherCondition','casual']])

ax.set_title('WeatherCondition distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='WeatherCondition',y='registered',data=HourlyRents_df[['WeatherCondition','registered']])

ax.set_title('WeatherCondition distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='WeatherCondition',y='TotalRentHourly',data=HourlyRents_df[['WeatherCondition','TotalRentHourly']])

ax.set_title('WeatherCondition distribution of TotalRidersHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='Hours',y='casual',data=HourlyRents_df[['Hours','casual']])

ax.set_title('Hourly distribution of CasualRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='Hours',y='registered',data=HourlyRents_df[['Hours','registered']])

ax.set_title('Hourly distribution of RegisteredRiders')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for WeatherCondition distribution of riders

sns.violinplot(x='Hours',y='TotalRentHourly',data=HourlyRents_df[['Hours','TotalRentHourly']])

ax.set_title('Hourly distribution of TotalRidersHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=DaysRents_df,x='holiday',y='casual',hue='season')

ax.set_title('Holiday wise distribution of casual')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=DaysRents_df,x='holiday',y='registered',hue='season')

ax.set_title('Holiday wise distribution of registered')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=DaysRents_df,x='holiday',y='TotalRentDay',hue='season')

ax.set_title('Holiday wise distribution of TotalRentDay')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='holiday',y='casual',hue='season')

ax.set_title('Holiday wise distribution of casual')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='holiday',y='registered',hue='season')

ax.set_title('Holiday wise distribution of registered')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='holiday',y='TotalRentHourly',hue='season')

ax.set_title('Holiday wise distribution of TotalRentHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Workingday distribution of riders with season

sns.barplot(data=DaysRents_df,x='workingday',y='casual',hue='season')

ax.set_title('Workingday wise distribution of casual')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Workingday distribution of riders with season

sns.barplot(data=DaysRents_df,x='workingday',y='registered',hue='season')

ax.set_title('Workingday wise distribution of registered')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=DaysRents_df,x='workingday',y='TotalRentDay',hue='season')

ax.set_title('Workingday wise distribution of TotalRentDay')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Workingday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='workingday',y='casual',hue='season')

ax.set_title('Workingday wise distribution of casual')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Workingday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='workingday',y='registered',hue='season')

ax.set_title('Workingday wise distribution of registered')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='workingday',y='TotalRentHourly',hue='season')

ax.set_title('Workingday wise distribution of TotalRentHourly')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Workingday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='workingday',y='casual',hue='Hours')

ax.set_title('Hours wise distribution of casual')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Workingday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='workingday',y='registered',hue='Hours')

ax.set_title('Hours wise distribution of registered')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of riders with season

sns.barplot(data=HourlyRents_df,x='workingday',y='TotalRentHourly',hue='Hours')

ax.set_title('Hours wise distribution of TotalRentDay')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for casuals outliers

sns.boxplot(data=DaysRents_df[['casual']])

ax.set_title('casual outliers')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for registered outliers

sns.boxplot(data=DaysRents_df[['registered']])

ax.set_title('registered outliers')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for TotalRiders outliers

sns.boxplot(data=DaysRents_df[['TotalRentDay']])

ax.set_title('TotalRentDay outliers')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for casuals outliers

sns.boxplot(data=HourlyRents_df[['casual']])

ax.set_title('casual outliers')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for registered outliers

sns.boxplot(data=HourlyRents_df[['registered']])

ax.set_title('registered outliers')

plt.show()



fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for TotalRiders outliers

sns.boxplot(data=HourlyRents_df[['TotalRentHourly']])

ax.set_title('TotalRentHourly outliers')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for casuals outliers

sns.boxplot(data=DaysRents_df[['temp','FeelinTemp','windspeed','Humidity']])

ax.set_title('temp_FeelingTemperature_windspeed_humidity_outliers')

plt.show()

fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for casuals outliers

sns.boxplot(data=HourlyRents_df[['temp','FeelinTemp','windspeed','Humidity']])

ax.set_title('temp_FeelingTemperature_windspeed_humidity_outliers')

plt.show()
#create dataframe for outliers

CorrectionOutliers = pd.DataFrame(DaysRents_df,columns=['windspeed','Humidity'])

#ColNames for outliers                     

ColNames=['windspeed','Humidity']       

                      

for Col in ColNames:

    q75,q25 = np.percentile(CorrectionOutliers.loc[:,Col],[75,25]) # Divide data into 75%quantile and 25%quantile.

    iqr = q75-q25 #Inter quantile range

    min = q25-(iqr*1.5) #inner fence

    max = q75+(iqr*1.5) #outer fence

    CorrectionOutliers.loc[CorrectionOutliers.loc[:,Col] < min,:Col] = np.nan  #Replace with NA

    CorrectionOutliers.loc[CorrectionOutliers.loc[:,Col] > max,:Col] = np.nan  #Replace with NA

#Replacing the outliers by mean values

CorrectionOutliers['windspeed'] = CorrectionOutliers['windspeed'].fillna(CorrectionOutliers['windspeed'].mean())

CorrectionOutliers['Humidity'] = CorrectionOutliers['Humidity'].fillna(CorrectionOutliers['Humidity'].mean())
#create dataframe for outliers

CorrectionOutlierss = pd.DataFrame(HourlyRents_df,columns=['windspeed','Humidity'])

#ColNames for outliers                     

ColNamess=['windspeed','Humidity']       

                      

for Col in ColNamess:

    q75,q25 = np.percentile(CorrectionOutlierss.loc[:,Col],[75,25]) # Divide data into 75%quantile and 25%quantile.

    iqr = q75-q25 #Inter quantile range

    min = q25-(iqr*1.5) #inner fence

    max = q75+(iqr*1.5) #outer fence

    CorrectionOutlierss.loc[CorrectionOutlierss.loc[:,Col] < min,:Col] = np.nan  #Replace with NA

    CorrectionOutlierss.loc[CorrectionOutlierss.loc[:,Col] > max,:Col] = np.nan  #Replace with NA

#Replacing the outliers by mean values

CorrectionOutlierss['windspeed'] = CorrectionOutlierss['windspeed'].fillna(CorrectionOutlierss['windspeed'].mean())

CorrectionOutlierss['Humidity'] = CorrectionOutlierss['Humidity'].fillna(CorrectionOutlierss['Humidity'].mean())
#Replacing the WindspeedOutliers

DaysRents_df['windspeed']=DaysRents_df['windspeed'].replace(CorrectionOutliers['windspeed'])

#Replacing the HumidityOutliers

DaysRents_df['Humidity']=DaysRents_df['Humidity'].replace(CorrectionOutliers['Humidity'])

DaysRents_df.head(5)
#Replacing the WindspeedOutliers

HourlyRents_df['windspeed']=HourlyRents_df['windspeed'].replace(CorrectionOutlierss['windspeed'])

#Replacing the HumidityOutliers

HourlyRents_df['Humidity']=HourlyRents_df['Humidity'].replace(CorrectionOutlierss['Humidity'])

HourlyRents_df.head(5)
fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for casuals outliers

sns.boxplot(data=DaysRents_df[['temp','FeelinTemp','windspeed','Humidity']])

ax.set_title('temp_FeelingTemperature_windspeed_humidity_outliers')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for casuals outliers

sns.boxplot(data=HourlyRents_df[['temp','FeelinTemp','windspeed','Humidity']])

ax.set_title('temp_FeelingTemperature_windspeed_humidity_outliers')

plt.show()
import matplotlib.pyplot as plt

plt.hist(DaysRents_df['temp'], bins=30)

plt.xlabel('temperature(°C)')

plt.ylabel('fraction of temperature')

plt.show()



plt.hist(DaysRents_df['FeelinTemp'], bins=30)

plt.xlabel('FeelingTemp(°C)')

plt.ylabel('fraction of temperature')

plt.show()



plt.hist(DaysRents_df['Humidity'], bins=30)

plt.xlabel('Humidity')

plt.ylabel('fraction of humidity')

plt.show()



import matplotlib.pyplot as plt

plt.hist(DaysRents_df['windspeed'], bins=30)

plt.xlabel('WindSpeed')

plt.ylabel('fraction of windspeed')

plt.show()



plt.hist(DaysRents_df['casual'], bins=30)

plt.xlabel('casual')

plt.ylabel('fraction of casual')



plt.hist(DaysRents_df['registered'], bins=30)

plt.xlabel('registered')

plt.ylabel('fraction of registered')

plt.show()



plt.hist(DaysRents_df['TotalRentDay'], bins=30)

plt.xlabel('TotalRentDay')

plt.ylabel('fraction of TotalRentDay')

plt.show()
import matplotlib.pyplot as plt

plt.hist(HourlyRents_df['temp'], bins=30)

plt.xlabel('temperature(°C)')

plt.ylabel('fraction of temperature')

plt.show()



plt.hist(HourlyRents_df['FeelinTemp'], bins=30)

plt.xlabel('FeelingTemp(°C)')

plt.ylabel('fraction of temperature')

plt.show()



plt.hist(HourlyRents_df['Humidity'], bins=30)

plt.xlabel('Humidity')

plt.ylabel('fraction of humidity')

plt.show()



plt.hist(HourlyRents_df['windspeed'], bins=30)

plt.xlabel('WindSpeed')

plt.ylabel('fraction of windspeed')

plt.show()



plt.hist(HourlyRents_df['casual'], bins=30)

plt.xlabel('casual')

plt.ylabel('fraction of casual')

plt.show()



plt.hist(HourlyRents_df['registered'], bins=30)

plt.xlabel('registered')

plt.ylabel('fraction of registered')

plt.show()



plt.hist(HourlyRents_df['TotalRentHourly'], bins=30)

plt.xlabel('TotalRentHourly')

plt.ylabel('fraction of TotalRentHourly')

plt.show()
def Pmf(data):

    return data.value_counts().sort_index()/len(data)
BusyDay = DaysRents_df['workingday'] == 1

SumRider = DaysRents_df['TotalRentDay']

BusyDay_SumRider = SumRider[BusyDay]

IdleDay_SumRider = SumRider[~BusyDay]

Pmf(BusyDay_SumRider).plot(label='WorkingDays')

Pmf(IdleDay_SumRider).plot(label='WorkingDays')

plt.xlabel('SumRider(cnt)')

plt.ylabel('Count')
import scipy

from scipy import stats

#Normal plot

fig=plt.figure(figsize=(15,8))

stats.probplot(DaysRents_df.casual.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()



fig=plt.figure(figsize=(15,8))

stats.probplot(DaysRents_df.registered.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()



fig=plt.figure(figsize=(15,8))

stats.probplot(DaysRents_df.TotalRentDay.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()
import scipy

from scipy import stats

#Normal plot

fig=plt.figure(figsize=(15,8))

stats.probplot(HourlyRents_df.casual.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()



fig=plt.figure(figsize=(15,8))

stats.probplot(HourlyRents_df.registered.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()



fig=plt.figure(figsize=(15,8))

stats.probplot(HourlyRents_df.TotalRentHourly.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()
fig=plt.figure(figsize=(15,8))

stats.probplot(HourlyRents_df.windspeed.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()



fig=plt.figure(figsize=(15,8))

stats.probplot(HourlyRents_df.Humidity.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()



fig=plt.figure(figsize=(15,8))

stats.probplot(HourlyRents_df.FeelinTemp.tolist(),dist='norm',plot=plt)

plt.xlabel("Normality", labelpad=30)

plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)

plt.show()
correMtr=DaysRents_df[["temp","FeelinTemp","Humidity","windspeed","casual","registered","TotalRentDay"]].corr()

mask=np.array(correMtr)

mask[np.tril_indices_from(mask)]=False

#Heat map for correlation matrix of attributes

fig,ax=plt.subplots(figsize=(15,8))

sns.heatmap(correMtr,mask=mask,vmax=0.8,square=True,annot=True,ax=ax)

ax.set_title('Correlation matrix of attributes')

plt.show()
correMtr=HourlyRents_df[["temp","FeelinTemp","Humidity","windspeed","casual","registered","TotalRentHourly"]].corr()

mask=np.array(correMtr)

mask[np.tril_indices_from(mask)]=False

#Heat map for correlation matrix of attributes

fig,ax=plt.subplots(figsize=(15,8))

sns.heatmap(correMtr,mask=mask,vmax=0.8,square=True,annot=True,ax=ax)

ax.set_title('Correlation matrix of attributes')

plt.show()
#load the required libraries

from sklearn import preprocessing,metrics,linear_model

from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split

#Split the dataset into the train and test data

xrain,xest,yrain,yest=train_test_split(HourlyRents_df.iloc[:,0:-3],HourlyRents_df.iloc[:,-1],test_size=0.2, random_state=40)



#Reset train index values

xrain.reset_index(inplace=True)

yrain=yrain.reset_index()



# Reset train index values

xest.reset_index(inplace=True)

yest=yest.reset_index()



print("xrain: {}\nxest: {}\nyrain: {}\nyest: {}".format(xrain.shape,xest.shape,yrain.shape,yest.shape))

print(yrain.head())

print(yest.head())



print(xrain.head())

print(xest.head())
fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(HourlyRents_df['TotalRentHourly'],HourlyRents_df['FeelinTemp'])

# ax.axhline(lw=2,color='black')

ax.set_title('Relationship Btw FeelinTemp and Casuals')

ax.set_xlabel('Casual')

ax.set_ylabel('FeelingTemp')

plt.show()
#Create a new dataset for train attributes

train_attr=xrain[['season','Month','Year','weekday','holiday','workingday','WeatherCondition','Humidity','FeelinTemp','windspeed','Hours']]

#Create a new dataset for test attributes

test_attr=xest[['season','Month','Year','weekday','holiday','workingday','Humidity','FeelinTemp','windspeed','WeatherCondition','Hours']]

#categorical attributes

cat_attr=['season','holiday','workingday','WeatherCondition','Year','Hours']

#numerical attributes

num_attr=['FeelinTemp','windspeed','Humidity','Month','weekday']
#To get dummy variables to encode the categorical features to numeric

EncodedXrain=pd.get_dummies(train_attr,columns=cat_attr)

print("Shape of the Encoded xrain: {}".format(EncodedXrain.shape))

EncodedXrain.head()
#Training dataset for modelling

Xrain=EncodedXrain

Yrain=yrain.TotalRentHourly.values



lr_model=linear_model.LinearRegression()

lr_model.fit(Xrain,Yrain)



lr_model.fit(Xrain,Yrain)

lr=lr_model.score(Xrain,Yrain)

print("Accuracy of lr: {}\nlr coef: {}\nlr Intercept Value: {}".format(lr,lr_model.coef_,lr_model.intercept_))



Forcast = cross_val_predict(lr_model,Xrain,Yrain,cv=10)

Forcast
fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(Yrain,Yrain-Forcast)

ax.axhline(lw=2,color='black')

ax.set_title('Cross validation prediction plot')

ax.set_xlabel('Forcast')

ax.set_ylabel('Residual')

plt.show()



r2_scores = cross_val_score(lr_model, Xrain, Yrain, cv=10)

print('R-squared scores :',np.average(r2_scores))
EncodedYrain=pd.get_dummies(test_attr,columns=cat_attr)

print('Shape of transformed dataframe :',test_attr.shape)

test_attr.head()



Xest=EncodedYrain

Yest=yest.TotalRentHourly.values



lr_Forcast=lr_model.predict(Xest)

lr_Forcast



import math

#Root mean square error 

rmse=math.sqrt(metrics.mean_squared_error(Yest,lr_Forcast))

#Mean absolute error

mae=metrics.mean_absolute_error(Yest,lr_Forcast)

print('Root mean square error :',rmse)

print('Mean absolute error :',mae)



fig, ax = plt.subplots(figsize=(15,8))

ax.scatter(Yest, Yest-lr_Forcast)

ax.axhline(lw=2,color='black')

ax.set_xlabel('Forcast')

ax.set_ylabel('Residuals')

ax.title.set_text("Residual Plot")

plt.show()
from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(min_samples_split=2,max_leaf_nodes=200)



dtr.fit(Xrain,Yrain)

dtr_score=dtr.score(Xrain,Yrain)

print('Accuracy of model :',dtr_score)



from sklearn import tree

import pydot

import graphviz



# export the learned model to tree

dot_data = tree.export_graphviz(dtr, out_file=None) 

graph = graphviz.Source(dot_data) 

graph



Forecasted=cross_val_predict(dtr,Xrain,Yrain,cv=3)

Forecasted



fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(Yrain,Yrain-Forecasted)

ax.axhline(lw=2,color='black')

ax.set_title('Cross validation prediction plot')

ax.set_xlabel('Forcasted')

ax.set_ylabel('Residual')

plt.show()



r2_scores = cross_val_score(dtr, Xrain, Yrain, cv=3)

print('R-squared scores :',np.average(r2_scores))
dtr_Forecasted=dtr.predict(Xest)

dtr_Forecasted



rmse=math.sqrt(metrics.mean_squared_error(Yest,dtr_Forecasted))

#Mean absolute error

mae=metrics.mean_absolute_error(Yest,dtr_Forecasted)

print('Root mean square error :',rmse)

print('Mean absolute error :',mae)



residuals = Yest-dtr_Forecasted

fig, ax = plt.subplots(figsize=(15,8))

ax.scatter(Yest, residuals)

ax.axhline(lw=2,color='black')

ax.set_xlabel('Observed')

ax.set_ylabel('Residual')

ax.set_title('Residual plot')

plt.show()
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100)



rf.fit(Xrain,Yrain)

rf_score =rf.score(Xrain,Yrain)

print('Accuracy of the model :',rf_score)



Forcasted=cross_val_predict(rf,Xrain,Yrain,cv=3)

Forcasted



fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(Yrain,Yrain-Forcasted)

ax.axhline(lw=2,color='black')

ax.set_title('Cross validation prediction plot')

ax.set_xlabel('Forcasted')

ax.set_ylabel('Residual')

plt.show()



r2_scores = cross_val_score(rf, Xrain, Yrain, cv=3)

print('R-squared scores :',np.average(r2_scores))
rf_Forcast=rf.predict(Xest)

rf_Forcast



rmse = math.sqrt(metrics.mean_squared_error(Yest,rf_Forcast))

print('Root mean square error :',rmse)

#Mean absolute error

mae=metrics.mean_absolute_error(Yest,rf_Forcast)

print('Mean absolute error :',mae)



fig, ax = plt.subplots(figsize=(15,8))

residuals=Yest-rf_Forcast

ax.scatter(Yest, residuals)

ax.axhline(lw=2,color='black')

ax.set_xlabel('Forcast')

ax.set_ylabel('Residuals')

ax.set_title('Residual plot')

plt.show()
HourlyRiders_df1 = pd.DataFrame(Yest,columns=['Yest'])

HourlyRiders_df2 = pd.DataFrame(rf_Forcast,columns=['rf_Forcast'])

HourlyRiders_Forcasting = pd.merge(HourlyRiders_df1,HourlyRiders_df2,left_index=True,right_index=True)

HourlyRiders_Forcasting.to_csv('HourlyBike_Renting_Python.csv')

HourlyRiders_Forcasting