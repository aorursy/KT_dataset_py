import numpy as np
import pandas as pd 
import seaborn as sns
import calendar
import matplotlib.pylab as plt
%matplotlib inline
import re
import gc
import glob


path ='../input/2016-parking-tickets-data/'
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
tickets = pd.concat(list_)
tickets.head()
tickets.dtypes
tickets['month_num'] = pd.to_datetime(tickets['date_of_infraction'],format='%Y%m%d').dt.month
tickets ['month'] =tickets['month_num'].apply(lambda x: calendar.month_abbr[x])
tickets['month'].value_counts().plot(kind='bar')
plt.show()
cntIns = tickets.groupby(['date_of_infraction']).size().reset_index(name='count')

x = pd.DataFrame(pd.to_datetime(cntIns['date_of_infraction'],format='%Y%m%d').dt.date)
y = pd.DataFrame(cntIns['count'])

timePlot = pd.concat([x,y], axis=1)

cntObs = timePlot['count'].sum() # count of observations
cntDays = y.shape[0] # count of days

minDate = timePlot['date_of_infraction'].min() # date of first observation
maxDate = timePlot['date_of_infraction'].max() # datet of last observation

dateRange = re.split('\,', str(maxDate - minDate))
dateRange = dateRange[0]

print("\n\nThe data includes", "{:,}".format(cntObs), "tickets given out across", cntDays, "days. The date range \nspans", dateRange, "from", minDate, "to", maxDate, ".\n")

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(111)
ax.set(xlabel='Date', ylabel='# Tickets')
ax.plot_date(x=timePlot['date_of_infraction'], y=timePlot['count'],ls='-', marker='o')
plt.show()
del x, y, timePlot, fig, ax
gc.collect()
print(tickets['infraction_description'].describe())
pd.crosstab(index=tickets['infraction_description'], columns='count').nlargest(10,'count')

tickets['infraction_description'].value_counts().nlargest(10).plot(kind='barh',width = 0.75,figsize=(13,6))
plt.title("Number of Infractions by Category", fontsize=16)
plt.ylabel("Infraction", fontsize=18)
plt.xlabel("No. of infractions", fontsize=18)
plt.show()



print(tickets['province'].describe())
print('Unique States:\n',tickets['province'].unique())
pd.crosstab(index=tickets['province'], columns='count').nlargest(10,'count')
tickets['province'].value_counts().plot(kind='bar',figsize=(13,6),width = 0.75)
plt.title("Number of Infractions by Province/State", fontsize=16)
plt.xlabel("Province/State", fontsize=18)
plt.ylabel("No. of infractions", fontsize=18)
plt.show()
tickets['locality'] = tickets['province'].apply(lambda x: 'Ontario' if x=='ON' else 'Other')
pd.crosstab(index=tickets['locality'], columns='count')
tickets['locality'].value_counts().plot(kind='barh',figsize=(13,6),width = 0.75)
plt.title("Number of Infractions by Locality", fontsize=16)
plt.xlabel("Locality", fontsize=18)
plt.ylabel("No. of infractions", fontsize=18)
plt.show()
on_ticks=tickets[tickets['locality']=='Ontario'] # All ontario data
on_ticks_sel = on_ticks.groupby('infraction_description')['infraction_description'].count().reset_index(name="count").nlargest(5,'count')
on_ticks_sel= on_ticks_sel.sort_values('infraction_description')
on_ticks_sel
other_ticks=tickets[tickets['locality']=='Other'] # All other data
other_ticks_sel = other_ticks.groupby('infraction_description')['infraction_description'].count().reset_index(name="count").nlargest(5,'count')
other_ticks_sel= other_ticks_sel.sort_values('infraction_description')
other_ticks_sel
import matplotlib.patches as mpatches
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k') #size of the plot
plt.bar(on_ticks_sel['infraction_description'],on_ticks_sel['count'],color = 'y')
plt.bar(other_ticks_sel['infraction_description'],other_ticks_sel['count'],color = 'b')
plt.xticks(other_ticks_sel['infraction_description'],rotation=90)
blue_patch=mpatches.Patch(color='b',label='Other') 
green_patch=mpatches.Patch(color='y',label='Ontario')
plt.legend(handles=[blue_patch,green_patch]) #providing the labels
plt.xlabel('Type of Infraction',fontsize=16)
plt.ylabel('Number of Infractions',fontsize=16)
plt.show()
on_ticks_month = on_ticks.groupby('month_num')['month_num'].count().reset_index(name="count")
on_ticks_month
other_ticks_month = other_ticks.groupby('month_num')['month_num'].count().reset_index(name="count")
other_ticks_month
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k') #size of the plot
plt.bar(on_ticks_month['month_num'],on_ticks_month['count'],color = 'y')
plt.bar(other_ticks_month['month_num'],other_ticks_month['count'],color = 'b')
labels = labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
plt.xticks(other_ticks_month['month_num'],labels,rotation=45)
blue_patch=mpatches.Patch(color='b',label='Other') 
green_patch=mpatches.Patch(color='y',label='Ontario')
plt.legend(handles=[blue_patch,green_patch]) #providing the labels
plt.xlabel('Month',fontsize=16)
plt.ylabel('Number of Infractions',fontsize=16)
plt.show()
print(tickets['location2'].describe())

print('\nThere are', "{:,}".format(tickets['location2'].describe()[1]), 'unique locations.\n')

tempticks=tickets[tickets['location2']!='NaN']
tempticks['location2'] = tempticks['location2'].str.replace('\d+', '')
print(tempticks['location2'].describe())
pd.crosstab(index=tempticks['location2'], columns='count').nlargest(10,'count')
tempticks['location2'].value_counts().nlargest(10).plot(kind='barh',figsize=(13,6), width = 0.75)
plt.title("Number of Infractions by Street", fontsize=16)
plt.ylabel("Street Name", fontsize=18)
plt.xlabel("No. of infractions", fontsize=18)
plt.show()