# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# we are using the inline backend
%matplotlib inline 

import matplotlib as mpl
import matplotlib.pyplot as plt

import folium

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
'''
# Retriving Dataset
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

# Depricated
# df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
df_confirmed.head()
df_confirmed.shape
df_deaths.head()
df_deaths.shape
df_covid19.head()
df_covid19.shape
df_covid19.set_index('Country_Region',inplace= True)
df_covid19.head()
df_covid19.loc[['US', 'India'], :]
world_map = folium.Map(location=[51.000, .1278], zoom_start=2)

deaths = folium.map.FeatureGroup()

# loop through the 100 crimes and add each to the map
for lat, lng, label, country, confirmed in zip(df_covid19.Lat, df_covid19.Long_, df_covid19.Deaths, df_covid19.index.values, df_covid19.Confirmed):
    folium.CircleMarker(
        [lat, lng],
        radius=5, # define how big you want the circle markers to be
        color='yellow',
        fill=True,
        popup=country+' '+str(label)+'/'+str(confirmed),
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(world_map)

world_map
df_tot = pd.DataFrame(df_covid19[['Confirmed','Deaths', 'Recovered', 'Active']].sum(axis=0))
df_tot.transpose()
df_tot.transpose().style.background_gradient(cmap='Wistia',axis=1)
df_covid19[['Confirmed','Deaths', 'Recovered', 'Active']].sort_values(by='Confirmed', ascending= False).style.background_gradient(cmap='Wistia',axis=0)
df_confirmed.head()
df_confirmed[df_confirmed['Country/Region'].str.contains('US')]
tot_cols = df_confirmed.shape[1]
print('Tot cols=', tot_cols)

dt_cols = tot_cols - 4
print('Date cols=', dt_cols)

#df_confirmed.iloc[:, 69]
#The date column indexes are 4 to tot_cols-1

start_dt = df_confirmed.columns.values[4]
end_dt = df_confirmed.columns.values[tot_cols-1]
print('Start Dt=', start_dt, ' End Dt=', end_dt)

#Rename the date columns to integer sequence(days from start dt) for easy access
cols=df_confirmed.columns.values
#print(cols[0:5])
new_cols = list(cols[0:4]) + list(map(str, range(dt_cols)))
#print(new_cols)
#print(len(new_cols))

df_confirmed.columns = new_cols
print(df_confirmed.shape)
import os
#os.remove("/kaggle/working/top6-10.png")
df_confirmed1 = df_confirmed.groupby('Country/Region').sum(axis=0)
#Lat and Long too got summed up..so not to use from here

#Sort by the last column i.e. latest total
top5=df_confirmed1.sort_values(new_cols[-1]).tail()

top5_t = top5[list(list(map(str, range(dt_cols))))].transpose()

top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 confirmed cases of Top 5 countries')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Confirmed Cases')


# Reference lines 
#threshold = 10000
x1 = [40,57]
y1 = [0, 131071]
plt.plot(x1,y1,"--",linewidth =2,color = "gray")
plt.annotate("Doubles/Day",xy=(52,130000),fontsize=10,alpha = 0.5, rotation=60.01)

x2 = [40,73]
y2 = [0, 131072]
plt.annotate("Doubles/2nd Day",xy=(60,115000),fontsize=10,alpha = 0.5, rotation=46.00)
plt.plot(x2,y2,"--",linewidth =2,color = "gray")


plt.grid(True)
plt.savefig('confirm_top5.jpg')

plt.show()

x = [0,17]
#y = 2**(x+np.log2(threshold))
list(map(lambda x1 : 2**x1, x))

#Top from 6 to 10 countries
#Sort by the last column i.e. latest total
top5=df_confirmed1.sort_values(new_cols[-1]).tail(10).head()
#top5

top5_t = top5[list(list(map(str, range(dt_cols))))].transpose()
top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 confirmed cases of from #6 to #10 countries')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Confirmed Cases')
plt.grid(True)
plt.savefig('confirm_top6_10.png')

plt.show()

#select countries of interest
sel_countries = df_confirmed1.loc[['India','Pakistan', 'Bangladesh', 'Sri Lanka', 'Burma', 'Nepal']]

top5_t = sel_countries[list(map(str, range(dt_cols)))].transpose()
top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 confirmed cases - India and neighbors')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Confirmed Cases')
plt.grid(True)
plt.savefig('confirm_india.png')

plt.show()

#df_confirmed.iloc[:, 69]
#The date column indexes are 4 to tot_cols-1
'''
As the df_death and df_confirmed have same structure will use the columns from df_confirmed

start_dt = df_confirmed.columns.values[4]
end_dt = df_confirmed.columns.values[tot_cols-1]
print('Start Dt=', start_dt, ' End Dt=', end_dt)

#Rename the date columns to integer sequence(days from start dt) for easy access
cols=df_confirmed.columns.values
#print(cols[0:5])
new_cols = list(cols[0:4]) + list(map(str, range(dt_cols)))
#print(new_cols)
#print(len(new_cols))
'''
df_deaths.columns = new_cols
print(df_deaths.shape)
df_deaths.head()
df_death1 = df_deaths.groupby('Country/Region').sum(axis=0)
#Lat and Long too got summed up..so not to use from here

#Sort by the last column i.e. latest total
top5=df_death1.sort_values(new_cols[-1]).tail()

top5_t = top5[list(list(map(str, range(dt_cols))))].transpose()

top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 deaths in top 5 countries')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Deaths')
plt.grid(True)
plt.savefig('death_count_top5.jpg')

plt.show()

top5.head()
#Sort by the last column i.e. latest total
top5=df_death1.sort_values(new_cols[-1]).tail(10).head()

top5_t = top5[list(list(map(str, range(dt_cols))))].transpose()

top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 deaths in top #6-#10 countries')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Deaths')
plt.grid(True)
plt.savefig('death_count_top6_10.jpg')

plt.show()

#Sort by the last column i.e. latest total
top5=df_death1.loc[['India','Pakistan', 'Bangladesh', 'Sri Lanka', 'Burma', 'Nepal']]

top5_t = top5[list(list(map(str, range(dt_cols))))].transpose()

top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 deaths in India and neighboring countries')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Deaths')
plt.grid(True)
plt.savefig('death_count_india.jpg')

plt.show()

#Join df_confirmed1 and df_death1
df_merge=pd.merge(left=df_confirmed1, right=df_death1, on = 'Country/Region')
#df_merge.head()

#df_top5_ratio = df_merge.sort_values(new_cols[-1]+'_x').tail(10)
df_top5_ratio = df_merge.sort_values(new_cols[-1]+'_x')
df_top5_ratio.head()

#for col in map(str, list(range(dt_cols))):
#df_ratio = pd.DataFrame(columns=new_cols)

for col in map(str, list(range(0, dt_cols))):
    #print(df_top5_ratio[col+'_y'], " / ", df_top5_ratio[col+'_x'])
    #print(df_top5_ratio[col+'_y']/df_top5_ratio[col+'_x'])
    df_top5_ratio[col] = df_top5_ratio[col+'_y']/df_top5_ratio[col+'_x']

df_top5_ratio.fillna(0, inplace=True)

df_top5_ratio[df_top5_ratio>0.25] = 0.000

df_top5_ratio1 = df_top5_ratio.loc[:, list(map(str, range(dt_cols)))].round(3)

df_top5_ratio1.loc[:, list(map(str, range(dt_cols)))]


#Death ratio of top 5 countries

top5_t = df_top5_ratio1.tail()
top5_t = top5_t.transpose()

top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 death ratio wrt confirmed cases in top 5 countries')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Death Ratio')
plt.grid(True)
plt.savefig('death_ratio_top5.jpg')

plt.show()

#Death ratio of top #6 to #10 countries

top5_t = df_top5_ratio1.tail(10).head()
top5_t = top5_t.transpose()

top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 death ratio wrt confirmed cases in top #6 to #10 countries')
plt.xlabel('Days (from 1/22/20 to ' + end_dt+')')
plt.ylabel('Death Ratio')
plt.grid(True)
plt.savefig('death_ratio_top6_10.jpg')

plt.show()

#Death ratio of India and neighboring countries

top5_t = df_top5_ratio1.loc[['India','Pakistan', 'Bangladesh', 'Sri Lanka', 'Burma', 'Nepal']]
top5_t = top5_t.transpose()

top5_t.plot(kind='line', figsize=(10, 6))
#plt.ylim(0, 100000)
#y_ticks = list(range(0, 100000, 5000))
#plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
#plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 death ratio wrt confirmed cases in India and neighboring countries')
plt.xlabel('Days (from '+start_dt+' to ' + end_dt+')')
plt.ylabel('Death Ratio')
plt.grid(True)
plt.savefig('death_ratio_India.jpg')

plt.show()

# Retriving Dataset
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_confirmed.head()
del df_confirmed2
tot_cols = df_confirmed.shape[1]
print('Tot cols=', tot_cols)

dt_cols = tot_cols - 4
print('Date cols=', dt_cols)

start_dt = df_confirmed.columns.values[4]
end_dt = df_confirmed.columns.values[tot_cols-1]
print('Start Dt=', start_dt, ' End Dt=', end_dt)

#Rename the date columns to integer sequence(days from start dt) for easy access
cols=df_confirmed.columns.values
#print(cols[0:5])
new_cols = list(cols[0:4]) + list(map(str, range(dt_cols)))
print(new_cols)
#print(len(new_cols))

#df_confirmed.columns = new_cols
#print(df_confirmed.shape)

import datetime
numdays=30
base = datetime.datetime.today()
date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]
date_list
from datetime import datetime
datelist = pd.date_range(start="2020-01-22",end="2020-02-22").tolist()
datelist
dts_delete=[date.strftime("%-m/%-d/%y") for date in datelist]
df_confirmed2 = df_confirmed.drop(dts_delete, 1)
df_confirmed2.head()
tot_cols = df_confirmed2.shape[1]
print('Tot cols=', tot_cols)

dt_cols = tot_cols - 4
print('Date cols=', dt_cols)

start_dt = df_confirmed2.columns.values[4]
end_dt = df_confirmed2.columns.values[tot_cols-1]
print('Start Dt=', start_dt, ' End Dt=', end_dt)

#Rename the date columns to integer sequence(days from start dt) for easy access
cols=df_confirmed2.columns.values
#print(cols[0:5])
new_cols = list(cols[0:4]) + list(map(str, range(dt_cols)))
#print(new_cols)
#print(len(new_cols))

df_confirmed2.columns = new_cols
print(df_confirmed.shape)
df_confirmed2.head()
df_confirmed2['0dcount'] = df_confirmed2['0']  # First cols has base value
for col in map(str, list(range(1, dt_cols))):
    #print(df_top5_ratio[col+'_y'], " / ", df_top5_ratio[col+'_x'])
    #print(df_top5_ratio[col+'_y']/df_top5_ratio[col+'_x'])
    #df_top5_ratio[col] = df_top5_ratio[col+'_y']/df_top5_ratio[col+'_x']
    #print(df_confirmed2[col].head())
    #df_confirmed2[col+'dcount'] = df_confirmed2[]
    df_confirmed2[col+'dcount'] = df_confirmed2[col]-df_confirmed2[str(int(col)-1)]
df_confirmed2.head()
#delete original cumulative total cols
df_confirmed3 = df_confirmed2.drop(list(map(str, range(dt_cols))), 1)
df_confirmed3.head()
df_confirmed3 = df_confirmed3.groupby('Country/Region').sum(axis=0)
df_confirmed3.head()
list(map(str, range(dt_cols)))
#Death ratio of India and neighboring countries

my_countries = df_confirmed3.loc[['India','US', 'Italy', 'Spain']]
#my_countries = df_confirmed3.loc[['Italy','India']]
my_countries = my_countries.drop(['Lat', 'Long'], 1)
my_countries.columns = list(map(str, range(dt_cols)))
my_countries = my_countries.transpose()

sel={'US':0, 'Italy':1, 'India':2,  'Spain':3}
for k in sel:
    f = plt.figure(sel[k])
    my_countries[k].plot(kind='bar', figsize=(10, 6))
    plt.title('COVID-19 daily confirmed counts of select countries - '+k)
    plt.xlabel('Days (from '+start_dt+' to ' + end_dt+')')
    plt.ylabel('Confirmed Counts')
    plt.grid(True)
    plt.savefig('daily_bar_'+k+'.png')
    f.show()
#plt.savefig('death_ratio_India.jpg')

#plt.show()

df_confirmed[df_confirmed.iloc[:, 0].notnull()]
df_confirmed.set_index('Country/Region', inplace=True)
df_confirmed.head()
df_sel=df_confirmed.loc[['China', 'Italy', 'Spain', 'India'], :].groupby(['Country/Region']).sum(axis=0)
df_sel.head(50)
#df_confirmed.drop('Province/State', axis=1, inplace=True)
#df_confirmed.head()
df_sel.drop(['Lat', 'Long'], axis=1, inplace=True)
df_sel.head()
df_sel = df_sel.transpose()
df_sel.head()
print ('Matplotlib version: ', mpl.__version__) 
mpl.style.use(['ggplot'])
df_sel.dtypes
type(df_sel)
dates = list(map(str, pd.date_range(start="2020-01-21",end="2020-03-28")))
dates1 = list(map(lambda x:str(int(x[5:7]))+'/'+str(int(x[8:10]))+'/'+'20', dates))
dates1[0:5]
df_sel.plot(kind='line', figsize=(14, 8))
#plt.ylim(0, 100000)
y_ticks = list(range(0, 100000, 5000))
plt.yticks(y_ticks)
#plt.xticks(dates1) 
#plt.xticks(y_ticks) 
plt.xticks(list(np.arange(0,len(d),int(len(d)/5))),d[:-1:int(len(d)/5)]+[d[-1]])
plt.title('COVID-19 of selected countries')
plt.xlabel('Dates')
plt.ylabel('Confirmed Cases')
plt.show()
from datetime import datetime, timedelta,date
d = [datetime.strptime(date,'%m/%d/%y').strftime("%d %b") for date in df_sel.index]
d[0:6]
