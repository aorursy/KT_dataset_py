# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from datetime import datetime

import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
# Any results you write to the current directory are saved as output.
sar=pd.read_csv('/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')
sar['Remain']=sar['Cumulative number of case(s)']-sar['Number recovered']-sar['Number of deaths']
top_n=40
top_sar_country=sar.groupby(['Country']).agg({'Cumulative number of case(s)':'max'}).reset_index().sort_values(by=['Cumulative number of case(s)'],ascending=False).head(top_n)
top_sar_country.head()
plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
plt.bar(x=top_sar_country['Country'],height=top_sar_country['Cumulative number of case(s)'])
plt.show()
top_sar_country[top_sar_country['Cumulative number of case(s)']>100]
plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
china_sar=sar[(sar['Country']=='China') & (sar['Cumulative number of case(s)']>0)]
x=china_sar['Date'].values
plt.plot(sar[(sar['Country']=='China') & (sar['Cumulative number of case(s)']>0)]['Date'],sar[(sar['Country']=='China') & (sar['Cumulative number of case(s)']>0)]['Remain'])
plt.axvline(x='2003-05-13',ymin= 0, label='pyplot vertical line',color='r')
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.annotate('60 days from peak infected at 3068 case till 38 infected case at 2003-07-11', (41,3100))
plt.xticks(x[::2])
plt.show()
plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
start_top=0
for c in top_sar_country['Country'].values[start_top:]:
    plt.plot(sar[(sar['Country']==c) & (sar['Cumulative number of case(s)']>0)]['Remain'].values)
plt.gca().legend(top_sar_country['Country'].values[start_top:start_top+10])
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.show()
sar['Date'].nunique()
plt.figure(figsize=(16, 12))
plt.xticks(rotation=90)
start_top=3
for c in top_sar_country['Country'].values[start_top:]:
    plt.plot(sar[(sar['Country']==c) & (sar['Cumulative number of case(s)']>0)]['Remain'].values)
plt.gca().legend(top_sar_country['Country'].values[start_top:start_top+20])
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.show()
x=['China','Other countries']
y=[top_sar_country[top_sar_country['Country']=='China']['Cumulative number of case(s)'].values[0],np.sum(top_sar_country[top_sar_country['Country']!='China']['Cumulative number of case(s)'])]

plt.bar(x,y)
for i, v in enumerate(y):
    plt.annotate(str(v), ( x[i],y[i]))
plt.show()
plt.pie(y,labels=['China','Other Countries'],autopct='%1.1f%%')
plt.show()
# Calcualte Total case of COVID by using historical SAR data and growth in China outbound tarveller  
81054*3316/5329*80/5
covid_pd=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_pd.tail()
covid_pd.rename(columns={'Country/Region':'Country'},inplace=True)
covid_pd=covid_pd.replace(to_replace='Mainland China',value='China')
covid_pd=covid_pd.replace(to_replace=' Azerbaijan',value='Azerbaijan')
covid_pd=covid_pd.replace(to_replace='UK',value='United Kingdom')
covid_pd=covid_pd.replace(to_replace='US',value='United States')
covid_pd['Remain']=covid_pd['Confirmed']-covid_pd['Recovered']-covid_pd['Deaths']
lat_long=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
lat_long.head()
china_province_covid=covid_pd[covid_pd['Country']=='China'].groupby('Province/State').agg({'Confirmed':'max'}).reset_index()
china_lat_long=pd.merge(china_province_covid,lat_long,on='Province/State')
cn_geo_data = "../input/china-regions-map/china.json"
#I use code from this link. His map is so cool.
#https://www.kaggle.com/pongsaksawang/coronavirus-propagation-visualization-forecast

map = folium.Map(location=[35, 105], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=4)

tooltip = 'Hubei'
lat=china_lat_long[china_lat_long['Province/State']=='Hubei']['Lat']
lon=china_lat_long[china_lat_long['Province/State']=='Hubei']['Long']
con=str(int(china_lat_long[china_lat_long['Province/State']=='Hubei']['Confirmed'].values[0]))
folium.Marker([lat, lon], popup=con+" cases", tooltip=tooltip).add_to(map)

folium.Choropleth(
    geo_data=cn_geo_data,
    name='choropleth',
    key_on='feature.properties.name',
    fill_color='blue',
    fill_opacity=0.18,
    line_opacity=0.7
).add_to(map)

for i in range(len(china_lat_long)):
      folium.CircleMarker(location = [china_lat_long.loc[i,'Lat'],china_lat_long.loc[i,'Long']], 
                        radius = np.log(china_lat_long.loc[i,'Confirmed'])*3, 
                    
                        color = '#E80018', 
                        fill_opacity = 0.7,
                        weight = 2, 
                        fill = True
                       ,fillColor = '#E80018'
                         ).add_to(map)

map
country_covid=covid_pd.groupby(['ObservationDate','Country']).agg({'Confirmed':'max','Deaths':'max','Recovered':'max','Remain':'sum'}).reset_index()
top_n=10
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
country_covid_remain=country_covid.pivot(index='ObservationDate',columns='Country',values='Remain').reset_index()
country_covid_remain=country_covid_remain.fillna(0)
plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
start_top=0
df=country_covid_remain[top_country['Country'].values[start_top]]
plt.bar(country_covid_remain['ObservationDate'],df)
for c in top_country['Country'].values[start_top+1:]:
    plt.bar(country_covid_remain['ObservationDate'],country_covid_remain[c],bottom=df)
    df=df+country_covid_remain[c]

plt.gca().legend(top_country['Country'].values[start_top:start_top+top_n])
plt.xlabel('Date')
plt.ylabel('Remain Infected cases')
#plt.yscale("log")
plt.show()
top_n=10
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
plt.figure(figsize=(16, 8))
plt.xticks(rotation=90)
top_c=np.append(top_country['Country'].values,"South Korea")
start_top=0
for c in top_c:
    plt.plot(country_covid[(country_covid['Country']==c) & (country_covid['Remain']>0)]['Remain'].values)
plt.plot(country_covid[(country_covid['Country']=='South Korea') & (country_covid['Remain']>0)]['Remain'].values)
plt.gca().legend(top_c)
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')

plt.show()
top_n=5
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
#To compare with China need log(data)

plt.figure(figsize=(16, 8))
plt.xticks(rotation=90)

#Add 30 days for china casue spread start since end of Dec-19
china_p=np.ones(66)
a=country_covid[(country_covid['Country']=='China') & (country_covid['Remain']>0)]['Remain'].values
china_p=np.append(china_p,a)
plt.plot(china_p)

top_c=np.append(top_country['Country'].values,["South Korea","Switzerland",'Iran','Thailand'])
start_top=0
for c in top_c:
    plt.plot(country_covid[(country_covid['Country']==c) & (country_covid['Remain']>0)]['Remain'].values)
plt.gca().legend(np.append("China",top_c))
plt.axvline(x=92,ymin= 0, label='pyplot vertical line',color='r')
plt.axvline(x=77,ymin= 0, label='pyplot vertical line',color='b')
plt.axvline(x=53,ymin= 0, label='pyplot vertical line',color='c')
plt.axvline(x=45,ymin= 0, label='pyplot vertical line',color='y')
plt.axvline(x=35,ymin= 0, label='pyplot vertical line',color='r')
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.yscale("log")
plt.annotate('China Peak at 92 Days', (92,2000))
plt.annotate('Thailand Peak at 77 Days', (77,1000))
plt.annotate('South Korea Peak at 53 Days', (53,300))
plt.annotate('Iran Peak at 46 Days', (45,100))
plt.annotate('Switzerland Peak at 35 Days', (35,20))

plt.show()

country='South Korea'
peak_remain=np.max(country_covid[(country_covid['Country']==country)]['Remain'])
peak_d=country_covid[(country_covid['Country']==country) & (country_covid['Remain']==peak_remain)]['ObservationDate'].values[0]
st_d=country_covid[(country_covid['Country']==country)]['ObservationDate'].values[0]
print(peak_d,datetime.strptime(peak_d,'%m/%d/%Y')-datetime.strptime(st_d,'%m/%d/%Y'))
top_n=100
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
current_date=max(covid_pd['ObservationDate'])
peak_ca=[]
peak_da=[]
peak_date=[]
for c in  top_country['Country'].values:
    peak_remain=np.max(country_covid[(country_covid['Country']==c)]['Remain'])
    peak_d=country_covid[(country_covid['Country']==c) & (country_covid['Remain']==peak_remain)]['ObservationDate'].values[0]
    st_d=country_covid[(country_covid['Country']==c)]['ObservationDate'].values[0]
    if ((datetime.strptime(current_date,'%m/%d/%Y')-datetime.strptime(peak_d,'%m/%d/%Y')).days)>=5:
        peak_ca.append(c)
        peak_da.append((datetime.strptime(peak_d,'%m/%d/%Y')-datetime.strptime(st_d,'%m/%d/%Y')).days)
        peak_date.append(peak_d)
        #print(c,peak_d,(datetime.strptime(peak_d,'%m/%d/%Y')-datetime.strptime(st_d,'%m/%d/%Y')).days)
        
peak_da=np.reshape(peak_da,(-1,1))
peak_ca=np.reshape(peak_ca,(-1,1))
peak_date=np.reshape(peak_date,(-1,1))
r=np.concatenate((peak_ca,peak_da,peak_date),axis=1)
country_peak=pd.DataFrame(data=r,columns=['Country','Peak Day','Peak Date'])
country_peak['Peak Day']=country_peak['Peak Day'].astype('int')
country_peak=country_peak[(country_peak['Country']!='China')]
country_peak=country_peak[(country_peak['Country']!='Others')]
country_peak=country_peak.sort_values(by=['Peak Day'])
country_peak
plt.figure(figsize=(16, 8))
plt.bar(country_peak['Country'],height=country_peak['Peak Day'])
plt.xticks(rotation=90)
plt.show()
country_covid['ObservationDate']=pd.to_datetime(country_covid['ObservationDate'])
select=['Italy','United States','Germany','France','Spain']
plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()
select=['South Korea','Iran']
plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()
select=['Switzerland','Netherlands','United Kingdom','Sweden','Belgium']
plt.xticks(rotation=90)

for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()
select=['Taiwan','Hong Kong','Singapore','Malaysia','Thailand','Japan']

plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Confirmed'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()
select=['Taiwan','Hong Kong','Thailand','Malaysia','Singapore','Japan']
plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()
import pandas as pd
temp = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
temp['dt']=pd.to_datetime(temp['dt'])
temp=temp.set_index('dt')
temp=temp.replace(to_replace='Bosnia And Herzegovina',value='Bosnia and Herzegovina')

covid_current=covid_pd.groupby('Country').agg({'Confirmed':'max','Deaths':'max','Recovered':'max'}).reset_index()

covid_current['Remain']=covid_current['Confirmed']-covid_current['Recovered']-covid_current['Deaths']
covid_current['%Recovered Rate']=covid_current['Recovered']/covid_current['Confirmed']*100
covid_current['%Deaths']=covid_current['Deaths']/covid_current['Confirmed']*100
covid_current.replace(np.inf,0,inplace=True)

top_n=100
top_country=covid_current.sort_values(by=['Confirmed'],ascending=False)['Country'].values[:top_n]
country_lat_long=lat_long.groupby('Country/Region').agg({'Lat':'mean','Long':'mean'}).reset_index()
covid_map=covid_current.merge(country_lat_long,left_on='Country',right_on='Country/Region',how='left')
#Change country name to match with wolrd json file
covid_map=covid_map.replace(to_replace='United States',value='United States of America')
covid_map=covid_map.sort_values(by=['%Recovered Rate'],ascending=False)
covid_map.head(10)
#world_geo="/kaggle/input/geo-json-world/world-countries.json"
world_geo="/kaggle/input/world-countries/world-countries.json"
covid_map.head(5)
map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)
#Remove Top eg (China , Small country)
folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=covid_map,
    columns=['Country','%Recovered Rate'],
    fill_color='PuBu'
).add_to(map)


map
#Screen outlier : few sampling of confime)
covid_death=covid_map.sort_values(by=['%Deaths'],ascending=False)
covid_death=covid_death[covid_death['Confirmed']>50]
covid_death[covid_death['Confirmed']>50].sort_values(by=['%Deaths'],ascending=False).head()
map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=covid_death,
    columns=['Country','%Deaths'],
    fill_color='OrRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.


map

#Find average temperature on  back 20 years
a=np.array([])
covid_current['temperature']=0
for c in top_country:
    avg=np.mean((temp[(temp['Country']==c) & (temp.index.month.isin([3]))].\
                 groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).\
                 reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])
    covid_current.loc[covid_current['Country']==c,['temperature']]=avg
    
    #if(math.isnan(avg)):
    #    print(c,avg)

ir_temp=np.mean((temp[(temp['Country']=='Ireland') & (temp.index.month.isin([3]))].\
             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).\
             reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])
covid_current.loc[covid_current['Country']=='Republic of Ireland',['temperature']]=ir_temp
covid_current.loc[covid_current['Country']=='North Ireland',['temperature']]=ir_temp

vatican_temp=np.mean((temp[(temp['Country']=='Italy') & (temp.index.month.isin([3]))].\
             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).\
             reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])

covid_current.loc[covid_current['Country']=='Vatican City',['temperature']]=vatican_temp

palestine_temp=np.mean((temp[(temp['Country']=='Israel') & (temp.index.month.isin([3]))].\
             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).\
             reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])

covid_current.loc[covid_current['Country']=='Palestine',['temperature']]=palestine_temp

denmark_temp=np.mean((temp[(temp['Country']=='Denmark (Europe)') & (temp.index.month.isin([3]))].\
             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).\
             reset_index().dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:]))
#covid_current.loc[covid_current['Country']=='Denmark',['temperature']]=vatican_temp
covid_current.loc[covid_current['Country']=='Denmark',['temperature']]=denmark_temp

#Change country name to match with wolrd json file
covid_current=covid_current.replace(to_replace='United States',value='United States of America')
all_temp=temp[(temp.index>'2010-01-01')&(temp.index.month.isin([3]))].groupby('Country').agg({'AverageTemperature':'mean'}).reset_index().sort_values(by=['AverageTemperature'])
all_temp=all_temp.replace(to_replace='United States',value='United States of America')
all_temp_risk=all_temp[(all_temp['AverageTemperature']>-0) & (all_temp['AverageTemperature']<15)]
#Change country name to match with wolrd json file
all_temp=all_temp.replace(to_replace='United States',value='United States of America')
pd=covid_current.sort_values(by=['Confirmed'],ascending=False)
case=pd['Confirmed'].values
t=pd['temperature'].values
country=pd['Country'].values
plt.figure(figsize=(12, 8))

plt.scatter(t,case)
plt.xlabel('Celsius')
plt.ylabel('confirmed')
plt.yscale('log')
plt.axhline(y=100000,xmin= 0,color='g')
plt.axhline(y=1000,xmin= 0,color='g')

plt.axvline(x=0,ymin= 0,color='r')
plt.axvline(x=15,ymin= 0,color='r')
for i, txt in enumerate(pd['Country'][:15]):
    plt.annotate(txt, ( t[i],case[i]))
plt.show()
max_con=100000
min_con=10000
pd=covid_current[(covid_current['Confirmed']<100000) & (covid_current['Confirmed']>10000) & (covid_current['Country']!='China')]
case_low=pd['Confirmed'].values
temp_low=pd['temperature'].values
country_low=pd['Country'].values
total=pd.count()[0]
in_range=pd[(pd['temperature']>=-5) & (pd['temperature']<=10)].count()[0]
plt.figure(figsize=(10, 8))

plt.xlabel('Celsius')
plt.ylabel('confirmed')
plt.yscale('log')

plt.scatter(temp_low,np.log(case_low) )
for i, txt in enumerate(country_low):
    plt.annotate(txt, ( temp_low[i],np.log(case_low[i])))
plt.show()
print(in_range,"in",total,"countries of ",str(min_con),"-",str(max_con)," cases temperature range = -5 to 10 Celsius")
pd=covid_current[(covid_current['Confirmed']<10000) & (covid_current['temperature']<15) & (covid_current['temperature']>0) & (covid_current['Country']!='China')]
case=pd['Confirmed'].values
t=pd['temperature'].values
country=pd['Country'].values
plt.figure(figsize=(10, 8))
plt.xlabel('Celsius')
plt.ylabel('Confirmed)')
plt.scatter(t,case)
for i, txt in enumerate(country):
    plt.annotate(txt, ( t[i],case[i]))
plt.show()
map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=covid_current,
    columns=['Country','temperature'],
    fill_color='PuRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.


map
plt.figure(figsize=(8, 12))
plt.barh(all_temp_risk['Country'],all_temp_risk['AverageTemperature'])
plt.show()
check_n=all_temp_risk.merge(covid_current,on='Country',how='left')
check_n[(check_n['Confirmed'].isnull()) & (check_n['AverageTemperature']>0) & (check_n['AverageTemperature']<10)]['Country']
map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=all_temp,
    columns=['Country','AverageTemperature'],
    fill_color='PuRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.


map
country_covid_confirm=country_covid.pivot(index='ObservationDate',columns='Country',values='Confirmed').reset_index()
country_covid_confirm=country_covid_confirm.fillna(0)
top_n=10
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
new_case=country_covid_confirm.iloc[:,1:].diff()
new_case.replace(np.nan,0,inplace=True)
start_top=1
for c in (top_country['Country'].values[start_top:]):
    plt.plot(new_case[c])
plt.gca().legend(top_country['Country'].values[start_top:top_n])
plt.xlabel('Days')
plt.ylabel('New Cases')
plt.show()
import pandas as pd
top_n=50
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
a=np.array([])
for c in (top_country['Country'].values):
    day_data=covid_pd[covid_pd['Country']==c].groupby('ObservationDate').agg({'Confirmed':'max'}).sort_values(by=['ObservationDate']).reset_index()
    day_accum=day_data['Confirmed'].values
    min_date=day_data['ObservationDate'][0]
    max_change=0
    if day_accum[0]>0:
        d=day_data['ObservationDate'][0]
    else:
        d=''
    for i in range(1,len(day_accum)):
        if day_accum[i-1]>0 and (day_accum[i]-day_accum[i-1])/day_accum[i-1] > max_change:
            max_change=(day_accum[i]-day_accum[i-1])/day_accum[i-1]
            if d=='':
                d=day_data['ObservationDate'][i]
    a=np.append(a,[max_change,d])
a=a.reshape((-1,2))
country=top_country['Country'].values.reshape((-1,1))
s=np.concatenate([country,a],axis=1)
spread=pd.DataFrame(s,columns=['Country','max_speed','Start_date'])
spread['max_speed']=pd.to_numeric(spread['max_speed'])
spread=spread.sort_values(by=['max_speed'],ascending=False)
map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=spread[2:],
    columns=['Country','max_speed'],
    fill_color='PuBu'
).add_to(map)


map
covid_current=pd.merge(covid_current, spread, on='Country',how='left')
pop_dense=pd.read_csv('/kaggle/input/migration-data-worldbank-1960-2018/migration_population.csv')
pop_data=pop_dense[pop_dense['year']==2018].groupby(['country','year']).agg({'population':'max','pop_density':'max','region':'max',
                                           'incomeLevel':'max','lendingType':'max',
                                            'longitude':'mean','latitude':'mean'}).reset_index()
pop_data.head()
lat_long.head()
covid_current=pd.merge(covid_current, lat_long, left_on='Country',right_on='Country/Region')
covid_current=covid_current.iloc[:,:14]
covid_current=pd.merge(covid_current, pop_data, left_on='Country',right_on='country',how='left')
covid_map=covid_map.dropna()
map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=all_temp,
    columns=['Country','AverageTemperature'],
    fill_color='PuRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.

for i in range(len(covid_map)):
      folium.CircleMarker(location = [covid_map['Lat'].iloc[i],covid_map['Long'].iloc[i]], 
                        radius = np.log(covid_map['Confirmed'].iloc[i])*2, 
                        color = '#E80018', 
                        fill_opacity = 0.7,
                        weight = 2, 
                        fill = True
                       ,fillColor = '#E80018'
                         ).add_to(map)

map
plt.figure(figsize=(16, 12))
plt.scatter(np.log(covid_current['Confirmed']),np.log(covid_current['pop_density']))
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( np.log(covid_current['Confirmed'][i]),np.log(covid_current['pop_density'][i])))
plt.xlabel('Confirmed')
plt.ylabel('pop_density')
plt.show()
covid_current.corr(method ='pearson').iloc[:,6:]
covid_current.corr(method ='spearman').iloc[:,6:]
covid_current.corr(method ='kendall').iloc[:,6:]
plt.figure(figsize=(16, 12))
plt.scatter(np.log(covid_current['population']),covid_current['%Recovered Rate'])
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( np.log(covid_current['population'][i]),covid_current['%Recovered Rate'][i]))
covid_current.head()
covid_current[covid_current['Country']!='China'].groupby('region').agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})
rate=covid_current[covid_current['Country']!='China'].groupby('region').agg({'%Recovered Rate':'mean'}).reset_index().sort_values(by=['%Recovered Rate'])
plt.figure(figsize=(12, 8))
plt.barh(rate['region'],rate['%Recovered Rate'])
plt.xlabel('%Recovered')
plt.show()
rate=covid_current[covid_current['Country']!='China'].groupby('region').agg({'%Deaths':'mean'}).reset_index().sort_values(by=['%Deaths'])
plt.figure(figsize=(12, 8))
plt.barh(rate['region'],rate['%Deaths'])
plt.xlabel('%Deaths')
plt.show()
covid_current.groupby('incomeLevel').agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})
rate_income=covid_current.groupby('incomeLevel').agg({'%Recovered Rate':'mean','%Deaths':'mean'}).reset_index().sort_values(by=['%Recovered Rate'])
plt.figure(figsize=(12, 8))
plt.barh(rate_income['incomeLevel'],rate_income['%Recovered Rate'])

plt.xlabel('%Recovered')

plt.show()
rate_income=covid_current.groupby('incomeLevel').agg({'%Recovered Rate':'mean','%Deaths':'mean'}).reset_index().sort_values(by=['%Deaths'])
plt.figure(figsize=(12, 8))
plt.barh(rate_income['incomeLevel'],rate_income['%Deaths'])
plt.xlabel('%Deaths')
plt.show()
urban_pd=pd.read_csv('/kaggle/input/urbanize-percentage-by-country-2020/urban percentage by country.csv')
urban_pd.rename(columns={'COUNTRY':'Country'},inplace=True)
urban_pd.head()
covid_current=pd.merge(covid_current, urban_pd, on='Country',how='left')
covid_current.head()
covid_current.corr(method ='pearson').iloc[:,6:]
covid_current.corr(method ='kendall').iloc[:,6:]
covid_current.corr(method ='spearman').iloc[:,6:]
plt.figure(figsize=(16, 12))
plt.scatter(np.log(covid_current['Confirmed']),covid_current['%Urban'])
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( np.log(covid_current['Confirmed'][i]),covid_current['%Urban'][i]))
plt.xlabel('log(Confimred)')
plt.ylabel('%Urban')
plt.show()
plt.figure(figsize=(16, 12))
plt.scatter(covid_current['%Recovered Rate'],covid_current['%Urban'])
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( covid_current['%Recovered Rate'][i],covid_current['%Urban'][i]))
plt.xlabel('%Recovered Rate')
plt.ylabel('%Urban')
plt.show()
