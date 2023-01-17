# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset_2012_2017 = pd.read_csv("../input/Chicago_Crimes_2012_to_2017.csv")
# dataset_2012_2017.head()
dataset_2005_2007 = pd.read_csv("../input/Chicago_Crimes_2005_to_2007.csv",skiprows=533719,header=None,names=['unnamed','ID','Case Number','Date','Block','IUCR','Primary Type','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community Area','FBI Code','X Coordinate','Y Coordinate','Year','Updated On','Latitude','Longitude','Location'])
# dataset_2005_2007.head()
"""
Plot the amount of arrests in 2012_2017
There are more arrests in 2012 until 2017 
how come?
"""
arrest_counts_2012_2017 =dataset_2012_2017['Arrest'].count()
arrest_counts_2005_2007 = dataset_2005_2007['Arrest'].count()
print(arrest_counts_2012_2017)
print(arrest_counts_2005_2007)
primary_type = dataset_2005_2007['Primary Type']
primary_type_ = dataset_2012_2017['Primary Type']

assault_2005_2007 = 0
narcotics_2005_2007 = 0
other_offense_2005_2007 = 0
battery_2005_2007 = 0
theft_2005_2007 = 0
criminal_trepass_2005_2007 = 0
motor_vehicle_theft_2005_2007 = 0
robbery_2005_2007 = 0
deceptive_practice_2005_2007 = 0
criminal_damage_2005_2007 = 0
prostitution_2005_2007 = 0

assault_2012_2017 = 0
narcotics_2012_2017 = 0
other_offense_2012_2017 = 0
battery_2012_2017 = 0
theft_2012_2017 = 0
criminal_trepass_2012_2017 = 0
motor_vehicle_theft_2012_2017 = 0
robbery_2012_2017 = 0
deceptive_practice_2012_2017 = 0
criminal_damage_2012_2017 = 0
prostitution_2012_2017 = 0

for i in primary_type:
    if 'ASSAULT'in i:
        assault_2005_2007 += 1
    if 'NARCOTICS' in i:
        narcotics_2005_2007 +=1
    if 'OTHER OFFENSE' in i:
        other_offense_2005_2007 += 1
    if 'BATTERY' in i:
        battery_2005_2007 += 1
    if 'THEFT' in i:
        theft_2005_2007 += 1
    if 'CRIMINAL TRESPASS' in i:
        criminal_trepass_2005_2007 += 1
    if 'MOTOR VEHICLE THEFT' in i:
        motor_vehicle_theft_2005_2007 += 1
    if 'ROBBERY' in i:
        robbery_2005_2007 += 1
    if 'DECEPTIVE PRACTICE' in i:
        deceptive_practice_2005_2007 += 1
    if 'CRIMINAL DAMAGE' in i:
        criminal_damage_2005_2007 += 1
    if 'PROSTITUTION' in i:
        prostitution_2005_2007 += 1
        
        
for j in primary_type_:
    if 'ASSAULT'in j:
        assault_2012_2017 += 1
    if 'NARCOTICS' in j:
        narcotics_2012_2017 +=1
    if 'OTHER OFFENSE' in j:
        other_offense_2012_2017 += 1
    if 'BATTERY' in j:
        battery_2012_2017 += 1
    if 'THEFT' in j:
        theft_2012_2017 += 1
    if 'CRIMINAL TRESPASS' in j:
        criminal_trepass_2012_2017 += 1
    if 'MOTOR VEHICLE THEFT' in j:
        motor_vehicle_theft_2012_2017 += 1
    if 'ROBBERY' in j:
        robbery_2012_2017 += 1
    if 'DECEPTIVE PRACTICE' in j:
        deceptive_practice_2012_2017 += 1
    if 'CRIMINAL DAMAGE' in j:
        criminal_damage_2012_2017 += 1
    if 'PROSTITUTION' in j:
        prostitution_2012_2017 += 1
    

# print(descrption.iteritems())
"""
 Here I want to make a list of all the primary times of crime in 2005_2007
"""
print('assault: ' , assault_2005_2007)
print('narcotics: ', narcotics_2005_2007)
print('prostitution: ',prostitution_2005_2007)
print('robbery: ',robbery_2005_2007)
print('theft: ',theft_2005_2007)
print('other offense: ',other_offense_2005_2007)
print('criminal damage: ', criminal_damage_2005_2007)
print('criminal trepass: ', criminal_trepass_2005_2007)
print('deceptive trepass: ', deceptive_practice_2005_2007)
print('battery: ', battery_2005_2007)
print('\n')
print('assault: ' , assault_2012_2017)
print('narcotics: ', narcotics_2012_2017)
print('prostitution: ',prostitution_2012_2017)
print('robbery: ',robbery_2012_2017)
print('theft: ',theft_2012_2017)
print('other offense: ',other_offense_2012_2017)
print('criminal damage: ', criminal_damage_2012_2017)
print('criminal trepass: ', criminal_trepass_2012_2017)
print('deceptive trepass: ', deceptive_practice_2012_2017)
print('battery: ', battery_2012_2017)

""" 
    Plot the information of the different crimes in a bar plot.
"""
import matplotlib.pyplot as plt; plt.rcdefaults()

crime_type = ('Assault', 'Narcotics', 'Other offense', 'Battery', 'Theft', 'Criminal trepass'
           ,'Motor vechile theft','Robbery','Deceptive practice','Criminal damage', 'Prostitution')

fig, ax = plt.subplots()
y_pos = np.arange(len(crime_type))
bar_width = 0.35
opacity = 0.8

crime_count_2005_2007 = [assault_2005_2007,narcotics_2005_2007,other_offense_2005_2007
               ,battery_2005_2007,theft_2005_2007,criminal_trepass_2005_2007,
                motor_vehicle_theft_2005_2007,robbery_2005_2007,deceptive_practice_2005_2007,
               criminal_damage_2005_2007, prostitution_2005_2007
              ]

crime_count_2012_2017 = [assault_2012_2017,narcotics_2012_2017,other_offense_2012_2017
               ,battery_2012_2017,theft_2012_2017,criminal_trepass_2012_2017,
                motor_vehicle_theft_2012_2017,robbery_2012_2017,deceptive_practice_2012_2017,
               criminal_damage_2012_2017, prostitution_2012_2017
              ]

plt.bar(y_pos, crime_count_2005_2007,  bar_width, alpha=opacity, color='indigo',label='2005_2007')
crimes_2012_2017 = plt.bar(y_pos + bar_width, crime_count_2012_2017, bar_width,
                 alpha=opacity,
                 color='lime',
                 label='2012_2017')


plt.xticks(y_pos + bar_width, crime_type, rotation='vertical')
plt.ylabel('Crime count')
plt.xlabel('Crime years')
plt.title('Crime Chicago 2005/2017')
plt.legend()

plt.tight_layout()
plt.show()
dataset_2005_2007 = dataset_2005_2007.dropna()
all_longs = dataset_2005_2007['Longitude']
all_lats = dataset_2005_2007['Latitude']
all_lats = all_lats.tolist()
all_longs = all_longs.tolist()
print(all_longs[0])
print(len(all_lats))
print(type(all_lats))
import folium
# Make an empty map
m = folium.Map(location=[all_lats[0],all_longs[0]], zoom_start=13)

for i in range(1330891):
    folium.Marker([all_lats[i],all_longs[i]]).add_to(m)
# folium.Marker([all_lats[0],all_longs[0]]).add_to(m)
# folium.Marker([all_lats[1],all_longs[1]]).add_to(m)
# folium.Marker([all_lats[2],all_longs[2]]).add_to(m)

 
# Save it as html
m.save('crimes_map_2005_2007.html')


import folium
theft_map = folium.Map(location = [all_lats[0], all_longs[0]], zoom_start = 9)

for j in range(0,len(dataset_2005_2007['Primary Type'])):
    if dataset_2005_2007.iloc[j]['Primary Type'] == 'THEFT':
        folium.Marker([dataset_2005_2007.iloc[j]['Latitude'],dataset_2005_2007.iloc[j]['Longitude']]).add_to(theft_map)
        print('Yes sir this is theft')
    else:
        print('No sir no theft')

theft_map.save('theft_map_2005_2007.html')
""" 
    Now it is time to get 20% of the total thefts in 2005/2007 and put this on a map with the 
    street name where it happened. So you might be asking why I get 20%? Well this is way faster and
    when I want to use deep learning for this I would love to have my data separated and train that small amount.
"""

import folium

small_map = folium.Map(location=[20, 0],tiles='Stamen Toner', zoom_start=2)

for t in range(63993):
    if dataset_2005_2007.iloc[t]['Primary Type'] == 'THEFT':
        folium.Marker([dataset_2005_2007.iloc[t]['Latitude'], dataset_2005_2007.iloc[t]['Longitude']], popup=dataset_2005_2007.iloc[t]['Block']).add_to(small_map)
        print('Theft')
    else:
        print('No')
        
small_map.save('small_map_2005_2007.html')

import keras
all_blocks = []
all_theft_lats = []
all_theft_longs = []
all_theft_dates = []

for u in range(0,len(dataset_2005_2007['Primary Type'])):
    if dataset_2005_2007.iloc[u]['Primary Type'] == 'THEFT':
        all_blocks.append(dataset_2005_2007.iloc[u]['Block'])
        all_theft_lats.append(dataset_2005_2007.iloc[u]['Latitude'])
        all_theft_longs.append(dataset_2005_2007.iloc[u]['Longitude'])
        all_theft_dates.append(dataset_2005_2007.iloc[u]['Date'])

theft_list = pd.DataFrame({
    'Latitude': all_theft_lats,
    'Longitude': all_theft_longs,
    'Block': all_blocks,
    'Date' : all_theft_dates
})
theft_list['Date'] = pd.to_datetime(theft_list['Date'])
theft_list
import datetime as dt

months = theft_list['Date'].dt.month
months
theft_list = pd.DataFrame({
    'Latitude': all_theft_lats,
    'Longitude': all_theft_longs,
    'Block': all_blocks,
    'Date' : all_theft_dates,
    'Months': months
})
theft_list


#for g in  range(0,len(theft_list['Date'])):
    #if(theft_list[g]['Date'] == 2006):
       # print("Yes it is")
#theft_list[['Block']].plot(figsize=(20,10), linewidth=5, fontsize=20)
#plt.xlabel('Year', fontsize=20);
X = theft_list.iloc[:,0:3]
y = theft_list.iloc[:,3]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
theft_list.dtypes

from sklearn.ensemble import RandomForestClassifier
