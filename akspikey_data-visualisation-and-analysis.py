import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time

import folium



import os

print(os.listdir("../input"))
permit_data = pd.read_csv("../input/film-permits.csv")
permit_data.shape
permit_data.head()
unique_event_type = permit_data.EventType.unique()

print('The unique event types are ', str(unique_event_type))
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(20, 10)



sns.countplot(x="EventType", data=permit_data, ax=ax)

sns.despine()
pd.value_counts(permit_data.EventType)
sns.set_style('ticks')

fig, ax = plt.subplots(ncols=4)

fig.set_size_inches(20, 10)



sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'Shooting Permit'], ax=ax[0])

ax[0].set(xlabel='Shooting Permit')

sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'Rigging Permit'], ax=ax[1])

ax[1].set(xlabel='Rigging Permit')

sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'Theater Load in and Load Outs'], ax=ax[2])

ax[2].set(xlabel='Theater Load in and Load Outs')

sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'DCAS Prep/Shoot/Wrap Permit'], ax=ax[3])

ax[3].set(xlabel='DCAS Prep/Shoot/Wrap Permit')

sns.despine()
permit_data['EventTimeinHours'] = permit_data.apply(lambda x: abs(time.mktime(time.strptime(x['StartDateTime'], '%Y-%m-%dT%H:%M:%S.%f'))-time.mktime(time.strptime(x['EndDateTime'], '%Y-%m-%dT%H:%M:%S.%f')))/(60*60), axis=1)

permit_data.head()[['StartDateTime', 'EndDateTime', 'EventTimeinHours']]
permit_data[permit_data['EventTimeinHours'] == max(permit_data['EventTimeinHours'])]['EventTimeinHours']/24
permit_data[permit_data['EventTimeinHours'] == min(permit_data['EventTimeinHours'])]['EventTimeinHours']
permit_data.Category.unique()
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(20, 10)



sns.countplot(x="Category", data=permit_data, ax=ax, order = permit_data.Category.value_counts().index)

sns.despine()
print('The number of sub-category are ', len(permit_data.SubCategoryName.unique()))



sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(20, 10)



sns.countplot(y="SubCategoryName", data=permit_data, ax=ax, order = permit_data.SubCategoryName.value_counts().index)

sns.despine()
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(20, 10)



sns.countplot(y="SubCategoryName", data=permit_data[permit_data.Category == 'Television'], ax=ax)

sns.despine()
permit_data.Country.unique()
group_event_country = permit_data.groupby(['EventType', 'Country']).size()

group_by_country = pd.DataFrame(group_event_country).reset_index()

group_by_country
geo_corrs = {

    'Netherlands': (52.2379891, 5.53460738161551),

    'United States of America' : (39.7837304, -100.4458825),

    'Australia': (-24.7761086, 134.755),

    'Canada': (61.0666922, -107.9917071),

    'France': (46.603354, 1.8883335),

    'Germany': (51.0834196, 10.4234469),

    'Japan': (36.5748441, 139.2394179),

    'Panama': (8.3096067, -81.3066246),

    'United Kingdom': (54.7023545, -3.2765753)

}
m = folium.Map(location=[20,0], tiles="Mapbox Bright", zoom_start=2)



for i in range(0,len(group_by_country)):

    corr = geo_corrs[group_by_country['Country'][i]]

    folium.Circle(

      location=[corr[0], corr[1]],

      popup=group_by_country['Country'][i],

      radius=int(group_by_country[0][i])*10

    ).add_to(m)



m