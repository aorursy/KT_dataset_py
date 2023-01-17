import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
terror = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', low_memory=False,

                    usecols=[0, 1, 2, 8, 10, 11, 12, 26, 27, 29, 35, 41, 82, 98])

terror = terror.rename(columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'country_txt':'country', 'region_txt':'region', 'provstate':'province_or_state', 'attacktype1_txt':'attack_type', 'natlty1_txt':'nationality', 'targtype1_txt':'target', 'weaptype1_txt':'weapon_type', 'nkill':'killed'})

terror['killed'] = terror['killed'].fillna(0).astype(int)

terror = terror[terror['year'] > 1995]



terror.head(3)
x = terror['month']

bins = 12

months = range(1,13)



plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

plt.grid(linestyle='dashed')

plt.hist(x, bins=bins, normed=True, color='brown')

plt.xlabel('Month of Attack')

plt.xticks(months, months)

plt.ylabel('Probability')

plt.title('Probability of Attack in a Given Month')

plt.show()
x = terror.groupby('year').count()['success']

years = terror['year'].unique()



plt.plot(x)

plt.grid(linestyle='dashed')

plt.xlabel('Year')

plt.ylabel('Successful Terrorist Attacks')

plt.title('Successful Terrorist Attacks from 1996-2016')

plt.xticks(years, years, rotation=90)

plt.show()
ax = terror['region'].value_counts().plot(kind='barh', color='red')

ax.set_xlabel('Regions')

ax.set_ylabel('Number of Terrorist Attacks')

ax.set_title('Terrorist Attacks in Regions')
middle_east_region = terror[terror['region'] == 'Middle East & North Africa']

middle_east_y = middle_east_region['country'].value_counts()

middle_east_x = middle_east_y.keys()



plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

plt.barh(middle_east_x, middle_east_y)

plt.title('Terrorist Attacks in Middle East and North Africa')

plt.xlabel('Number of Attacks')

plt.ylabel('Countries in Middle East & North Africa Region')



south_asia_region = terror[terror['region'] == 'South Asia']

south_asia_y = south_asia_region['country'].value_counts()

south_asia_x = south_asia_y.keys()

plt.subplot(1,2,2)

plt.barh(south_asia_x, south_asia_y, color='green')

plt.title('Terrorist Attacks in South Asia')

plt.xlabel('Number of Attacks')

plt.ylabel('Countries in South Asia')

plt.tight_layout()

plt.show()

print('The average number of terrorist attacks per country in the Middle East is {0:.0f} while in South Asia the average number of terrorist attacks are {1:.0f}.'.format(middle_east_y.mean(), south_asia_y.mean()))
iraq_y = terror[terror['country'] == 'Iraq']['city'].value_counts()

iraq_x = range(iraq_y.count())

plt.scatter(iraq_y, iraq_x)

plt.xlabel('Number of Attacks on Iraqian Cities')

plt.ylabel('Cities')

plt.title('Attacks on Iraqian Cities')

plt.show()

print('In Iraq, {0} has experienced {1} terrorist attacks.'.format(iraq_y.index[0], iraq_y.values[0]))
no_iraq = terror[terror['country'] != 'Iraq']

y = no_iraq['region'].value_counts()

x = y.keys()



plt.barh(x, y, color='green')

plt.xlabel('Regions')

plt.ylabel('Number of Terrorist Attacks')

plt.title('Terrorist Attacks in Regions')

plt.show()
plt.figure(figsize=(15, 5))

y = terror['weapon_type'].value_counts()

x = y.keys()

plt.subplot(1, 2, 1)

plt.bar(x, y, color='pink')

plt.xlabel('Weapon Types')

plt.ylabel('Number of Attacks')

plt.xticks(rotation = 90)

plt.title('Weapon Types Used by Terrorists')



y = terror['attack_type'].value_counts()

x = y.keys()

plt.subplot(1, 2, 2)

plt.bar(x, y, color='teal')

plt.xlabel('Attack Type')

plt.ylabel('Number of Attacks')

plt.xticks(rotation=90)

plt.title('Attack Types Used by Terrorists')

plt.show()
y = terror[terror['weapon_type'] == 'Firearms'].groupby('attack_type').count()['id']

x = y.index

explode=(0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

percent = 100.*y/y.sum()

patches, texts = plt.pie(y, explode=explode, startangle=90, radius=1.1)

labels = ['{0} - {1:1.2f}%'.format(i,j) for i,j in zip(x, percent)]



plt.legend(patches, labels, bbox_to_anchor=(-0.2, 1), fontsize=10)

plt.axis('equal')

plt.title('Distribution of Attack Types for Firearms')

plt.show()
y = terror['target'].value_counts()

x = y.keys()



plt.figure(figsize=(10,5))

plt.barh(x, y)

plt.ylabel('Target Type')

plt.xlabel('Number of Attacks')

plt.title('Terrorist Attacks Against Targets')

plt.show()
y = terror[terror['target'] == 'Private Citizens & Property']['country'].value_counts()



plt.hist(y, bins=20, color='orange')

plt.xlabel('Number of Attacks on Private Citizens & Property')

plt.ylabel('Number of Countries')

plt.title('Distribution of Terrorist Attacks on Private Citizens & Property')

plt.show()

print('Most of the terrorist attacks on Private Citizens & Property occur in {}.'.format(y.index[0]))