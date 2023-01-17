import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
terror = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', low_memory=False,

                    usecols=[0, 1, 2, 8, 10, 11, 12, 26, 27, 29, 35, 41, 81, 82, 98])

terror = terror.rename(columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'country_txt':'country', 'region_txt':'region', 'provstate':'province_or_state', 'attacktype1_txt':'attack_type', 'natlty1_txt':'nationality', 'targtype1_txt':'target', 'weaptype1_txt':'weapon_type', 'nkill':'killed'})

terror['killed'] = terror['killed'].fillna(0).astype(int)

terror = terror[terror['year'] > 1995]



non_west_region, west_region = (pd.DataFrame() for dataframe in range(2))

west_region_names = ['North America', 'Western Europe', 'Australasia & Oceania']

for region in west_region_names:

    west_region =  pd.concat([terror[terror['region'] == region], west_region])



non_west_region_names = ['Middle East & North Africa', 'South Asia', 'Sub-Saharan Africa', 'Southeast Asia', 'Eastern Europe', 'South America', 'Central Asia', 'East Asia', 'Central America & Caribbean']

for region in non_west_region_names:

    non_west_region =  pd.concat([terror[terror['region'] == region], non_west_region])
print('The country that experienced the most terrorist attacks from 1996 - 2016 is {}.'.format(terror['country'].value_counts().index[0]))

print('The country that experienced the second most terrorist attacks from 1996 - 2016 is {}.'.format(terror['country'].value_counts().index[1]))
x = terror['year'].unique()

y = terror.groupby('year').count()['id']



plt.plot(x, y)

plt.xlabel('Year')

plt.ylabel('Number of Attacks')

plt.title('Number of Terrorist Activities Per Year')

plt.xticks(x, x, rotation='90') # x, label, rotation

plt.show()



print('There are an average of {:.0f} terrorist attacks per year with a standard deviation of {:.2f}.'.format(y.mean(), y.std()))
count = terror['region'].value_counts()

region = terror['region'].value_counts().keys()



plt.bar(region, count)

plt.ylabel('Attacks')

plt.xlabel('Regions')

plt.title('Terrorist Attacks by Region from 1996-2016')

plt.xticks(rotation=75)

plt.show()

print("The region that experienced the most terrorist attacks from 1996 - 2016 is {}.".format(count.index[0]))

print("The region that experienced the second most terrorist attacks from 1996 - 2016 is {}.".format(count.index[1]))
plt.figure(figsize=(15, 5))

x = terror['weapon_type'].value_counts().keys()

y = terror['weapon_type'].value_counts()

plt.subplot(1, 2, 1)

plt.bar(x, y)

plt.xlabel('Weapon Types')

plt.ylabel('Number of Attacks')

plt.xticks(rotation = 90)

plt.title('Weapon Types Used by Terrorists')



x = terror['attack_type'].value_counts().keys()

y = terror['attack_type'].value_counts()

plt.subplot(1, 2, 2)

plt.bar(x, y)

plt.xlabel('Attack Type')

plt.ylabel('Number of Attacks')

plt.xticks(rotation=90)

plt.title('Attack Types Used by Terrorists')

plt.show()
west_x = west_region['weapon_type'].value_counts().keys()

west_y = west_region['weapon_type'].value_counts()

non_west_x = non_west_region['weapon_type'].value_counts().keys()

non_west_y = non_west_region['weapon_type'].value_counts()



plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

plt.bar(west_x, west_y)

plt.xlabel('Weapon Types')

plt.ylabel('Number of Attacks')

plt.xticks(rotation = 90)

plt.title('Weapon Types Used by Terrorists in the Western Regions')



plt.subplot(1, 2, 2)

plt.bar(non_west_x, non_west_y)

plt.xlabel('Weapon Types')

plt.ylabel('Number of Attacks')

plt.xticks(rotation = 90)

plt.title('Weapon Types Used by Terrorists in non Western Regions')

plt.show()
labels = ['Successful Attacks', 'Failed Attacks']



west_europe_total_attacks = terror.groupby('region').count()['id'].loc['Western Europe']

west_europe_success = terror.groupby('region').sum()['success'].loc['Western Europe']

west_europe_fail = west_europe_total_attacks - west_europe_success

west_attack_probability = [west_europe_success, west_europe_fail]



east_europe_total_attacks = terror.groupby('region').count()['id'].loc['Eastern Europe']

east_europe_success = terror.groupby('region').sum()['success'].loc['Eastern Europe']

east_europe_fail = east_europe_total_attacks - east_europe_success

east_attack_probability = [east_europe_success, east_europe_fail]

sa_total_attacks = terror.groupby('region').count()['id'].loc['South America']

sa_success = terror.groupby('region').sum()['success'].loc['South America']

sa_fail = sa_total_attacks - sa_success

sa_attack_probability = [sa_success, sa_fail]



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

plt.pie(west_attack_probability, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('Percentage of Terrorists Attacks in Western Europe')



plt.subplot(1,3,2)

plt.pie(east_attack_probability, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('Percentage of Terrorists Attacks in Eastern Europe')



plt.subplot(1,3,3)

plt.pie(sa_attack_probability, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('Percentage of Terrorists Attacks in South America')

plt.tight_layout()

plt.show()
west_result, non_west_result = (pd.DataFrame() for dataframe in range(2))



west_x = west_region.groupby('weapon_type').sum()['killed']

non_west_x = non_west_region.groupby('weapon_type').sum()['killed']



for weapon in terror['weapon_type'].unique():

    if weapon not in west_x:

        west_result = west_x.append(pd.Series([0], index=[weapon]))

    if weapon not in non_west_x:

        non_west_result = non_west_x.append(pd.Series([0], index=[weapon]))

if non_west_result.empty:

    non_west_result = non_west_x

if west_result.empty:

    west_result = west_x



west_result = west_result.sort_index()

non_west_result = non_west_result.sort_index()



weapons = terror.groupby('weapon_type').count()['id'].sort_index().index



plt.figure(figsize=(25, 8))

plt.subplot(1, 2, 1)

plt.barh(weapons, non_west_result, color='b', align='center', alpha=.25, label='Non-Western Regions')

plt.barh(weapons, west_result, color='g', align='center', alpha=.5, label='Western Regions')

plt.xlabel('Fatalities')

plt.ylabel('Weapon Type')

plt.title('Number of Fatalities by Weapon Type from 1996-2016')

plt.legend()

plt.xticks(rotation=45)

plt.show()
x = terror['year'].unique()

west_y = west_region.groupby('year').killed.sum()

non_west_y = non_west_region.groupby('year').killed.sum()



plt.plot(x, west_y, label='the West Region')

plt.plot(x, non_west_y, label='Non West Regions')

plt.xlabel('Year')

plt.ylabel('Number of Fatalities')

plt.title('Fatalities Per Year')

plt.xticks(x, x, rotation='90') # x, xlabel, rotation

plt.legend()

plt.show()



west_death = sum(west_region.groupby('year').sum()['killed'])

west_attack = sum(west_region.groupby('year').count()['id'])

nonwest_death = sum(non_west_region.groupby('year').sum()['killed'])

nonwest_attack = sum(non_west_region.groupby('year').count()['id'])



print('There were {} fatalities and {} attacks in the Western regions, roughly {} deaths per attack.'.format(west_death, west_attack, round(west_death/west_attack, 1)))

print('There were {} fatalities and {} attacks in the non-Western regions, roughly {} deaths per attack.'.format(nonwest_death, nonwest_attack, round(nonwest_death/nonwest_attack, 1)))