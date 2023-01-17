import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import matplotlib

matplotlib.rcParams['xtick.major.size'] = 0

matplotlib.rcParams['ytick.major.size'] = 0





df = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1', low_memory = False)

# The column listing number of perpetrators has many attacks listed as -99.0, and some as -9.0.

# I can't find any discription of the meaning of these values, so I'll replace them with nan to make analysis possible.

nperps = df['nperps'].replace(-99.0, np.nan).replace(-9.0, np.nan)

df['nperps'] = nperps



print('A little section of the dataset:')

display(df.head())



print('Total number of attacks documented:', df.shape[0])
year_count = df.groupby('iyear')['eventid'].count()



plt.subplots(figsize = (16, 10))

plt.bar(year_count.index, year_count.values, color = 'black')

plt.title('Number of Attacks Over Time', fontsize = 20, y = 1.02)

plt.show()
print('Worst year:', year_count.idxmax())

print('Number of incidents in that year:', year_count.max())



print('\nNumber of incidents in 2017 (most recent year in the data):', year_count[2017])
# defining a new colormap, so matplotlib doesn't re-use colors

n = len(df['region_txt'].unique())

color = plt.cm.rainbow(np.linspace(0,1,n))

c = zip(range(n), color)

colors = [color for color in c]

count = 0



fig, ax = plt.subplots(figsize = (26, 15))

for region in df['region_txt'].unique():

    subset = df[df['region_txt'] == region]

    year_count = subset.groupby('iyear')['eventid'].count()

    

    sns.lineplot(year_count.index, year_count.values, label = region, color = colors[count][1], linewidth = 5)

    if year_count.values[-1] > 110:

        plt.text(year_count.index[-1] + 0.5, year_count.values[-1], s = region, color = 'white', fontsize = 16, fontweight = 'bold')

    count += 1



plt.title('Number of Attacks by Region over Time', fontsize = 30, y = 1.03, color = 'white', fontweight = 'bold')

ax.set_facecolor('black')

fig.set_facecolor('black')

plt.yticks(fontsize = 16, color = 'white', fontweight = 'bold')

plt.xticks(fontsize = 16, color = 'white', fontweight = 'bold')

plt.legend(loc = 'best', prop={'size': 20});
subset = df[df['region_txt'] == 'Western Europe']

subset = subset[subset['iyear'] < 2000]

year_count = subset.groupby('iyear')['eventid'].count()

print('Mean number of attacks per year in Western Europe before 2000: {:,.2f}'.format(year_count.mean()))
region_list = ['North America', 'Western Europe', 'Eastern Europe', 'East Asia', 'Sub-Saharan Africa', 'Southeast Asia']



fig, axes = plt.subplots(3, 2, figsize = (18, 12))

plt.suptitle('Number of Attacks Over Time in a Subset of Regions', fontsize = 22)



subset = df[df['region_txt'] == 'North America']

year_count = subset.groupby('iyear')['eventid'].count()

axes[0, 0].plot(year_count, color = 'black')

axes[0, 0].set_title('North America', fontsize = 16, y = 1.02)



subset = df[df['region_txt'] == 'South America']

year_count = subset.groupby('iyear')['eventid'].count()

axes[0, 1].plot(year_count, color = 'black')

axes[0, 1].set_title('South America', fontsize = 16, y = 1.02)



subset = df[df['region_txt'] == 'Western Europe']

year_count = subset.groupby('iyear')['eventid'].count()

axes[1, 0].plot(year_count, color = 'black')

axes[1, 0].set_title('Western Europe', fontsize = 16, y = 1.02)



subset = df[df['region_txt'] == 'Eastern Europe']

year_count = subset.groupby('iyear')['eventid'].count()

axes[1, 1].plot(year_count, color = 'black')

axes[1, 1].set_title('Eastern Europe', fontsize = 16, y = 1.02)



subset = df[df['region_txt'] == 'Southeast Asia']

year_count = subset.groupby('iyear')['eventid'].count()

axes[2, 0].plot(year_count, color = 'black')

axes[2, 0].set_title('Southeast Asia', fontsize = 16, y = 1.02)



subset = df[df['region_txt'] == 'East Asia']

year_count = subset.groupby('iyear')['eventid'].count()

axes[2, 1].plot(year_count, color = 'black')

axes[2, 1].set_title('East Asia', fontsize = 16, y = 1.02)



plt.show()
region_count = df.groupby('region_txt')['eventid'].count().sort_values(ascending = False)



plt.subplots(figsize = (20, 12))

sns.barplot(x = region_count.index, y = region_count.values, palette = 'viridis')

plt.xticks(rotation = 80, fontsize = 15)

plt.xlabel('')

sns.despine(left = True)

plt.title('Number of Attacks by Region, since 1970', fontsize = 22, y = 1.02);



# it would be nice to normalize these values by population size
print('Number of countries in the dataset:', len(df.country_txt.unique()))



country_count = df.groupby('country_txt')['eventid'].count().sort_values(ascending = False)



fig, ax = plt.subplots(figsize = (20, 12))

sns.barplot(country_count.index, country_count.values, palette = 'inferno', alpha = 0.9)

plt.title('Number of Attacks by Country, since 1970', fontsize = 26, y = 0.94)

plt.xticks(rotation = 90)

ax.set_xticks([])

ax.margins(x = 0.005)

plt.xlabel('')

fig.set_facecolor('snow')

ax.set_facecolor('snow')

sns.despine(left = True);
# there are 205 countries, so we want the 102nd one in our ordered list



country_count = df.groupby('country_txt')['eventid'].count().sort_values(ascending = False)

median = country_count[102]

name = country_count[country_count == 66].index[0]



print('The median country is {}, with {} attacks'.format(name, median))
country_count = df.groupby('country_txt')['eventid'].count().sort_values(ascending = False)

print('Least attacked countries:\n', country_count.tail(12))
country_count = df.groupby('country_txt')['eventid'].count().sort_values(ascending = False)

fig, ax = plt.subplots(figsize = (12, 20))

plt.barh(country_count[:50].index, country_count[:50].values, color = 'black')

plt.title('Number of Attacks In Top 50 Countries, since 1970', fontsize = 22, y = 1.02)

plt.xticks(fontsize = 16)

plt.yticks(fontsize = 14)

ax.tick_params(axis = 'x', length = 5, width = 2)

plt.show()



print('{:,.1f}% of all attacks took place in Iraq'.format((country_count[0] / country_count.sum()) * 100))

print('{:,.1f}% of all attacks took place in Pakistan'.format((country_count[1] / country_count.sum()) * 100))

print('{:,.1f}% of all attacks took place in Afghanistan'.format((country_count[2] / country_count.sum()) * 100))

print('{:,.1f}% of all attacks took place in India\n'.format((country_count[3] / country_count.sum()) * 100))



combined_percent = ((country_count[0] + country_count[1] + country_count[2] + country_count[3]) / country_count.sum()) * 100

print('Together, they account for {:,.1f}% of the attacks'.format(combined_percent))
city_count = df.groupby('city')['eventid'].count().sort_values(ascending = False)



fig, ax = plt.subplots(figsize = (12, 20))

plt.barh(city_count[1:21].index, city_count[1:21].values, color = 'black')

plt.xticks(fontsize = 16)

ax.tick_params(axis = 'x', length = 5, width = 2)

plt.title('Number of Attacks in Top 20 Cities, since 1970', fontsize = 22, y = 1.02)

ax.margins(x = 0.02);
country_count = df.groupby('country_txt')['eventid'].count().sort_values(ascending = False)



data = dict(type = 'choropleth', 

            locations = country_count.index,

            locationmode = 'country names',

            z = country_count.values, 

            text = country_count.index)

layout = dict(title = 'Number of Attacks Since 1970', 

              height = 1200,

              geo = dict(showframe = False, 

                         projection = {'type': 'mercator'}))



fig = go.Figure(data = [data], layout=layout)

iplot(fig)
print('Total number killed: {}'.format(int(df['nkill'].sum())))

print('Total number wounded: {}\n'.format(int(df['nwound'].sum())))

print('Mean number killed per attack: {:,.2f}'.format(df['nkill'].mean()))

print('Standard deviation in number killed per attack: {:,.2f}\n'.format(df['nkill'].std()))

print('Mean number wounded per attack: {:,.2f}'.format(df['nwound'].mean()))

print('Standard deviation in number wounded per attack: {:,.2f}'.format(df['nwound'].std()))
killed = df.groupby('nkill').count()['eventid']

greater_than_20 = killed[21:]

killed = killed[:20]

int_index = [int(num) for num in killed.index]

killed = killed.reindex(int_index)

killed = killed.append(pd.Series(greater_than_20.sum(), index = ['>20']))



plt.subplots(figsize = (16, 10))

sns.barplot(killed.index, killed.values, color = 'black', alpha = 0.9)

plt.xlabel('Number Killed', fontsize = 14)

plt.ylabel('Number of Attacks', fontsize = 14)

plt.title('Number of Attacks By Number Killed', fontsize = 26, y = 1.02)

sns.despine(left = True)

plt.show()



none_killed = df[df['nkill'] == 0]

print('Number of attacks in which no one was killed:', len(none_killed))

print('Percent of all incidents: {:,.1f}%\n'.format((len(none_killed) / len(df['nkill']) * 100)))



one_killed = df[df['nkill'] == 1]

print('Number of attacks in which one person was killed:', len(one_killed))

print('Percent of all incidents: {:,.1f}%'.format((len(one_killed) / len(df['nkill']) * 100)))



over_20_killed = df[df['nkill'] > 20]

print()

print('Number of attacks in which more than 20 people were killed:', len(over_20_killed))

print('Percent of total incidents: {:,.1f}%'.format((len(over_20_killed)/len(df['nkill'])) * 100))



over_100_killed = df[df['nkill'] > 100]

print()

print('Number of attacks in which more than 100 people were killed:', len(over_100_killed))

print('Percent of total incidents: {:,.1f}%'.format(len(over_100_killed)/len(df['nkill']) * 100))



over_1000_killed = df[df['nkill'] > 1000]

print()

print('There were 4 attacks in which more than 1000 people were killed')

print('Their records are displayed below:')

display(over_1000_killed)
relative_deadliness = df.groupby(by = 'region_txt').mean()['nkill'].sort_values(ascending = False)



plt.subplots(figsize = (12, 8))

sns.barplot(relative_deadliness.index, relative_deadliness.values, palette = 'viridis')

plt.xticks(rotation = 70)

plt.xlabel('')

sns.despine(left = True)

plt.title('Average Number Killed Per Attack By Region', fontsize = 18, y = 1.02);
# defining a new colormap, so matplotlib doesn't re-use colors

n = len(df['region_txt'].unique())

color = plt.cm.gist_ncar(np.linspace(0,1,n))

c = zip(range(n), color)

colors = [color for color in c]

count = 0



fig, ax = plt.subplots(figsize = (18, 12))

for region in df['region_txt'].unique():

    region_data = df[df['region_txt'] == region].groupby('iyear').mean()['nkill']

    

    plt.plot(region_data.index, region_data.values, label = region, color = colors[count][1], linewidth = 3)

    count += 1

    

plt.legend()

plt.title('Average Number Killed per Attack', fontsize = 22, y = 1.02)

ax.set_facecolor('black')

sns.despine(left = True);
ee = df[df['region_txt'] == 'Eastern Europe']

ee = ee[ee['iyear'] == 2004]

print('Worst Eastern European attacks in 2004:')

ee.sort_values('nkill', ascending = False).head(6)
killed2009 = df[df['iyear'] == 2009].sort_values(by = 'nkill', ascending = False)

print('Worst attacks in 2009:')

killed2009.head(6)
ea = df[df['region_txt'] == 'East Asia']

ea = ea[ea['iyear'] > 2008]

print('Worst East Asian attacks since 2008:')

ea.sort_values('nkill', ascending = False).head(10)
from decimal import Decimal

country_killed = df.groupby('country_txt')['nkill'].sum()

country_count = df.groupby('country_txt')['eventid'].count()

r2, p = stats.pearsonr(country_count, country_killed)



print('Correlation between number of attacks and number killed by country: {:,.3f}'.format(r2))

print('p-value of that correlation: {:,.1E}'.format(Decimal(p)))
# defining a new colormap, so matplotlib doesn't re-use colors

n = len(df['region_txt'].unique())

color = plt.cm.nipy_spectral(np.linspace(0,1,n))

c = zip(range(n), color)

colors = [color for color in c]

count = 0



fig, ax = plt.subplots(figsize = (18, 12))

for region in df['region_txt'].unique():

    region_data = df[df['region_txt'] == region].groupby('iyear').sum()['nkill']

    

    plt.plot(region_data.index, region_data.values, label = region, color = colors[count][1], linewidth = 3)

    count += 1

    

plt.legend()

plt.title('Numbers Killed by Region', fontsize = 22, y = 1.02)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

ax.set_facecolor('black')

sns.despine(left = True);
mean_killed_over_time = df.groupby('iyear').mean()['nkill']

fig, ax = plt.subplots(figsize = (12, 8))

plt.plot(mean_killed_over_time.index, mean_killed_over_time.values, color = 'darkred', linewidth = 5)

plt.title('Mean Number Killed per Attack', fontsize = 22, y = 1.02)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

sns.despine(left = True);
wounded = df.groupby('nwound').count()['eventid'].drop(8.5)

greater_than_20 = wounded[21:]

wounded = wounded[:20]

wounded = wounded.append(pd.Series(greater_than_20.sum(), index = ['>20']))



fig, ax = plt.subplots(figsize = (16, 10))

sns.barplot(wounded.index, wounded.values, color = 'black', alpha = 0.9)

plt.xlabel('Number Wounded', fontsize = 14)

plt.ylabel('Number of Attacks', fontsize = 14)

ax.set_xticklabels(list(range(21)) + ['>20'])

plt.title('Number of Attacks By Number Wounded', fontsize = 26, y = 1.02)

sns.despine(left = True)

plt.show()



none_wounded = df[df['nwound'] == 0]

print('Number of incidents in which no one was wounded:', len(none_wounded))

print('Percent of total incidents: {:,.1f}%\n'.format(len(none_wounded)/len(df['nwound']) * 100))



one_wounded = df[df['nwound'] == 1]

print('Number of incidents in which one person was wounded:', len(one_wounded))

print('Percent of total incidents: {:,.1f}%\n'.format(len(one_wounded)/len(df['nwound']) * 100))



over_30_wounded = df[df['nwound'] > 30]

print('Number of incidents in which more than 100 people were wounded:', len(over_30_wounded))

print('Percent of total incidents: {:,.1f}%\n'.format(len(over_30_wounded)/len(df['nwound']) * 100))



over_1000_wounded = df[df['nwound'] > 1000]

print('There were {} attacks in which more than 1000 people were wounded'.format(len(over_1000_wounded)))

print('Their records are displayed below:')

display(over_1000_wounded)
no_casualties = none_wounded[none_wounded['nkill'] == 0]

print('Number of attacks in which no one was killed or wounded:', len(no_casualties))

print('Percent of total attacks: {:,.1f}%'.format((len(no_casualties) / len(df['nkill'])) * 100))
relative_deadliness = df.groupby(by = 'region_txt').mean()['nwound'].sort_values(ascending = False)



plt.subplots(figsize = (12, 8))

sns.barplot(relative_deadliness.index, relative_deadliness.values, palette = 'viridis')

plt.xticks(rotation = 70)

plt.xlabel('')

sns.despine(left = True)

plt.title('Average Number Wounded Per Attack By Region', fontsize = 18, y = 1.02);
sarin_removed = df.drop(58841)



regions_wounded = sarin_removed.groupby(by = 'region_txt').sum()['nwound']

regions_attacked = sarin_removed.groupby(by = 'region_txt').count()['eventid']

relative_deadliness = (regions_wounded / regions_attacked).sort_values(ascending = False)



print("East Asia's high average is hugely affected by the 1995 sarin gas attack on Tokyo")

print('If it is removed from the calculation, East Asia has an average of {:,.2f} wounded per attack\n'.format(relative_deadliness['East Asia']))





wtc_removed = df.drop([73126, 73127])



regions_wounded = wtc_removed.groupby(by = 'region_txt').sum()['nwound']

regions_attacked = wtc_removed.groupby(by = 'region_txt').count()['eventid']

relative_deadliness = (regions_wounded / regions_attacked).sort_values(ascending = False)



print("North America's high average is also hugely affected by the World Trade Center attacks")

print('If they are removed from the calculation, its average plunges to {:,.2f} wounded per attack'.format(relative_deadliness['North America']))
wtc_removed = df.drop([73126, 73127])



relative_deadliness = (wtc_removed.groupby(by = 'region_txt').mean()['nkill']).sort_values(ascending = False)



relative_deadliness.index.name = ''

relative_deadliness
fig, ax = plt.subplots(figsize = (16, 12))

sns.scatterplot(x = 'nkill', y = 'nwound', data = df, hue = 'region_txt', palette = 'gist_rainbow')

sns.despine(left = True)

handles, labels = ax.get_legend_handles_labels()

plt.title('Scatterplot of Numbers Killed and Wounded', fontsize = 23, y = 1.02)

plt.xlabel('Number Killed', fontsize = 16)

plt.ylabel('Number Wounded', fontsize = 16)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

plt.legend(bbox_to_anchor=(1.05, 0.32), loc=2, borderaxespad=0., handles=handles[1:], labels=labels[1:]);
attack_type = df.groupby('attacktype1_txt')['eventid'].count().sort_values(ascending = False)

# the values for the other attack type variables (i.e. attacktype2 and attacktype3) are mostly null



fig, ax = plt.subplots(figsize = (12, 8))

plt.bar(attack_type.index, attack_type.values, color = 'black')

plt.title('Number of Incidents by Attack Type, since 1970', fontsize = 18, y = 1.02)

plt.xticks(rotation = 60)

sns.despine(left = True)

plt.show()
n = len(df['attacktype1_txt'].unique())

color = plt.cm.prism(np.linspace(0,1,n))

c = zip(range(n), color)

colors = [color for color in c]

count = 0



fig, ax = plt.subplots(figsize = (20, 10))

for a_type in df['attacktype1_txt'].unique():

    subset = df[df['attacktype1_txt'] == a_type]

    year_count = subset.groupby('iyear')['eventid'].count()

    

    plt.plot(year_count.index, year_count.values, label = a_type , linewidth = 5, color = colors[count][1])

    count += 1

    

plt.legend(loc = 'best', prop={'size': 16})

plt.title('Number of Attacks by Attack Type Over Time', fontsize = 23, y = 1.03)

ax.set_facecolor('black')
average_deadliness_type = df.groupby('attacktype1_txt').mean()['nkill'].sort_values(ascending = False)



fig, ax = plt.subplots(figsize = (12, 8))

sns.barplot(average_deadliness_type.index, average_deadliness_type.values, palette = 'magma')

plt.xticks(rotation = 70)

plt.xlabel('')

plt.ylabel('Mean Number Killed Per Attack', fontsize = 14)

plt.title('Average Deadliness of Attack Types', fontsize = 20, y = 1.02)

sns.despine(left = True);
wtc_removed = df.drop([73126, 73127])



average_deadliness_type = wtc_removed.groupby('attacktype1_txt').mean()['nkill'].sort_values(ascending = False)



print('If the World Trade Center attacks are removed, the average for hijacking drops to {:,.2f}'.format(average_deadliness_type['Hijacking']))
na = df['nperps'].isna().sum()

print('Percent of attacks which have no information for number of perpetrators: {:,.1f}%'.format((na / len(df)) * 100))
mean_perpetrators = df.groupby('attacktype1_txt').mean()['nperps']

del mean_perpetrators.index.name

print('Mean number of perpetrators involved by attack type:')

display(mean_perpetrators.sort_values())



median_perpetrators = df.groupby('attacktype1_txt').median()['nperps']

del median_perpetrators.index.name

print('Median number of perpetrators involved by attack type:')

display(median_perpetrators.sort_values())
fig, ax = plt.subplots(figsize = (16, 12))

sns.scatterplot(x = 'nperps', y = 'nkill', data = df, hue = 'attacktype1_txt', palette = 'nipy_spectral')

plt.title('Attacks, by Number of Perpetrators and Number Killed', fontsize = 20, y = 1.02)

plt.ylabel('Number Killed', fontsize = 16)

plt.xlabel('Number of Perpetrators', fontsize = 16)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

handles, labels = ax.get_legend_handles_labels()

plt.legend(bbox_to_anchor=(1.05, 0.32), loc=2, borderaxespad=0., handles=handles[1:], labels=labels[1:])

sns.despine(left = True);
perps_kill = df[['nperps', 'nkill']]

correlation = perps_kill.corr()['nkill']['nperps']

print('Correlation between number of perpetrators and number killed: {:,.3f}'.format(correlation))
none_killed = pd.DataFrame(columns = ['method', 'percent none killed'])



for method in df['attacktype1_txt'].unique():

    method_data = df[df['attacktype1_txt'] == method]

    percent_none_killed = ((method_data['nkill'] == 0).sum() / len(method_data)) * 100

    

    none_killed = none_killed.append(pd.DataFrame({'method': method, 'percent none killed': percent_none_killed}, index = [0]))

    

none_killed = none_killed.sort_values(by = 'percent none killed', ascending = False)



fig, ax = plt.subplots(figsize = (16, 8))

sns.barplot(none_killed['method'], none_killed['percent none killed'], palette = 'viridis')

plt.title('Probability of No Fatalities by Attack Type', fontsize = 22, y = 1.02)

plt.ylabel('Percent of Attacks in Which No One Died', fontsize = 14)

plt.xlabel('')

plt.yticks(fontsize = 14)

plt.xticks(fontsize = 14)

sns.despine(left = True)

plt.xticks(rotation = 70);
assassination = df[df['attacktype1_txt'] == 'Assassination']

percent_one_killed = ((assassination['nkill'] == 1).sum() / len(assassination)) * 100



print('Percent of assassinations in which exactly one person was killed: {:,.2f}'.format(percent_one_killed))
data = []

method_name_list = []

fig, ax = plt.subplots(figsize = (16, 10))

for method in df['attacktype1_txt'].unique():

    method_data = df[df['attacktype1_txt'] == method]['nkill'].dropna().clip(0, 100)

    data.append(method_data)

    method_name_list.append(method)



plt.boxplot(data)

ax.set_xticklabels(method_name_list, rotation = 70)

plt.title('Boxplot of Numbers Killed by Attack Type (for < 100 killed)', fontsize = 22, y = 1.02)

plt.ylabel('Number Killed')

sns.despine(left = True);
data = []

method_name_list = []

fig, ax = plt.subplots(figsize = (16, 10))

for method in df['attacktype1_txt'].unique():

    method_data = df[df['attacktype1_txt'] == method]['nkill'].dropna()

    data.append(method_data)

    method_name_list.append(method)



plt.boxplot(data)

ax.set_xticklabels(method_name_list, rotation = 70)

plt.title('Boxplot of Numbers Killed by Attack Type', fontsize = 22, y = 1.02)

plt.ylabel('Number Killed')

sns.despine(left = True);
weapon = df.groupby('weaptype1_txt').count()['eventid'].sort_values(ascending = False)



fig, ax = plt.subplots(figsize = (16, 10))

plt.bar(weapon.index, weapon.values, color = 'yellow', alpha = 0.96)

ax.set_facecolor('black')

plt.xticks(rotation = 84)

plt.title('Number of Attacks by Weapon Used', fontsize = 22, y = 1.03);
assassination = df[df['attacktype1_txt'] == 'Assassination']



as_weapons = assassination.groupby('weaptype1_txt').count()['eventid'].sort_values(ascending = False)



print('Number of assassinations by weapon type:')

for index, row in as_weapons.iteritems():

    print(index, row)
weapon_kills = df.groupby('weaptype1_txt').mean()['nkill'].sort_values(ascending = False)

weapon_kills.index.name = ''

print('Average number killed per attack by weapon type:')

weapon_kills
wtc_removed = df.drop([73126, 73127])

weapon_kills = wtc_removed.groupby('weaptype1_txt').mean()['nkill'].sort_values(ascending = False)

weapon_kills.index.name = ''

print('Average number killed per attack by weapon type, with WTC attack removed:')

weapon_kills
suicide = df.groupby('suicide')['eventid'].count()



fig, ax = plt.subplots(figsize = (6, 6))

plt.bar(suicide.index, suicide.values, color = 'darkred', alpha = 0.9)

plt.xticks([0, 1])

ax.set_xticklabels(['Non-Suicide', 'Suicide'], fontsize = 16, y = - 0.02)

#ax.tick_params(size = 0)

plt.title('Number of Incidents by Whether They Were Suicide Attacks', fontsize = 16, y = 1.04)

sns.despine(left = True)

plt.show()



suicide_percentage = (suicide[1] / (suicide[0] + suicide[1])) * 100

print('Percentage of incidents which were suicide attacks: {:,.1f}%\n'.format(suicide_percentage))





suicide_average_killed = df.groupby('suicide').sum()['nkill'] / suicide



print('The average number killed per suicide attack was {:,.2f}'.format(suicide_average_killed[1]))

print('In comparison, the average number killed per non-suicide attack was {:,.2f}'.format(suicide_average_killed[0]))

print('So suicide attacks are rare in history, but tend to be extremely deadly')

print('This effect is not contingent on the inclusion of the World Trade Center Attacks')
suicide_over_time = df.groupby('iyear').mean()['suicide'] * 100



fig, ax = plt.subplots(figsize = (16, 7))

plt.plot(suicide_over_time.index, suicide_over_time.values, color = 'darkred', linewidth = 5)

plt.title('Suicide Attacks over Time', fontsize = 22, y = 1.05)

plt.yticks(fontsize = 14)

plt.xticks(fontsize = 14)

plt.ylabel('% of Attacks Which Were Suicidal', fontsize = 16)

sns.despine(left = True);
target = df.groupby('targtype1_txt')['eventid'].count().sort_values(ascending = False)



fig, ax = plt.subplots(figsize = (20, 10))

sns.barplot(target.index, target.values, palette = 'plasma')

plt.title('Number of Attacks by Target Type', fontsize = 26, y = 1.02)

plt.xticks(rotation = 80)

plt.xlabel('')

plt.setp(ax.get_xticklabels(), fontsize = 16)

sns.despine(left = True)

plt.show()
def plot_bar(region, axis):

    region_data = df[df['region_txt'] == region]

    

    target = region_data.groupby('targtype1_txt')['eventid'].count().sort_values(ascending = False)



    axis.bar(target.index, target.values, color = 'black')

    axis.set_title(region, fontsize = 22, y = 1.02)

    for tick in axis.get_xticklabels():

        tick.set_rotation(90)

    plt.setp(axis.get_xticklabels(), fontsize = 16)

    sns.despine(left = True)

    

fig, axes = plt.subplots(6, 2, figsize = (20, 60))

plt.suptitle('Number of Attacks by Target Type in Each Region', fontsize = 26, y = 0.905)

plt.subplots_adjust(hspace = 1.2)



count = 0

for country in df['region_txt'].unique()[:6]:

    plot_bar(country, axes[count, 0])

    count += 1



count = 0

for country in df['region_txt'].unique()[6:]:

    plot_bar(country, axes[count, 1])

    count += 1
abortion = df[df['targtype1_txt'] == 'Abortion Related']

print('Number of abortion related terror attacks which have taken place in North America since 1970:', len(abortion))

print('Number of those attacks which took place in the United States:', len(abortion['country_txt'] == 'United States'))
target = df.groupby('targtype1_txt')['eventid'].count().sort_values(ascending = False)

average_target_killed = (df.groupby('targtype1_txt').sum()['nkill'] / target).sort_values(ascending = False)



fig, ax = plt.subplots(figsize = (20, 10))

plt.bar(average_target_killed.index, average_target_killed.values, color = 'black')

plt.title('Average Number Killed by Target Type', fontsize = 22, y = 1.02)

plt.xticks(rotation = 80, fontsize = 14)

sns.despine(left = True);
average_target_killed = (df.groupby('targtype1_txt').sum()['nkill'] / target)

std_target_killed = (df.groupby('targtype1_txt').std()['nkill'])

#average_target_killed.sort_values(ascending = False)

std_target_killed



target_killed = pd.DataFrame({'average': average_target_killed.values, 'standard deviation': std_target_killed.values}, 

                             index = average_target_killed.index).sort_values(by = 'average', ascending = False)



fig, ax = plt.subplots(figsize = (16, 10))

plt.bar(target_killed.index, target_killed['average'], yerr = target_killed['standard deviation'], color = 'grey', alpha = 0.6)

plt.title('Average Number Killed by Target Type, with error bars', fontsize = 20, y = 1.02)

plt.xticks(rotation = 90, fontsize = 14)

plt.ylim(0, 25)

sns.despine(left = True);
data = []

method_name_list = []

fig, ax = plt.subplots(figsize = (16, 10))

for method in target_killed.index:

    method_data = df[df['targtype1_txt'] == method]['nkill'].dropna()

    data.append(method_data)

    method_name_list.append(method)



plt.boxplot(data)

ax.set_xticklabels(method_name_list, rotation = 90, fontsize = 14)

plt.title('Boxplot of Numbers Killed by Target Type', fontsize = 22, y = 1.02)

plt.ylabel('Number Killed')

sns.despine(left = True);
group = df.groupby('gname').count()['eventid'].sort_values(ascending = False)



worst_groups = group[1:31]



plt.subplots(figsize = (10, 12))

plt.barh(worst_groups.index, worst_groups.values, color = 'black')

plt.title('Groups Which Have Made the Most Attacks', fontsize = 20, y = 1.02)

plt.xlabel('Number of Attacks', fontsize = 14)

plt.show()



print('Together, these 30 groups have made {} attacks'.format(worst_groups.sum()))

print('That is {:,.1f}% of the attacks in the dataset\n'.format((worst_groups.sum() / len(df) * 100)))





print('Number of incidents for which the group responsible was unknown:', group['Unknown'])

print('Percent of total incidents: {:,.1f}%\n'.format((group['Unknown'] / len(df)) * 100))



groups_10 = group[group >= 10]

print('Number of groups which perpetrated 10 or more attacks:', len(groups_10))

print('There are {} groups listed in total, so {:,.1f}% of the groups perpetrated 10 or more attacks\n'.format(

    len(group), len(groups_10) / len(group) * 100))



print('Collectively, these groups which have done 10 or more attacks commited {} attacks'.format(groups_10.drop('Unknown').sum()))

print('That is {:,.1f}% of all attacks'.format((groups_10.drop('Unknown').sum() / len(df)) * 100))

print('That, together with the attacks for which there is no information, accounts for {:,.1f}% of attacks\n'.format((groups_10.sum() / len(df)) * 100))





individual = df[df['individual'] == 1]



percent_individual = (len(individual) / len(df)) * 100

print('Percentage of attacks which were perpetrated by an independent actor: {:,.2}%'.format(percent_individual))
independent = df[df['individual'] == 1].groupby('region_txt').count()['eventid'].sort_values(ascending = False)

independent.index.name = ''



fig, ax = plt.subplots(figsize = (16, 8))

plt.bar(independent.index, independent.values, color = 'black')

plt.xticks(rotation = 80)

sns.despine(left = True)

plt.title('Number of Lone Wolf Attacks by Region', fontsize = 18, y = 1.02);
independent = df[df['individual'] == 1].groupby('iyear').count()['eventid']



fig, ax = plt.subplots(figsize = (14, 8))

plt.plot(independent.index, independent.values, color = 'red', linewidth = 3)

ax.set_facecolor('grey')

fig.set_facecolor('lightgrey')

plt.setp(ax.get_xticklabels(), fontsize = 14)

plt.setp(ax.get_yticklabels(), fontsize = 14)

plt.title('Number of Lone Wolf Attacks by Year', fontsize = 20, y = 1.03)

plt.show()





year2017 = df[df['iyear'] == 2017]

na = year2017[year2017['region_txt'] == 'North America']

na_independent = na[na['individual'] == 1]

eu = year2017[year2017['region_txt'] == 'Western Europe']

eu_independent = eu[eu['individual'] == 1]

print('Percent of North American attacks which were independent in 2017: {:,.1f}%'.format((len(na_independent) / len(na)) * 100))

print('Percent of Western European attacks which were independent in 2017: {:,.1f}%'.format((len(eu_independent) / len(eu)) * 100))
individual_killed = df.groupby('individual').sum()['nkill'] / df.groupby('individual').count()['eventid']



print('Average number killed in non-independent attacks: {:,.1f}'.format(individual_killed[0]))

print('Average number killed in independent attacks: {:,.1f}'.format(individual_killed[1]))
group = df.groupby('gname').count()['eventid'].sort_values(ascending = False)

worst_groups = group[1:31]



average_kills_dict = {}

for group in worst_groups.index:

    group_data = df[df['gname'] == group]['nkill']

    

    average_kills_dict[group] = group_data.mean()

    

average_kills = pd.Series(average_kills_dict).sort_values(ascending = False)



fig, ax = plt.subplots(figsize = (20, 10))

plt.bar(average_kills.index, average_kills.values, color = 'r')

plt.xticks(rotation = 90)

plt.title('Average Number Killed Per Attack, for groups with most attacks', fontsize = 24, y = 1.04)

sns.despine(left = 'True')

ax.set_facecolor('black')

plt.setp(ax.get_yticklabels(), fontsize = 16)

plt.setp(ax.get_xticklabels(), fontsize = 14);
print('Number of FLNC attacks:', len(df[df['gname'] == 'Corsican National Liberation Front (FLNC)']['nkill']))

print('Number of people FLNC has killed:', int(df[df['gname'] == 'Corsican National Liberation Front (FLNC)']['nkill'].sum()))



print()

print('Number of FDN attacks:', len(df[df['gname'] == 'Nicaraguan Democratic Force (FDN)']))

print('Number of people the FDN has killed:', int(df[df['gname'] == 'Nicaraguan Democratic Force (FDN)']['nkill'].sum()))
success = df.groupby('success')['eventid'].count()



fig, ax = plt.subplots(figsize = (8, 8))

plt.bar(success.index, success.values, color = 'black')

plt.xticks([0, 1])

plt.title('Number of Attacks by Successfulness', fontsize = 18, y = 1.04)

ax.set_xticklabels(['Failure', 'Success'], fontsize = 16, y = - 0.02)

sns.despine(left = True)

plt.show()



percent_successful = (success[1] / (success[0] + success[1])) * 100

print('Percentage of attacks which were successful: {:,.2f}'.format(percent_successful))
country_success_rate = df.groupby('country_txt').mean()['success']



data = dict(type = 'choropleth', 

            locations = country_success_rate.index,

            locationmode = 'country names',

            z = country_success_rate.values, 

            text = country_success_rate.index)

layout = dict(title = 'Percentage of Attacks Which Were Successful', 

              height = 1200,

              geo = dict(showframe = False, 

                         projection = {'type': 'mercator'}))



fig = go.Figure(data = [data], layout=layout)

iplot(fig)
country_success_rate = df.groupby('country_txt').mean()['success']



print('Countries with the lowest success rate:')

for index, row in country_success_rate.sort_values().head(20).iteritems():

    print(index, '{:,.1f}'.format(row))
grouped = df.groupby('country_txt')

killed = grouped.count()['nkill']

success_rate = grouped.mean()['success'] * 100

fig, ax = plt.subplots(figsize = (12, 8))

sns.scatterplot(killed, success_rate, color = 'black')

plt.title('Scatterplot of Countries by Success Rate and Number Killed', fontsize = 20, y = 1.02)

plt.ylabel('% Attacks Successful in Country', fontsize = 16)

plt.xlabel('Number Killed in Country', fontsize = 16)

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

sns.despine(left = True);
year_success = df.groupby('iyear').mean()['success'] * 100



fig, ax = plt.subplots(figsize = (18, 8))

plt.plot(year_success.index, year_success.values, color = 'black', linewidth = 5)

plt.title('Success Rate over Time', fontsize = 22, y = 1.03)

plt.ylabel('Percentage of Attacks which were Successful', fontsize = 14)

plt.yticks(fontsize = 14)

plt.xticks(fontsize = 16)

plt.ylim(0, 100)

sns.despine(left = True);
n = len(df['region_txt'].unique())

color = plt.cm.rainbow(np.linspace(0,1,n))

c = zip(range(n), color)

colors = [color for color in c]

count = 0



fig, ax = plt.subplots(figsize = (16, 10))

for region in df['region_txt'].unique():

    region_data = df[df['region_txt'] == region]

    year_success = region_data.groupby('iyear').mean()['success'] * 100



    plt.plot(year_success, label = region, color = colors[count][1])

    count += 1

    

plt.legend(bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0.)

plt.title('Success Rate Over Time by Region', fontsize = 22, y = 1.02)

sns.despine(left = True);
n = 4

color = plt.cm.viridis(np.linspace(0,1,n))

c = zip(range(n), color)

colors = [color for color in c]

count = 0



fig, ax = plt.subplots(figsize = (16, 12))

for region in ['Sub-Saharan Africa', 'Middle East & North Africa', 'South Asia', 'Southeast Asia']:

    region_data = df[df['region_txt'] == region]

    year_success = region_data.groupby('iyear').mean()['success'] * 100

    

    plt.plot(year_success, linewidth = 4, color = colors[count][1])

    plt.text(year_success.index[-1] + 0.5, year_success.values[-1], region)

    count += 1

    

sns.despine(left = True)

plt.title('Success Rate Over Time in Most Troubled Regions', fontsize = 22, y = 1.02);
region_success = df.groupby(['region_txt', 'success'])['eventid'].count()



fraction_dict = {}

for region in df['region_txt'].unique():

    success = region_success[region][1]

    failure = region_success[region][0]

    fraction_successful = (success / (success + failure)) * 100

    fraction_dict[region] = fraction_successful



fraction_series = pd.Series(fraction_dict).sort_values(ascending = False)



plt.subplots(figsize = (12, 6))

plt.bar(fraction_series.index, fraction_series.values, alpha = 0.6)

plt.xticks(rotation = 80)

plt.title('Percentage of Attacks Which Were Successful By Region', fontsize = 18, y = 1.02)

sns.despine(left = True)

plt.show()
method_success = df.groupby(['attacktype1_txt', 'success'])['eventid'].count()



fraction_dict = {}

for method in df['attacktype1_txt'].unique():

    success = method_success[method][1]

    failure = method_success[method][0]

    fraction_successful = (success / (success + failure)) * 100

    fraction_dict[method] = fraction_successful



fraction_series = pd.Series(fraction_dict).sort_values(ascending = False)



plt.subplots(figsize = (12, 6))

plt.bar(fraction_series.index, fraction_series.values, alpha = 0.6)

plt.xticks(rotation = 80)

plt.title('Percentage of Attacks Which Were Successful By Attack Type', fontsize = 18, y = 1.02)

sns.despine(left = True)

plt.show()
target_success = df.groupby(['targtype1_txt', 'success'])['eventid'].count()



fraction_dict = {}

for target in df['targtype1_txt'].unique():

    success = target_success[target][1]

    failure = target_success[target][0]

    fraction_successful = (success / (success + failure)) * 100

    fraction_dict[target] = fraction_successful



fraction_series = pd.Series(fraction_dict).sort_values(ascending = False)



plt.subplots(figsize = (12, 6))

plt.bar(fraction_series.index, fraction_series.values, alpha = 0.6)

plt.xticks(rotation = 90)

plt.title('Percentage of Attacks Which Were Successful By Target Type', fontsize = 18, y = 1.02)

sns.despine(left = True)

plt.show()
individual = df[df['individual'] == 1]

independent_success = individual.groupby('success')['eventid'].count()



print('Percentage of attacks by independent actors which were successful: {:,.1f}%'.format(

        (independent_success[1] / (independent_success[1] + independent_success[0])) * 100))