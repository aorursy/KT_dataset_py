import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from scipy import stats
from scipy.stats.distributions import beta
from scipy.special import beta as beta_func

import pymc3 as pm

import pandas as pd

import datetime

import seaborn as sns

import folium
from folium.plugins import MarkerCluster

from IPython.display import HTML

plt.style.use('bmh')
data = pd.read_csv('../input/police_killings.csv', encoding = "ISO-8859-1")
data
months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
          'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11,
          'December': 12}

race = {'Black': 1, 'White':2, 'Hispanic/Latino':3, 'Native American':4, 'Unknown': 5}
locations = []

names = []

state = []

dates = []
black_dates = []
white_dates = []
hispanic_dates =[]
native_dates = []
unknown_dates = []
asian_pacific_dates = []

black_income = []
white_income = []
hispanic_income =[]
native_income = []
unknown_income = []
asian_pacific_income = []

black_women = 0
white_women = 0
hispanic_women = 0
native_women = 0
asian_pacific_women = 0

black_unarmed = 0
white_unarmed = 0
hispanic_unarmed = 0
native_unarmed = 0
asian_pacific_unarmed = 0


for i in range(len(data['day'])):    
    names.append(data['name'][i])
    
    state.append(data['state'][i])
    
    locations.append([data['latitude'][i], data['longitude'][i]])
    
    dates.append(datetime.datetime(data['year'][i], months[data['month'][i]], data['day'][i]))
    
    if data['raceethnicity'][i] == 'Black':
        black_dates.append(datetime.datetime(data['year'][i], months[data['month'][i]], data['day'][i]))
        black_income.append(data['p_income'][i])
        if data['gender'][i] == 'Female':
            black_women += 1
        if data['armed'][i] == 'No':
            black_unarmed += 1
        
    elif data['raceethnicity'][i] == 'White':
        white_dates.append(datetime.datetime(data['year'][i], months[data['month'][i]], data['day'][i]))
        white_income.append(data['p_income'][i])
        if data['gender'][i] == 'Female':
            white_women += 1
        if data['armed'][i] == 'No':
            white_unarmed += 1
        
    elif data['raceethnicity'][i] == 'Hispanic/Latino':
        hispanic_dates.append(datetime.datetime(data['year'][i], months[data['month'][i]], data['day'][i]))
        hispanic_income.append(data['p_income'][i])
        if data['gender'][i] == 'Female':
            hispanic_women += 1
        if data['armed'][i] == 'No':
            hispanic_unarmed += 1
        
    elif data['raceethnicity'][i] == 'Native American':
        native_dates.append(datetime.datetime(data['year'][i], months[data['month'][i]], data['day'][i]))
        native_income.append(data['p_income'][i])
        if data['gender'][i] == 'Female':
            native_women += 1
        if data['armed'][i] == 'No':
            native_unarmed += 1
        
    elif data['raceethnicity'][i] == 'Asian/Pacific Islander':
        asian_pacific_dates.append(datetime.datetime(data['year'][i], months[data['month'][i]], data['day'][i]))
        asian_pacific_income.append(data['p_income'][i])
        if data['gender'][i] == 'Female':
            asian_pacific_women += 1
        if data['armed'][i] == 'No':
            asian_pacific_unarmed += 1
        
    elif data['raceethnicity'][i] == 'Unknown':
        unknown_dates.append(datetime.datetime(data['year'][i], months[data['month'][i]], data['day'][i]))
        unknown_income.append(data['p_income'][i])

# personal income by race
black_income = [int(i) for i in black_income if i != '-']
white_income = [int(i) for i in white_income if i != '-']
hispanic_income = [int(i) for i in hispanic_income if i != '-']
native_income = [int(i) for i in native_income if i != '-']
unknown_income = [int(i) for i in unknown_income if i != '-']
asian_pacific_income = [int(i) for i in asian_pacific_income if i != '-']
mpl_dates = mdates.date2num(dates)
mpl_black_dates = mdates.date2num(black_dates)
mpl_white_dates = mdates.date2num(white_dates)
mpl_hispanic_dates = mdates.date2num(hispanic_dates)
mpl_native_dates = mdates.date2num(native_dates)
mpl_asian_pacific_dates = mdates.date2num(asian_pacific_dates)
mpl_unknown_dates = mdates.date2num(unknown_dates)
fig, ax = plt.subplots(1,1)
plt.title('Police Killings January to June 2015')
plt.xlabel('Date')
ax.hist(mpl_dates, bins=40, color='darkred', alpha = .7)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
plt.show()
fig, ax = plt.subplots(figsize=(8,5))
plt.title('Police Killings 2015 Race/Ethnicity Breakdown')
plt.xlabel('Date')
ax.hist(mpl_black_dates, bins=30, color='k', alpha = .5, label='Black')
ax.hist(mpl_hispanic_dates, bins=30, color='b', alpha = .4, label = 'Hispanic')
ax.hist(mpl_white_dates, bins=30, color='r', alpha = .2, label='White')
ax.hist(mpl_native_dates, bins=30, color='g', alpha = .4, label='Native American')
ax.hist(mpl_unknown_dates, bins=30, color='c', alpha = .4, label='Race Unknown')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
plt.legend()
plt.show()
# These population percent numbers have been pulled from wikipedia.

white_percent = 0.733
black_percent = 0.216
native_percent = 0.008
asian_pacific_percent = 0.054


# These population numbers have been pulled from wikipedia.

white_pop = 233657078
black_pop = 40241818
native_pop = 2597817
asian_pacific_pop = 16614625 + 560021

black_total = len(black_dates)
white_total = len(white_dates)
hispanic_total = len(hispanic_dates)
native_total = len(native_dates) 
asian_pacific_total = len(asian_pacific_dates)
unknown_total = len(unknown_dates)
n_groups = 5

both_totals = [(black_total), (white_total), (native_total), (asian_pacific_total), hispanic_total]

men_totals = [(black_total - black_women), (white_total - white_women) ,
                    (native_total - native_women) , (asian_pacific_total- asian_pacific_women) , 
                (hispanic_total - hispanic_women)]

women_totals = [(black_women), (white_women), (native_women) , (asian_pacific_women) , hispanic_women]

fig, ax = plt.subplots(figsize=(8,5))

index = np.arange(n_groups)
bar_width = 0.35

opacity = 1

rects1 = ax.bar(index, men_totals, bar_width, alpha=opacity, color='b', label='Men')

rects2 = ax.bar(index + bar_width, women_totals, bar_width,
                alpha=opacity, color='pink',
                label='Women')

rects3 = ax.bar(index + (.5*bar_width), both_totals, 2 * bar_width,
                alpha=0.2, color='k',
                label='Gender Total')

ax.set_xlabel('Race/Ethnicity')
ax.set_ylabel('Number of Victims')
ax.set_title('Number of Race Killed By Police')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('{}{}'.format('Black:',black_total), 
                   '{}{}'.format('White:',white_total), 
                   '{}{}'.format('Native:',native_total), 
                   '{}{}'.format('Asian/Pacific:',asian_pacific_total), 
                   '{}{}'.format('Hispanic:',hispanic_total)))
ax.legend()

plt.show()
n_groups = 4

both_percents = ((black_total) / black_pop, (white_total) / white_pop,
            (native_total) / native_pop, 
            (asian_pacific_total) / asian_pacific_pop)

men_percents = ((black_total - black_women) / black_pop, (white_total - white_women) / white_pop,
            (native_total - native_women) / native_pop, 
            (asian_pacific_total- asian_pacific_women) / asian_pacific_pop)

women_percents = ((black_women) / black_pop, (white_women) / white_pop,
            (native_women) / native_pop, (asian_pacific_women) / asian_pacific_pop)

fig, ax = plt.subplots(figsize=(8,5))

index = np.arange(n_groups)
bar_width = 0.35

opacity = 1

rects1 = ax.bar(index, men_percents, bar_width,
                alpha=opacity, color='b',
                label='Men')

rects2 = ax.bar(index + bar_width, women_percents, bar_width,
                alpha=opacity, color='pink',
                label='Women')

rects3 = ax.bar(index + (.5*bar_width), both_percents, 2 * bar_width,
                alpha=0.2, color='k',
                label='Gender Total')

ax.set_xlabel('Race/Ethnicity')
ax.set_ylabel('Percent')
ax.set_title('Percent of Race Killed By Police')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Black', 'White', 'Native', 'Asian'))
ax.legend()

plt.show()
black_unarmed_percent = black_unarmed/black_total
white_unarmed_percent = white_unarmed/white_total
hispanic_unarmed_percent = hispanic_unarmed/hispanic_total
native_unarmed_percent = native_unarmed/native_total
asian_pacific_unarmed_percent = asian_pacific_unarmed/asian_pacific_total
n_groups = 5

unarmed_percent = (black_unarmed_percent, white_unarmed_percent, hispanic_unarmed_percent, 
                   native_unarmed_percent, asian_pacific_unarmed_percent)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 1

rects1 = ax.bar(index + (.5*bar_width), unarmed_percent, 2 * bar_width,
                alpha=1, color='darkred',
                label='Unarmed')

ax.set_xlabel('Race/Ethnicity')
ax.set_ylabel('Percent')
ax.set_title('Percent of Racial Group Unarmed and Killed By Police')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Black', 'White', 'Hispanic', 'Native', 'Asian/Pacific Is.'))

plt.show()
state_count = {}
count = []
top7_states = {}
top7_nums = []


for i in state:
    state_count[i] = state.count(i)

for i in state_count:
    count.append(state_count[i])

top7_nums = sorted(count)[-7:]
top7_nums.reverse()

for i in state_count:
    if state_count[i] in top7_nums:
        top7_states[i] = state_count[i]

#population estimates from factfinder.census.gov for US states in 2015

ca_pop = 39144818
tx_pop = 27469114
fl_pop = 20271272
az_pop = 6828065
ok_pop = 3911338
ga_pop = 10214860
ny_pop = 19795791

ca_percent = top7_states['CA'] / ca_pop
tx_percent = top7_states['TX'] / tx_pop
fl_percent = top7_states['FL'] / fl_pop
az_percent = top7_states['AZ'] / az_pop
ok_percent = top7_states['OK'] / ok_pop
ga_percent = top7_states['GA'] / ga_pop
ny_percent = top7_states['NY'] / ny_pop

state_percents = [ca_percent,tx_percent,fl_percent,az_percent,ok_percent,ga_percent,ny_percent]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:gray', 'tab:cyan']

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,5))
plt.setp(ax1, yticks=range(len(top7_states)), yticklabels=sorted(top7_states, key=top7_states.get, reverse=True))

ax1.barh(range(len(top7_states)), top7_nums, color=colors, alpha=.75)

ax1.set_title('Top 7 States with Most Killings')


ax2.barh(range(len(top7_states)), state_percents, color=colors, alpha=.75)

ax2.set_title('Per State Capita');
race = [i for i in data['raceethnicity']]

m = folium.Map(location=[40, -100], tiles='Open Street Map',
                   zoom_start=3, control_scale=True)



marker_cluster = folium.plugins.MarkerCluster().add_to(m)

for index in range(0, len(data)):
    folium.Marker(locations[index], popup= race[index]).add_to(marker_cluster)
m
m = folium.Map(location=[40, -100], tiles='Open Street Map',
                   zoom_start=4, control_scale=True)

marker_cluster = folium.plugins.MarkerCluster().add_to(m)

for index in range(0, len(data)):
    if data['raceethnicity'][index] == 'Black':
        folium.Marker(locations[index], popup= data['cause'][index]).add_to(marker_cluster)
m
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
life_expectancy = 78.96

plt.figure(figsize=(10,5))
age = [int(i) for i in data['age'] if i != 'Unknown']
plt.hist(age, color = 'darkred', bins=len(set(age)), alpha = .6)
plt.xlabel('Age')
plt.ylabel('Number of Police Killings')
plt.axvline(np.mean(age), color='darkred', linestyle='dashed', alpha=1,
            linewidth=3, label='{}{}'.format('Average Age: ',round(np.mean(age))))
plt.axvline(life_expectancy, color='k', linestyle='dashed', alpha=1,
            linewidth=3, label='{}{}'.format('Life Expectancy ',life_expectancy))
plt.title('Age of Persons Killed')
plt.legend();
plt.figure(figsize=(10,5))
income0 = [float(i) for i in data['p_income'] if i != '-']
plt.hist(income0, alpha=.4, color = 'darkgreen', bins=50)
plt.axvline(np.mean(income0), color='darkgreen', linestyle='dashed', alpha=.8,
            linewidth=3, label='{}{}'.format('Average Personal Income: ',round(np.mean(income0))))
plt.xlabel('Personal Income')
plt.ylabel('Number of Police Killings')
plt.title('Personal Income of Persons Killed')
plt.legend();
national_median_h_income = 56516

plt.figure(figsize=(10,5))
income1 = [int(i) for i in data['h_income'] if len(str(i)) > 3]
plt.hist(income1, alpha=.4, color = 'darkgreen', bins=50)
plt.axvline(np.mean(income1), color='darkgreen', linestyle='dashed', alpha=.8,
            linewidth=3, label='{}{}'.format('Median Household Income of Victims: ',round(np.median(income1))))
plt.axvline(national_median_h_income, color='b', linestyle='dashed', alpha=.8,
            linewidth=3, label='{}{}'.format('National Household Median Income (2015): ',national_median_h_income))
plt.xlabel('Household Income')
plt.ylabel('Number of Police Killings')
plt.title('Household Income of Persons Killed')
plt.legend();
plt.figure(figsize=(15,5))
plt.hist(black_income, alpha=1, color = 'k', bins=50, label='Black Personal Income')
plt.hist(white_income, alpha=.5, color = 'r', bins=50, label='White Personal Income')
plt.hist(income0, alpha=.2, color = 'green', bins=50, label='Total Personal Income')

plt.axvline(np.mean(black_income), color='k', linestyle='dashed', alpha=.8,
            linewidth=3, label=('{}{}'.format('Black Mean: ',round(np.mean(black_income)))))

plt.axvline(np.mean(white_income), color='r', linestyle='dashed', alpha=.5,
            linewidth=3, label=('{}{}'.format('White Mean: ',round(np.mean(white_income)))))

plt.axvline(np.mean(income0), color='green', linestyle='dashed', alpha=.4,
            linewidth=3, label='{}{}'.format('Total Mean: ',round(np.mean(income0))))


plt.xlabel('Personal Income')
plt.ylabel('Number of Police Killings')
plt.title('Personal Income of Persons Killed')
plt.legend();
plt.figure(figsize=(8,8))
weapon = [i for i in data['armed']]
weapon_dict = {x:weapon.count(x) for x in weapon}

plt.pie(weapon_dict.values(), labels=weapon_dict.keys(),autopct='%.2f',startangle=180)
plt.title('Victim "Armed" Status');
black_weapon = [data['armed'][i] for i in range(len(data['armed'])) if data['raceethnicity'][i] == 'Black']
black_weapon_dict = {x:black_weapon.count(x) for x in black_weapon}

white_weapon = [data['armed'][i] for i in range(len(data['armed'])) if data['raceethnicity'][i] == 'White']
white_weapon_dict = {x:white_weapon.count(x) for x in white_weapon}


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,8))

ax1.pie(black_weapon_dict.values(), labels=black_weapon_dict.keys(),autopct='%.2f',startangle=180)
ax1.set_title('Black Victim "Armed" Status');

ax2.pie(white_weapon_dict.values(), labels=white_weapon_dict.keys(),autopct='%.2f',startangle=180)
ax2.set_title('White Victim "Armed" Status');
plt.figure(figsize=(8,8))
cause = [i for i in data['cause']]
cause_dict = {x:cause.count(x) for x in cause}

plt.pie(cause_dict.values(), labels=cause_dict.keys(), autopct='%.2f', startangle=180)
plt.title('Cause of Death');
black_cause = [data['cause'][i] for i in range(len(data['cause'])) if data['raceethnicity'][i] == 'Black']
black_cause_dict = {x:black_cause.count(x) for x in black_cause}

white_cause = [data['cause'][i] for i in range(len(data['cause'])) if data['raceethnicity'][i] == 'White']
white_cause_dict = {x:white_cause.count(x) for x in white_cause}


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,8))

ax1.pie(black_cause_dict.values(), labels=black_cause_dict.keys(),autopct='%.2f',startangle=180)
ax1.set_title('Black Victim Cause of Death');

ax2.pie(white_cause_dict.values(), labels=white_cause_dict.keys(),autopct='%.2f',startangle=180)
ax2.set_title('White Victim Cause of Death');
