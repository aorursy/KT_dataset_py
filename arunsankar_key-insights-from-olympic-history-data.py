from IPython.display import Image
Image(filename="../input/olympics-logo/Olympics Logos.png")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input/120-years-of-olympic-history-athletes-and-results/"))
athlete = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
noc = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')

print('Athlete data: \nRows: {}\nCols: {}'.format(athlete.shape[0],athlete.shape[1]))
print(athlete.columns)

print('\nNOC data: \nRows: {}\nCols: {}'.format(noc.shape[0],noc.shape[1]))
print(noc.columns)
athlete.head()
noc.head()
for x in athlete.columns:
    if athlete[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,athlete[x].isnull().values.ravel().sum()))

athlete['Medal'] = athlete['Medal'].fillna('No Medal')
for x in noc.columns:
    if noc[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,noc[x].isnull().values.ravel().sum()))
noc['region'].fillna(noc['notes'], inplace=True)
athlete = athlete.merge(noc, how='left', on='NOC')
athlete['Medal_Flag'] = athlete['Medal'].apply(lambda x: 0 if(x=='No Medal') else 1)
games_athletes = athlete.pivot_table(athlete, index=['Games'], aggfunc=lambda x: len(x.unique())).reset_index()[['Games','ID']]
fig, ax = plt.subplots(figsize=(22,6))
a = sns.barplot(x='Games', y='ID', data=games_athletes, ax=ax, color="#2196F3")
a.set_xticklabels(labels=games_athletes['Games'],rotation=90)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

ax.set_xlabel('Olympic Game', size=14, color="#0D47A1")
ax.set_ylabel('Number of Athletes', size=14, color="#0D47A1")
ax.set_title('Athletes in each Olympic game', size=18, color="#0D47A1")

plt.show()
summer = athlete[athlete['Season']=='Summer']

summer_athletes = summer.pivot_table(summer, index=['Year'], aggfunc=lambda x: len(x.unique())).reset_index()[['Year','ID']]
summer_sports = summer.groupby('Year')['Sport'].nunique().reset_index()
summer_events = summer.groupby('Year')['Event'].nunique().reset_index()

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(22,18))

sns.barplot(x='Year', y='ID', data=summer_athletes, ax=ax[0], color="palegreen")
sns.barplot(x='Year', y='Sport', data=summer_sports, ax=ax[1], color="palegreen")
sns.barplot(x='Year', y='Event', data=summer_events, ax=ax[2], color="palegreen")

j = 0
for i in ['Athletes', 'Sports', 'Events']:
    ax[j].set_xlabel('Year', size=14, color="green")
    ax[j].set_ylabel(i, size=14, color="green")
    ax[j].set_title(i + ' in each Summer Olympic game', size=18, color="green")
    j = j + 1
    
for i in range(3):
    for p in ax[i].patches:
        ax[i].text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),
                fontsize=12, color='black', ha='center', va='bottom')
plt.show()
summer_games_athletes = athlete[athlete['Season']=='Summer'].pivot_table(athlete, index=['Games'], 
                                                                         aggfunc=lambda x: len(x.unique())).reset_index()[['Games','ID']]
summer_games_athletes['ID1'] = summer_games_athletes['ID'].shift(1)
summer_games_athletes['Growth'] = ((summer_games_athletes['ID']-summer_games_athletes['ID1']))/summer_games_athletes['ID1']
summer_games_athletes.dropna(inplace=True)

fig, ax = plt.subplots(figsize=(22,6))
a = sns.barplot(x='Games', y='Growth', data=summer_games_athletes, ax=ax, color="palegreen")
a.set_xticklabels(labels=summer_games_athletes['Games'],rotation=90)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height(), '{:,.1%}'.format(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

ax.set_xlabel('Olympic Summer Game', size=14, color="green")
ax.set_ylabel('Growth of number of Athletes', size=14, color="green")
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
ax.set_title('YoY growth of Athlete participation in each Summer Olympic game', size=18, color="green")

plt.show()
sport_year = athlete[athlete['Season']=='Summer'].pivot_table(athlete, index=['Year','Sport'], aggfunc=lambda x: len(x.unique())).reset_index()[['Year','Sport','Event']]
sport_year = sport_year.pivot("Sport", "Year", "Event")
sport_year.fillna(0,inplace=True)
sport_year = sport_year.reindex(sport_year.sort_values(by=2016, ascending=False).index)
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(sport_year, annot=True, linewidths=0.05, ax=ax, cmap="RdYlGn")
ax.set_xlabel('Summer Game Year', size=14, color="green")
ax.set_ylabel('Sport', size=14, color="green")
ax.set_title('[Heatmap] Number of events in each sport over years', size=18, color="green")
plt.show()
game_sex = athlete[athlete['Season']=='Summer'].pivot_table(athlete, index=['Year','Sex'], aggfunc=lambda x: len(x.unique())).reset_index()[['Year','Sex','ID']]
game_sex = game_sex.pivot_table(game_sex, index=['Year'], columns='Sex', aggfunc=sum).reset_index()
game_sex.columns = ['Year','F','M']
game_sex.fillna(0,inplace=True)
game_sex['gender_ratio'] = game_sex['F'] / (game_sex['F'] + game_sex['M'])

fig, ax = plt.subplots(figsize=(8,6))
a = sns.scatterplot(x="M", y="F", hue="gender_ratio", palette='RdYlGn', data=game_sex, ax=ax)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']-175, point['y']+100, '{0:.0f}'.format(point['val']))

label_point(game_sex['M'],game_sex['F'],game_sex['Year'],ax)
        
ax.set_xlabel('Male Athletes', size=14, color="green")
ax.set_ylabel('Female Athletes', size=14, color="green")
ax.set_title('Male vs Female Athletes in Summer Olympic games', size=18, color="green")
plt.show()
fig, ax = plt.subplots(figsize=(20,6))
a = sns.boxplot(x="Year", y="Age", hue="Sex", palette={"M": "#B2EBF2", "F":"#F8BBD0"}, data=athlete[athlete['Season']=='Summer'], ax=ax)
        
ax.set_xlabel('Summer Game Year', size=14, color="green")
ax.set_ylabel('Age', size=14, color="green")
ax.set_title('Age distribution in Summer Olympic games', size=18, color="green")
plt.show()
year_sport_age = athlete[athlete['Season']=='Summer'].pivot_table(athlete, index=['Year','Medal'], aggfunc=np.mean).reset_index()[['Year','Medal','Age']]
year_sport_age = year_sport_age.pivot("Medal", "Year", "Age")
year_sport_age = year_sport_age.reindex(["Gold","Silver","Bronze","No Medal"])
f, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(year_sport_age, annot=True, linewidths=0.05, ax=ax, cmap="RdYlGn_r")
ax.set_xlabel('Summer Game Year', size=14, color="green")
ax.set_ylabel('Medal', size=14, color="green")
ax.set_title('[Heatmap] Mean Age of Medal Winners in Summer Olympic games', size=18, color="green")
plt.show()
t10_summer = athlete[(athlete['Season']=='Summer') & (athlete['Medal']!='No Medal')].groupby('region').count().reset_index()[['region','Medal']].sort_values('Medal', ascending=False).head(10)
f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="Medal", y="region", data=t10_summer, label="region", color="palegreen")

for p in ax.patches:
    ax.text(p.get_width() + 125,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Region', size=14, color="green")
ax.set_ylabel('Total Medals', size=14, color="green")
ax.set_title('[Horizontal Bar Plot] Top 10 countries with total medals in Summer Olympic games', size=18, color="green")
plt.show()
t3_summer = athlete[(athlete['Season']=='Summer') & (athlete['region'].isin(['USA'])) & (athlete['Medal']!='No Medal')]
t3_summer = pd.pivot_table(t3_summer, index=['Sport'], columns=['Year'], values=['ID'],  aggfunc=len, fill_value=0)
t3_summer = t3_summer.reindex(t3_summer['ID'].sort_values(by=2016, ascending=False).index)

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(t3_summer, annot=True, linewidths=0.05, ax=ax, cmap="RdYlGn")
ax.set_xlabel('Summer Game Year', size=14, color="green")
ax.set_ylabel('Sports', size=14, color="green")
ax.set_title('[Heatmap] USA Medals in Sports across Years', size=18, color="green")
plt.show()
t3_summer = athlete[(athlete['Season']=='Summer') & (athlete['region'].isin(['Russia'])) & (athlete['Medal']!='No Medal')]
t3_summer = pd.pivot_table(t3_summer, index=['Sport'], columns=['Year'], values=['ID'],  aggfunc=len, fill_value=0)
t3_summer = t3_summer.reindex(t3_summer['ID'].sort_values(by=2016, ascending=False).index)

f, ax = plt.subplots(figsize=(20, 14))
sns.heatmap(t3_summer, annot=True, linewidths=0.05, ax=ax, cmap="RdYlGn")
ax.set_xlabel('Summer Game Year', size=14, color="green")
ax.set_ylabel('Sports', size=14, color="green")
ax.set_title('[Heatmap] Russia Medals in Sports across Years', size=18, color="green")
plt.show()
t3_summer = athlete[(athlete['Season']=='Summer') & (athlete['region'].isin(['Germany'])) & (athlete['Medal']!='No Medal')]
t3_summer = pd.pivot_table(t3_summer, index=['Sport'], columns=['Year'], values=['ID'],  aggfunc=len, fill_value=0)
t3_summer = t3_summer.reindex(t3_summer['ID'].sort_values(by=2016, ascending=False).index)

f, ax = plt.subplots(figsize=(20, 14))
sns.heatmap(t3_summer, annot=True, linewidths=0.05, ax=ax, cmap="RdYlGn")
ax.set_xlabel('Summer Game Year', size=14, color="green")
ax.set_ylabel('Sports', size=14, color="green")
ax.set_title('[Heatmap] Germany Medals in Sports across Years', size=18, color="green")
plt.show()
winter_athletes = athlete[athlete['Season']=='Winter'].pivot_table(athlete, index=['Games'], aggfunc=len).reset_index()[['Games','Sport']]

fig, ax = plt.subplots(figsize=(22,6))
a = sns.barplot(x='Games', y='Sport', data=winter_athletes, ax=ax, color="lightskyblue")
a.set_xticklabels(labels=winter_athletes['Games'],rotation=90)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

ax.set_xlabel('Winter Olympic Game', size=14, color="#0D47A1")
ax.set_ylabel('Number of Athletes', size=14, color="#0D47A1")
ax.set_title('Athletes in each Winter Olympic game', size=18, color="#0D47A1")

plt.show()
winter_games_athletes = athlete[athlete['Season']=='Winter'].pivot_table(athlete, index=['Games'], 
                                                                         aggfunc=len).reset_index()[['Games','Sport']]
winter_games_athletes['Sport1'] = winter_games_athletes['Sport'].shift(1)
winter_games_athletes['Growth'] = ((winter_games_athletes['Sport']-winter_games_athletes['Sport1']))/winter_games_athletes['Sport1']
winter_games_athletes.dropna(inplace=True)

fig, ax = plt.subplots(figsize=(22,6))
a = sns.barplot(x='Games', y='Growth', data=winter_games_athletes, ax=ax, color="lightskyblue")
a.set_xticklabels(labels=winter_games_athletes['Games'],rotation=90)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height(), '{:,.1%}'.format(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

ax.set_xlabel('Olympic Winter Game', size=14, color="#0D47A1")
ax.set_ylabel('Growth of number of Athletes', size=14, color="#0D47A1")
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
ax.set_title('YoY growth of Athlete participation in each Winter Olympic game', size=18, color="#0D47A1")

plt.show()
winter_event = athlete[athlete['Season']=='Winter'].groupby('Games')['Sport'].nunique().reset_index()

fig, ax = plt.subplots(figsize=(22,6))
a = sns.barplot(x='Games', y='Sport', data=winter_event, ax=ax, color="lightskyblue")
a.set_xticklabels(labels=winter_event['Games'],rotation=90)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

ax.set_xlabel('Winter Olympic Game', size=14, color="#0D47A1")
ax.set_ylabel('Number of Sports', size=14, color="#0D47A1")
ax.set_title('Sports in each Winter Olympic game', size=18, color="#0D47A1")

plt.show()
winter_event = athlete[athlete['Season']=='Winter'].groupby('Games')['Event'].nunique().reset_index()

fig, ax = plt.subplots(figsize=(22,6))
a = sns.barplot(x='Games', y='Event', data=winter_event, ax=ax, color="lightskyblue")
a.set_xticklabels(labels=winter_event['Games'],rotation=90)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

ax.set_xlabel('Winter Olympic Game', size=14, color="#0D47A1")
ax.set_ylabel('Number of Events', size=14, color="#0D47A1")
ax.set_title('Events in each Winter Olympic game', size=18, color="#0D47A1")

plt.show()
games_gender = athlete[athlete['Season']=='Winter'].pivot_table(athlete, index=['Year','Games'], columns='Sex', aggfunc=len).reset_index()[['Year','Games','Age']]
games_gender.columns = ['Year','Games','F','M']
games_gender.fillna(0,inplace=True)
games_gender['gender_ratio'] = games_gender['F'] / (games_gender['F'] + games_gender['M'])

fig, ax = plt.subplots(figsize=(8,6))
a = sns.scatterplot(x="M", y="F", hue="gender_ratio", palette='RdYlGn', data=games_gender, ax=ax)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']-75, point['y']+50, '{0:.0f}'.format(point['val']))

label_point(games_gender['M'],games_gender['F'],games_gender['Year'],ax)
        
ax.set_xlabel('Male Athletes', size=14, color="#0D47A1")
ax.set_ylabel('Female Athletes', size=14, color="#0D47A1")
ax.set_title('Male vs Female Athletes in Winter Olympic games', size=18, color="#0D47A1")

plt.show()
fig, ax = plt.subplots(figsize=(20,6))
a = sns.boxplot(x="Year", y="Age", hue="Sex", palette={"M": "#B2EBF2", "F":"#F8BBD0"}, data=athlete[athlete['Season']=='Winter'], ax=ax)
        
ax.set_xlabel('Winter Game Year', size=14, color="#0D47A1")
ax.set_ylabel('Age', size=14, color="#0D47A1")
ax.set_title('Age distribution in Winter Olympic games', size=18, color="#0D47A1")
plt.show()
year_sport_age = athlete[athlete['Season']=='Winter'].pivot_table(athlete, index=['Year','Medal'], aggfunc=np.mean).reset_index()[['Year','Medal','Age']]
year_sport_age = year_sport_age.pivot("Medal", "Year", "Age")
year_sport_age = year_sport_age.reindex(["Gold","Silver","Bronze","No Medal"])
f, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(year_sport_age, annot=True, linewidths=0.05, ax=ax, cmap="RdYlGn_r")
ax.set_xlabel('Winter Game Year', size=14, color="#0D47A1")
ax.set_ylabel('Medal', size=14, color="#0D47A1")
ax.set_title('[Heatmap] Mean Age of Medal Winners in Winter Olympic games', size=18, color="#0D47A1")
plt.show()
#year_sport_age
t10_winter = athlete[(athlete['Season']=='Winter') & (athlete['Medal']!='No Medal')].groupby('region').count().reset_index()[['region','Medal']].sort_values('Medal', ascending=False).head(10)
f, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x="Medal", y="region", data=t10_winter, label="region", color="lightskyblue")

for p in ax.patches:
    ax.text(p.get_width() + 15,
            p.get_y() + (p.get_height()/2) + .1,
            '{:1.0f}'.format(p.get_width()),
            ha="center")

ax.set_xlabel('Region', size=14, color="#0D47A1")
ax.set_ylabel('Total Medals', size=14, color="#0D47A1")
ax.set_title('[Horizontal Bar Plot] Top 10 countries with total medals in Winter Olympic games', size=18, color="#0D47A1")
plt.show()
t3_winter = athlete[(athlete['Season']=='Winter') & (athlete['region'].isin(['Russia'])) & (athlete['Medal']!='No Medal')]
t3_winter = pd.pivot_table(t3_winter, index=['Sport'], columns=['Year'], values=['ID'],  aggfunc=len, fill_value=0)
t3_winter = t3_winter.reindex(t3_winter['ID'].sort_values(by=2014, ascending=False).index)

f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(t3_winter, annot=True, linewidths=0.05, ax=ax, cmap="RdYlGn")
ax.set_xlabel('Winter Game Year', size=14, color="green")
ax.set_ylabel('Sports', size=14, color="green")
ax.set_title('[Heatmap] Russia Medals in Sports across Years', size=18, color="green")
plt.show()