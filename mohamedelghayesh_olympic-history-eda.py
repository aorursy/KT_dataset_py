# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output 
# when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_athletes = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
df_regions = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
df_athletes.head()
df_regions.tail(10)
athletes = df_athletes.merge(df_regions, how='left', on='NOC')
athletes.head()
# keep column naming consistent
athletes.rename(columns={'region':'Region', 'notes':'Notes'}, inplace=True);
athletes.info()
athletes.query('Team != Region').Region.value_counts()[:10] # they are not the same
athletes.query('Team == "Russia-1"').head(1)
athletes.query('Team == "Denmark/Sweden"').head(1)
athletes[athletes['Notes'].isna() == False].head()
# what are the values in the notes column
notes_counts = athletes[athletes['Notes'].isna() == False].Notes.value_counts()
print(notes_counts.shape)
notes_counts
# looks like some athletes were of "UnKnown" notes areas
region_counts = athletes[athletes['Notes'].isna() == False].Region.value_counts()
print(region_counts.shape)
region_counts
# looks like the region has different values than the notes, 6 notes_counts vanished, new region_counts added
# but what about the Unknown athletes in the Notes
athletes.query('NOC == "UNK"') # unknown architecture competetors
from pandas.api.types import CategoricalDtype as CDtype
medal_order = ['Bronze','Silver','Gold']
medal_type = CDtype(categories=medal_order,ordered=True)
athletes['Medal'] = athletes['Medal'].astype(medal_type)

athletes.Medal.unique()
athletes['Age_g'] = pd.cut(athletes['Age'],bins=[0,15,30,45,100], \
                           labels=['0-15','15-30','30-45','45-100'], \
                           include_lowest=True)
athletes.head(1)
age_order = ['0-15','15-30','30-45','45-100']
age_g_type = CDtype(categories = age_order, ordered = True)
athletes['Age_g'] = athletes['Age_g'].astype(age_g_type)

athletes.Age_g.unique()
athletes['W_H_Ratio'] = athletes.Weight / athletes.Height
athletes.head(1)
athletes[athletes.Season == 'Winter'].Event.unique()  
# skiing, skating, snowboarding, sprints, sky jumps, Biathlon
athletes[athletes.Season == 'Summer'].Event.unique()  # Everything else
winter_sports = athletes[athletes.Season == 'Winter'].Sport.unique() 
winter_sports
summer_sports = athletes[athletes.Season == 'Summer'].Sport.unique()  
summer_sports
athletes.Weight.describe().loc[['min','max']]
athletes.Weight.mean(), athletes.Weight.std()
athletes.Weight.mean(), athletes.Weight.std()
plt.figure(figsize=(10,6))
bins = np.arange(25,214+2,2)
ax = sb.distplot(athletes['Weight'], kde=False, bins=bins)
ax.set_xlim(25,120)
plt.axvline(x=70.7, c='b')
plt.axvline(x=athletes.Weight.median(), c='c')
plt.axvline(x=athletes.Weight.mode()[0], c='r');
athletes.Height.describe().loc[['min','max']]
athletes.Height.mean(), athletes.Height.std()
plt.figure(figsize=(10,6))
bins = np.arange(127,226+3,3)
ax = sb.distplot(athletes['Height'], kde=False, bins=bins)
ax.set_xlim(140,210)
plt.axvline(x=175.34, c = 'b')
plt.axvline(x=athletes.Height.median(),c='c')
plt.axvline(x=athletes.Height.mode()[0],c='r');
athletes.Age.describe().loc[['min','max']]
athletes.Age.mean(), athletes.Age.std()
athletes.Age.mode()[0], athletes.Age.median()
# Wow, ages from 10 to 97
bins = np.arange(10,97+1,1)
ax = sb.distplot(athletes['Age'], kde=False, bins=bins)
ax.set_xlim(10,97)
plt.axvline(x=25.557, c='b')
plt.figure(figsize=(10,6))
bins = 10**np.arange(1,2+0.01,0.01) 
ticks = [10,15,20,25,30,40,50,60,70,100]

ax = sb.distplot(athletes['Age'], kde=False,bins=bins, color='blue')
ax.set_xscale('log') 
ax.set_xticks(ticks) 
ax.set_xticklabels(ticks) 
ax.set_xlim(12,70)
plt.axvline(x=athletes.Age.mean(), c='black')
plt.axvline(x=athletes.Age.median(), c='m')
plt.axvline(x=athletes.Age.mode()[0], c='r');
athletes.W_H_Ratio.plot(kind='hist',bins=80);
# plot from pandas 
bins = 10 ** np.arange(np.log10(athletes.W_H_Ratio.min()), np.log10(athletes.W_H_Ratio.max()) + 0.01, 0.01)
ticks = [0.2, 0.3, 0.4, 0.6]
ax = athletes.W_H_Ratio.plot(kind='hist', bins=bins, logx=True, xticks=ticks, xlim=(0.2,0.8), figsize=(10,6))
ax.set_xticklabels(ticks);
participants = athletes.Name.value_counts()
top_partic_athletes = participants[:15]
top_partic_athletes
plt.figure(figsize=(10,6))
ax = sb.barplot(y=top_partic_athletes.index, x=top_partic_athletes, palette='Blues_r')
ax.set_title('Overall Athlete Participation')
ax.set_xlabel('Frequency');
athletes[athletes.Name == 'Robert Tait McKenzie'].groupby(['Games','Event'])[['ID','Medal']].count()
athletes[athletes.Name == 'Robert Tait McKenzie'].Medal.value_counts()
athletes[athletes.Name == 'Heikki Ilmari Savolainen'].groupby(['Games','Event','Medal'])[['ID','Medal']].count()
athletes[athletes.Name == 'Heikki Ilmari Savolainen'].Medal.value_counts()
athletes[athletes.Name == 'Michael Fred Phelps, II'].groupby(['Games','Event','Medal'])[['ID','Medal']].count()
athletes[athletes.Name == 'Michael Fred Phelps, II'].Medal.value_counts()
# modern athletes participation
recent_athlete_part = athletes[athletes.Year >= 2006].Name.value_counts()[:15]
recent_athlete_part
plt.figure(figsize=(10,6))
ax = sb.barplot(y=recent_athlete_part.index, x=recent_athlete_part, palette='Blues_r')
ax.set_title('Athletes Participation 2006 - 2016')
ax.set_xlabel('Frequency');
athletes[(athletes.Name == 'Oleh Yuriyovych Verniaiev') & (athletes.Year >= 2010)] \
                                    .groupby(['Games','Event'])[['Medal']].count()
athletes[(athletes.Name == 'Marcel Van Minh Phuc Long Nguyen') & (athletes.Year >= 2010)] \
        .groupby(['Games','Event'])[['Medal']].count()
top_medal_athletes = athletes.groupby('Name').Medal.count().reset_index(name='Count') \
                                             .sort_values(by='Count', ascending=False).head(15)
top_medal_athletes
plt.figure(figsize=(10,6))
ax = sb.barplot(y=top_medal_athletes.Name, x=top_medal_athletes.Count, palette='Blues_r')
ax.set_title('Top Decorated Athletes')
ax.set_xlabel('Frequency');
event_f, event_m = athletes.groupby('Sex').Event.count()
event_f, event_m, event_f / (event_f + event_m)
athletes.Sex.unique()
male_medals = athletes.query('Sex == "M"').Medal.count()
female_medals = athletes.query('Sex == "F"').Medal.count()
male_medals, female_medals, male_medals / (male_medals + female_medals)
plt.figure(figsize=(10,6))
plt.bar(x=['M','F'], height=[male_medals,female_medals], width=0.4);
age_groups = athletes[athletes['Medal'].isna() == False].Age_g.value_counts().sort_index()

plt.figure(figsize=(10,6))
ticks = [100,200,500,1000,2000,5000,10000,20000,50000]

plt.bar(x=age_groups.index, height=age_groups)
plt.yscale('log')
plt.yticks(ticks,ticks);
top_partic = athletes.Team.value_counts().reset_index(name='Count')
top_partic.rename({'index':'Team'},axis=1,inplace=True)
top_partic_teams = top_partic.head(15)
top_partic_teams
plt.figure(figsize=(10,6))
ax = sb.barplot(data=top_partic_teams, y= 'Team', x= 'Count', palette='Blues_r')
ax.set_title('Top Participating Teams')
ax.set_xlabel('Frequency');
top_medal_teams = athletes.groupby('Team').Medal.count().reset_index(name='Count') \
                                          .sort_values(by='Count', ascending=False).head(15)
top_medal_teams
plt.figure(figsize=(10,6))
ax = sb.barplot(y=top_medal_teams.Team, x=top_medal_teams.Count, palette='Blues_r')
ax.set_title('Top Decorated Teams')
ax.set_xlabel('Frequency');
# Note: The index is calculated for the Top Participating/Decorated Teams 
# Note: using the inner join to drop the Teams that appear in one list only

top_teams_index = top_medal_teams.merge(top_partic_teams, how='inner', on='Team')
top_teams_index.rename(columns={'Count_x':'Medals','Count_y':'Paricipates'}, inplace=True)
top_teams_index['Decoration_Index'] = top_teams_index['Medals'] / top_teams_index['Paricipates']
decoration_index = top_teams_index.sort_values('Decoration_Index',ascending=False)
decoration_index
plt.figure(figsize=(10,6))
ax = sb.barplot(data=decoration_index, x= 'Decoration_Index', y='Team', palette='Blues_r')
ax.set_title('Decoration Index');
unique_athletes = athletes.Name.unique().shape[0]
decorated_athletes = athletes[athletes['Medal'].isna() == False]
non_decorated_athletes = athletes[athletes['Medal'].isna()]
unique_dec_ath = decorated_athletes.Name.unique().shape[0]

unique_athletes, unique_dec_ath, unique_dec_ath/unique_athletes
athletes.City.unique()
# using the year to fetch the host cities
unique_years = athletes.Year.unique()

host_cities = {}
for year in unique_years:
    host_cities[year] = athletes[athletes['Year'] == year].City.unique().tolist()
    
citylist = list(host_cities.values())
print(citylist)
# 2 olympic games can exist in one year 
# flatten to 1d list
city_flat_list = []
for cityline in citylist:
    for city in cityline:
        city_flat_list.append(city)

print(city_flat_list)
# using the geonamescache, map cities to countries
# https://pypi.org/project/geonamescache/
# https://livebook.manning.com/book/data-science-bookcamp/chapter-11/v-2/104
!pip3 install geonamescache 
import geonamescache

gc = geonamescache.GeonamesCache()
countries = gc.get_countries()
countries['US']['name']
#gc.search_cities('Cairo')[0]['timezone'].split('/')[0]
#alex = gc.search_cities('St. Louis')
#[alex[i]['countrycode'] for i in range(len(alex))] 
# this step maps the city name to the country name stored in the library, thus changing the historical name
cityinfo = [gc.search_cities(c) for c in city_flat_list]
isolist = [cityinfo[i][j]['countrycode'] for i in range(len(cityinfo)) for j in range(len(cityinfo[i]))]
#cityinfo[2][0]['countrycode']
print(isolist)
countrylist = [countries[iso]['name'] for iso in isolist]
np_countrylist = np.array(countrylist)
np_countrylist
hostcountries = pd.DataFrame(np_countrylist) # one column dataframe
hostcountries.columns = ['Country']

freq = hostcountries.Country.value_counts().sort_values(ascending=False)

plt.figure(figsize=(8,6))
sb.barplot(freq, freq.index, palette='Blues_r')
plt.title('Inaccurate Host Country Numbers')
# https://en.wikipedia.org/wiki/List_of_Olympic_Games_host_cities
# read the list of all cities, countries, continents and years 
# this step preserves the historical country name
olympic_hosts = pd.read_csv('../input/olympic-hosts/olympic_hosts.csv')
olympic_hosts.head(1)
# host countries 
country_freq = olympic_hosts.query('Year <= 2016').Country.value_counts()

plt.figure(figsize=(10,6))
sb.barplot(country_freq, country_freq.index, palette='Blues_r')
plt.title('Olympics Host Country');
# host continents
continents = olympic_hosts.query('Year <= 2016').Continent.value_counts().to_frame().reset_index()
continents.rename(columns={'index':'Continent', 'Continent':'Count'}, inplace=True)
continents
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
plt.figure(figsize=(10,6))
g=sb.barplot(y="Continent",x="Count",data=continents, palette='Blues_r')
for p in g.patches:
    #       count data to show   ,   (x_pos  ,  y_pos   )
    g.annotate(str(p.get_width()), (p.get_width()+0.01, p.get_y()+0.45),color='blue', fontweight='bold', fontsize=14)
plt.title('Olympics Host Continent');
 
plt.figure(figsize=[10,8])
sb.heatmap(athletes.corr(),annot=True,cmap='YlGnBu');
# weights and heights
plt.figure(figsize=(10,6))
plt.scatter(data = athletes, x='W_H_Ratio', y='Height', alpha=0.005)
plt.xscale('log')
plt.xlabel='Weight/Height Ratio' 
plt.ylabel='Frequency';
athletes.W_H_Ratio.mean(), athletes.query('Sex == "M"').W_H_Ratio.mean(), athletes.query('Sex == "F"').W_H_Ratio.mean()
g = sb.FacetGrid(data = athletes, col = 'Sex', height=5)
g.map(sb.distplot, 'W_H_Ratio', kde=False, bins= 100)
g.map(plt.axvline, x=0.4, c='b')
g.set(xlim = (0.2, 0.6), xlabel='Weight/Height Ratio', ylabel='Frequency')
plt.tight_layout()

#g.map(plt.grid)
decorated_m = athletes[(athletes['Medal'].isna() == False) & (athletes['Sex'] == "M")]
decorated_m.W_H_Ratio.mean()
decorated_f = athletes[(athletes['Medal'].isna() == False) & (athletes['Sex'] == "F")]
decorated_f.W_H_Ratio.mean()
non_decorated_m = athletes[(athletes['Medal'].isna()) & (athletes['Sex'] == "M")]
non_decorated_m.W_H_Ratio.mean()
non_decorated_f = athletes[(athletes['Medal'].isna()) & (athletes['Sex'] == "F")]
non_decorated_f.W_H_Ratio.mean()
g = sb.FacetGrid(data=athletes, col='Age_g', height=4, col_wrap=2)
g.map(sb.distplot, 'W_H_Ratio', kde=False, bins=100)
g.map(plt.axvline, x=0.4, c='b')
g.set(xlim = (0.1,1.0), xlabel='Weight/Height Ratio', ylabel='Frequency', yscale='log')

plt.tight_layout()
print('Below 15: ', athletes.query('Age_g == "0-15"').W_H_Ratio.mean())
print('15 to 30: ', athletes.query('Age_g == "15-30"').W_H_Ratio.mean())
print('30 to 45: ', athletes.query('Age_g == "30-45"').W_H_Ratio.mean())
print('Above 45: ', athletes.query('Age_g == "45-100"').W_H_Ratio.mean())
print('Below 15: ', athletes[(athletes['Medal'].isna() == False) & (athletes['Age_g'] == "0-15")].W_H_Ratio.mean())
print('15 to 30: ', athletes[(athletes['Medal'].isna() == False) & (athletes['Age_g'] == "15-30")].W_H_Ratio.mean())
print('30 to 45: ', athletes[(athletes['Medal'].isna() == False) & (athletes['Age_g'] == "30-45")].W_H_Ratio.mean())
print('Above 45: ', athletes[(athletes['Medal'].isna() == False) & (athletes['Age_g'] == "45-100")].W_H_Ratio.mean())
# for all athletes
ticks = [0.1,0.2,0.3,0.4,0.5,0.6,1.0]

plt.subplots(nrows=1, ncols=4, figsize=(16, 10))

plt.subplot(1,4,1)
g = sb.boxplot(data= non_decorated_athletes, x='Sex', y='W_H_Ratio')
g.set_title('Non Decorated Athletes')
g.set_yscale('log')
g.set_yticklabels(ticks)
g.set_yticks(ticks)
g.set_ylim(0.15, 1.2)

plt.subplot(1,4,2)
g = sb.violinplot(data= non_decorated_athletes, x='Sex', y='W_H_Ratio')
g.set_title('Non Decorated Athletes')
g.set_yscale('log')
g.set_yticklabels(ticks)
g.set_yticks(ticks)
g.set_ylim(0.15, 1.2)

# for decorated athetes
plt.subplot(1,4,3)
g = sb.boxplot(data=decorated_athletes, x='Sex', y='W_H_Ratio')
g.set_title('Decorated Athletes')
g.set_yscale('log')
g.set_yticklabels(ticks)
g.set_yticks(ticks)
g.set_ylim(0.15, 1.2)

plt.subplot(1,4,4)
g = sb.violinplot(data=decorated_athletes, x='Sex', y='W_H_Ratio')
g.set_title('Decorated Athletes')
g.set_yscale('log')
g.set_yticklabels(ticks)
g.set_yticks(ticks)
g.set_ylim(0.15, 1.2)

plt.tight_layout()
# for all athletes
plt.figure(figsize=[16,10])
g = sb.boxplot(data= non_decorated_athletes, x='Age_g', y='W_H_Ratio')
g.set_yscale('log')
g.set_yticklabels(ticks)
g.set_yticks(ticks)
plt.title('Non Decorated Athletes');
# for decorated athletes
plt.figure(figsize=[16,10])
sb.boxplot(data= decorated_athletes, x='Age_g', y='W_H_Ratio')
g.set_yscale('log')
g.set_yticklabels(ticks)
g.set_yticks(ticks)
plt.title('For Decorated Athletes');
athletes.head(1)
athletes.query('Age_g == "0-15"').groupby('Sex').ID.count()
athletes.query('Age_g == "0-15"').groupby('Sex').Medal.count()
team_medals = athletes.groupby(['Year','Team','Sport','Season'])['Medal'].value_counts() \
                                                                         .to_frame(name='Count') \
                                                                         .reset_index()
team_medals
us_medals = team_medals.query('Team == "United States"')[['Year','Count']].groupby('Year').Count.sum()
us_medals.head()
plt.figure(figsize=(16,6))
sb.lineplot(x=us_medals.index, y=us_medals)
plt.title('US Medals 1896 - 2016')
plt.xticks(us_medals.index, rotation=60);
g = sb.FacetGrid(data = athletes, col='Age_g', col_wrap=2, height=6)
g.map(sb.boxplot, 'Sex', 'W_H_Ratio', order=['F','M'])
plt.tight_layout();
g = sb.FacetGrid(data = athletes, col='Sex', col_wrap=2, height=6)
g.map(sb.boxplot, 'Age_g', 'W_H_Ratio',order=athletes.Age_g.unique().categories.tolist())
plt.tight_layout();
w_ticks = [0.1,0.2,0.3,0.4,0.6,0.7,0.8,1.0]
h_ticks = [130,140,150,160,170,180,190,200,210,220]
w_labels = ['{:.1f}'.format(t) for t in w_ticks]
h_labels = ['{:.1f}'.format(t) for t in h_ticks]

g = sb.FacetGrid(data = athletes, col='Age_g', col_wrap=4, height=8, aspect=0.6)
g.map(plt.scatter, 'W_H_Ratio', 'Height', alpha=0.05)
g.set(yscale='log', yticks=h_ticks, yticklabels=h_labels, ylim=(130,220), \
      xscale='log', xticks=w_ticks, xticklabels=w_labels, xlim=(0.15,1.0))

plt.xticks(w_ticks, w_ticks)
plt.yticks(h_ticks, h_ticks)
plt.tight_layout();
g = sb.FacetGrid(data = decorated_athletes, hue='Medal', col='Age_g', col_wrap=4, height=8, aspect=0.6)
g.map(plt.scatter, 'W_H_Ratio', 'Height', alpha=0.5)
g.set(yscale='log', ylim=(130,220), xscale='log', xlim=(0.15,1.0))

plt.xticks(w_ticks, w_ticks)
plt.yticks(h_ticks, h_ticks)
plt.legend()
plt.tight_layout();
g = sb.FacetGrid(data = decorated_athletes, hue='Medal', col='Sex', col_wrap=2, height=8, aspect=0.6)
g.map(plt.scatter, 'W_H_Ratio', 'Height', alpha=0.5)
g.set(yscale='log', ylim=(130,220), xscale='log', xlim=(0.2,1.0))

plt.xticks(w_ticks, w_ticks)
plt.yticks(h_ticks, h_ticks)
plt.legend()
plt.tight_layout();
g = sb.FacetGrid(data = decorated_athletes, col='Medal', col_wrap=3, height=5,col_order=['Gold','Silver','Bronze'])
g.map(sb.boxenplot, 'Age_g', 'W_H_Ratio', order=athletes.Age_g.unique().categories.tolist(), color='orange')
g.map(plt.axhline, y=decorated_athletes.W_H_Ratio.mean(), c='b')
g.map(plt.axhline, y=athletes.W_H_Ratio.mean(), c='b', linestyle='--');
g = sb.FacetGrid(data = decorated_athletes, col='Medal', col_wrap=3, height=5,col_order=['Gold','Silver','Bronze'])
g.map(sb.boxenplot, 'Sex', 'W_H_Ratio', order=['F','M'], color='orange')
g.map(plt.axhline, y=decorated_athletes.W_H_Ratio.mean(), c='b')
g.map(plt.axhline, y=athletes.W_H_Ratio.mean(), c='b', linestyle='--');
g = sb.FacetGrid(data = decorated_athletes, col='Medal', hue='Sex', col_wrap=3, height=5, \
                 col_order = ['Gold','Silver','Bronze'], palette = 'Greens_r')
g.map(sb.boxenplot, 'Age_g', 'W_H_Ratio', order=['0-15','15-30','30-45','45-100'])
g.map(plt.axhline, y=decorated_athletes.W_H_Ratio.mean(), c='b', alpha=0.3)
g.map(plt.axhline, y=athletes.W_H_Ratio.mean(), c='b', linestyle='--', alpha=0.3);
us_medals_season = team_medals.query('Team == "United States"')[['Year','Season','Count']].groupby(['Year','Season']) \
                                                                                          .Count.sum() \
                                                                                          .to_frame(name='Count') \
                                                                                          .reset_index()
us_medals_season.head()

g = sb.FacetGrid(data = us_medals_season, row = 'Season', height=4, aspect=3)
g.map(sb.lineplot, 'Year', 'Count')
plt.tight_layout()