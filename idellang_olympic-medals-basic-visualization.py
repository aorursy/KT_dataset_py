import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.csv', encoding='ISO-8859-1')
print(df.shape)
df.info()
df.head()
#rename columns into lower case
df.columns = df.columns.str.lower()
df.columns

df['year'].unique()
df['year'].value_counts(dropna = False) # there are 117 missing years
df[df['year'].isnull()].head() 
df.dropna(inplace = True)
print(df.shape)
df.info()
events = df.loc[:,'sport':'event']
events.head()
#here are all the sports of olympics. Most events are aquatics. 
events['sport'].value_counts()
events[events['sport'] == 'Aquatics'].loc[:,'discipline'].value_counts() 
#three types of aquatics discipline
events.groupby('sport')['discipline'].value_counts()
events.loc[events['sport'] == 'Athletics','event'].value_counts()
df['country'].unique().size

#there are 127 countries
#most number of countries in the dataset
df['country'].value_counts().head(10)
df['medal'].unique()
#only three types of medals
df['athlete'].value_counts().head(10)
#michael phelps is the most common athlete in the dataset
df.head()
def get_top(df, col = 'event', n = 5):
    return df.sort_values(by = col)[-n:]
medal_country = df.groupby(['medal','country'], as_index = False)['event'].count()
medal_group = medal_country.groupby('medal', as_index = False).apply(get_top).sort_values(by = ['medal','event'], ascending = False)
medal_group = medal_group.reset_index().drop(['level_0', 'level_1'], axis = 1)
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot()
sns.barplot(x = 'medal', y ='event', hue = 'country', data = medal_group, palette = 'Spectral', ax = ax, order = ['Gold','Silver','Bronze'])
plt.title('Top 5 countries by medal')
usa = df[df['country'] == 'United States']
usa.head()
plt.style.use('ggplot')
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot()
top5_usa_sport = usa.groupby('sport')['medal'].count().sort_values(ascending = False)
top5_usa_sport.head(10).plot(kind = 'bar', ax = ax)
ax.set_title('Top 10 highest medal earning sport in USA')
country_medal_count = df.groupby(['year','country'])['medal'].count().reset_index()
country_medal_count.sort_values(by = 'year')
fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot()
sns.lineplot( x = 'year', y = 'medal', hue = 'country', data = country_medal_count[country_medal_count['medal'] > 50])
plt.legend(ncol = 3, frameon = False, title = '')
plt.title('Medal count trend of countries with medals greater than 50')
c1 = (df['sport'] == 'Basketball')
c2 = (df['gender'] == 'Men')
basketball = df[c1&c2]
print(basketball.shape)
basketball.head()
basketball_group = basketball.groupby(['year','country','medal'], as_index = False)['event'].count()
order = ['Gold','Silver','Bronze']
g = sns.FacetGrid(data = basketball_group, col = 'medal', height = 6, sharex = False, col_order = order)
g.map(sns.countplot, 'country' )
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Basketball winnings by medal', fontsize = 15)
won_by_us = df.loc[df['country'] == 'United States', 'discipline'].values
plt.figure(figsize = (10,6))
df.loc[~df['discipline'].isin(won_by_us), 'discipline'].value_counts().plot(kind = 'bar')
plt.title('Disciplines not won by US Ever')
plt.figure(figsize = (10,6))
df['gender'].value_counts().plot(kind = 'pie', autopct='%1.0f%%', pctdistance=.5, labeldistance=1.1)
plt.title('distribution of men and women in summer olympics');
plt.figure(figsize = (10,6))
df.loc[:,['event','gender']].drop_duplicates().loc[:,'gender'].value_counts().plot(kind = 'pie', autopct = '%1.0f%%', pctdistance = .5)
plt.title('distribution of men and women on unique events')
order = ['Men','Women']
plt.figure(figsize = (10,6))
sns.countplot(x = 'year', data = df, hue = 'gender', hue_order = order)
plt.title('count of male vs female participants across years');
athlete_group = df.groupby(['year','athlete','sport'])['event'].count()
more_than_4 = athlete_group[athlete_group > 4]
more_than_4 = more_than_4.reset_index()
more_than_4.head()
g = sns.FacetGrid(data = more_than_4, col = 'sport', height = 7, sharex = False)
g.map(sns.barplot, 'athlete', 'event', ci = False)
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Athletes with more than 4 events', fontsize = 15)
