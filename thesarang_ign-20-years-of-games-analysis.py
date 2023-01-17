%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.axes_style("darkgrid")
import warnings 
warnings.filterwarnings('ignore')
import calendar
df = pd.read_csv('../input/ign.csv')
df.drop(labels= 'Unnamed: 0', axis=1, inplace=True) #droping unwanted 'Unnamed: 0' column.
df.head() 
df.shape #data has 18625 rows with 10 features
df.info() #Information about the columns' names, datatypes and null values
df.isnull().sum() #counting all the null values in columns
score_phrase = df.score_phrase.value_counts().reset_index()
score_phrase.rename(columns={'index':'review','score_phrase':'count'}, inplace=True)
score_phrase['percent'] = (score_phrase['count']/18625)*100
plt.figure(figsize=(15,6))
sns.barplot(x=score_phrase['review'], y=score_phrase['count'], data=score_phrase)
release_year = df.release_year.value_counts().reset_index()
release_year.rename(columns={'index':'year', 'release_year':'count'}, inplace=True)
release_year.sort_values(by='year', ascending=True, inplace=True)
plt.figure(figsize=(15,5))
sns.barplot(x='year', y='count', data=release_year) 
# Observations - Most games are released in year 2008 followed by 2009.
release_month = df.release_month.value_counts().reset_index()
release_month.rename(columns={'index':'release_months','release_month':'count'}, inplace=True)
release_month.sort_values(by='release_months', ascending=True, inplace=True)
release_month.release_months = release_month.release_months.apply(lambda x: calendar.month_abbr[x])
plt.figure(figsize=(15,6))
sns.barplot(x='release_months', y='count', data=release_month)
# Observations - Most games are released in November followed by October.
release_day = df.release_day.value_counts().reset_index()
release_day.rename(columns={'index': 'day', 'release_day': 'count'}, inplace=True)
release_day.sort_values(by='day', ascending=True, inplace=True)
plt.figure(figsize=(15,5))
sns.barplot(x='day', y='count', data=release_day)
# Observations - Most games are released approximately in mid-months particularly on 14th followed by 18th day
platform = df.platform.value_counts().reset_index()
platform.rename(columns={'index':'platform', 'platform':'count'}, inplace=True)
platform.sort_values(by='count', ascending=False, inplace=True)
plt.figure(figsize=(13,4))
platform[0:10].plot(x='platform', y='count', kind='barh', figsize=(15,6))
plt.show()
plt.figure(figsize=(18,11))
length_of_rows = 2
length_of_columns = 3
years = [2011, 2012, 2013, 2014, 2015, 2016]
for i in range(len(years)):
    plt.subplot(length_of_rows,length_of_columns,i+1)
    plats = df[df.release_year == years[i]]
    plats = plats.platform.value_counts().reset_index()
    plats.rename(columns={'index':'platform', 'platform':'count'}, inplace=True)
    plats = plats[plats['count'] > 15]
    plt.pie(x=plats['count'], labels=plats['platform'], autopct='%.0f%%', shadow=True)
    plt.xlabel(years[i])
print (df.score_phrase.unique())
plt.figure(figsize=(18,23))
plt.tight_layout()
length_of_rows = 4
length_of_columns = 3
reviews = ['Amazing', 'Great', 'Good', 'Awful', 'Okay', 'Mediocre', 'Bad', 'Painful', 'Unbearable', 'Disaster', 'Masterpiece']
for i in range(len(reviews)):
    grp_scorep_plat = df.groupby(['score_phrase','platform']).size().reset_index()
    grp_scorep_plat.rename(columns={'score_phrase':'review', 0:'count'}, inplace=True)
    grp_scorep_plat = grp_scorep_plat[(grp_scorep_plat['review'] == reviews[i])]
    grp_scorep_plat = grp_scorep_plat.sort_values(by='count', ascending=False)[0:10]
    plt.subplot(length_of_rows,length_of_columns,i+1)
    plt.pie(x=grp_scorep_plat['count'], labels=grp_scorep_plat['platform'], autopct='%0.1f%%', shadow=True)
    plt.xlabel(reviews[i])
df.genre.unique()
df.platform.value_counts()[0:20]
plt.figure(figsize=(19,12))
length_of_rows = 2
length_of_columns = 3
platform_genres = ['PC', 'PlayStation 2', 'Xbox 360', 'Wii', 'PlayStation 3', 'Nintendo DS']
for i in range(len(platform_genres)):
    genre_plat = df.groupby(['platform','genre']).size().reset_index()
    genre_plat.rename(columns={0:'count'}, inplace=True)
    genre_plat = genre_plat[genre_plat['platform'] == platform_genres[i]].sort_values(by='count', ascending=False)[:10]
    plt.subplot(length_of_rows, length_of_columns, i+1)
    sns.pointplot(y=genre_plat['genre'], x=genre_plat['count'])
    plt.xlabel(platform_genres[i])
plt.show()
plt.figure(figsize=(19,21))
length_of_rows = 4
length_of_columns = 3
reviews = ['Amazing', 'Awful', 'Bad', 'Disaster', 'Good', 'Great', 'Masterpiece', 'Mediocre', 'Okay', 'Painful', 'Unbearable']
for i in range(len(reviews)):
    gp = df.groupby(['score_phrase','genre']).size().reset_index().sort_values(by=0, ascending=False).rename(columns={0:'count'})
    gp = gp[gp['score_phrase'] == reviews[i]][:5]
    plt.subplot(length_of_rows, length_of_columns, i+1)
    sns.pointplot(x=gp['genre'], y=gp['count'])
#     sns.tsplot(gp,) 
    plt.xlabel(reviews[i])
plt.show()
