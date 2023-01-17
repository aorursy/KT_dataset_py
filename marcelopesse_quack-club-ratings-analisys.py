import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # beautiful graphs
import matplotlib.pyplot as plt #math stuf
import datetime as dt #for date math

from kaggle_secrets import UserSecretsClient # to hide my API key
user_secrets = UserSecretsClient() # to hide my API key

#Let's import the dataset
df = pd.read_csv('/kaggle/input/quack-club-score-original/quack_club_score.csv')
print(df.describe)
print('\n**************************************************************************\n')
print(df.describe(include='all'))
# Lowercase content
df = df.applymap(lambda s:s.lower() if isinstance(s, str) else s)

# Lowercase columns
df.columns = map(str.lower, df.columns)

# Remove arara zero
df = df.replace(0,np.NaN)

# Remove '?' char
df = df.replace('?',np.NaN)

# Add picks
df.loc[df['ep']==209,'pick'] = 'madz'
df.loc[df['ep']==210,'pick'] = 'storm'
df.loc[df['ep']==211,'pick'] = 'cosmos'
# Fix Cosmos scores
for ep in range(189,198):
    df.loc[df['ep']==ep,'cosmos'] = df.loc[df['ep']==ep,'partner']
    df.loc[df['ep']==ep,'partner'] = np.nan
    df.loc[df['ep']==ep,'partner'] = df.loc[df['ep']==ep,'partner 2']
    df.loc[df['ep']==ep,'partner 2'] = np.nan
    
for ep in range(198,212):
    df.loc[df['ep']==ep,'cosmos'] = df.loc[df['ep']==ep,'rune']
    df.loc[df['ep']==ep,'rune'] = np.nan
dates = {
    198:'03/01/20',
    199:'10/01/20',
    200:'17/01/20',
    201:'24/01/20',
    202:'31/01/20',
    203:'07/02/20',
    204:'14/02/20',
    205:'21/02/20',
    206:'28/02/20',
    207:'06/03/20',
    208:'13/03/20',
    209:'20/03/20',
    210:'27/03/20',
    211:'03/04/20'
}

for ep in dates:
    df.loc[df['ep']==ep,'date'] = dates[ep]
df = df[:-1]
import requests

def get_metacrit(series):
    parameters = {
    'platform': 'pc'
    }
    
    #my_key = user_secrets.get_secret("rapidapi_key")
    
    headers = {
    'x-rapidapi-host': 'chicken-coop.p.rapidapi.com',
    'x-rapidapi-key': '5fc0af0c7amsh9ceb6a2d4914fabp15856fjsn75748452ca35'
    }

    path = 'https://chicken-coop.p.rapidapi.com/games/'
    
    name = str(series['title'])
    response = requests.get(path + name,headers=headers, params=parameters)
    game_info = response.json()
    
    if (game_info['result'] == 'No result'):
        series['error'] = 1
    else:
        game_info = game_info['result']
        series['error'] = 0
        series['title'] = game_info['title']
        series['mc'] = game_info['score']
        series['release_date'] = game_info['releaseDate']
        series['developer'] = game_info['developer']
        series['rating'] = game_info['rating']
        series['genre'] = game_info['genre']
        series['publisher'] = game_info['publisher']
        series['also_available'] = game_info['alsoAvailableOn']
    return series
            
df = df.apply(get_metacrit, axis=1)
print(df['error'].sum())
print(df[df['error']==1][['ep','title']])
correct_title={
    45:'odallus the dark call',
    53:np.nan,
    66:'abzu',
    71:'valkyrie drive bhikkhuni',
    107:np.nan,
    115:np.nan,
    154:'ace combat 7 skies unknown',
    157:'toejam earl back in the groove',
    163:'peggle deluxe',
    177:'outer wilds',
    189:'underrail',
    192:'yooka laylee and the impossible lair',
    198:'team fortress 2',
    199:'jamestown legend of the lost colony',
    202:'halo reach remastered',
    204:'nioh complete edition'
}

for ep in correct_title:
    df.loc[df['ep']==ep,'title'] = correct_title[ep]
df = df.dropna(subset=['title'], axis='index')

df[df['error']==1] = df[df['error']==1].apply(get_metacrit, axis=1)
print('date column')
print(type(df['date'][1]))
print(df['date'][1])

print('\nrelease date column')
print(type(df['release_date'][1]))
print(df['release_date'][1])
df['date'] = pd.to_datetime(df['date'],dayfirst=True,format='%d/%m/%y')
correct_release_date = {
    80:'Aug 24, 2017',
    162:'Feb 25, 2016',
    205:'Dec 13, 2019',
    209:'Feb 26, 2020'
}

for ep in correct_release_date:
    df.loc[df['ep']==ep,'release_date'] = correct_release_date[ep]

df['release_date'] = pd.to_datetime(df['release_date'],format='%b %d, %Y')
df['mc'] = df['mc'].astype(float)/10.0
df_score = pd.melt(df, value_vars=['mc', 'arara', 'madz','storm','rune','cosmos','partner'])

sns.set_style('darkgrid')
sns.set(font_scale=2)
plt.figure(figsize=(20, 8))

ax = sns.boxplot(x='variable',y='value',orient='v',data=df_score,fliersize=8)
#ax = sns.swarmplot(x='variable', y='value', orient='v',data=df_score, size=3, color=".3", linewidth=0)

ax.set(xlabel='', ylabel='score')
df[['mc', 'arara', 'madz','storm','rune','cosmos','partner']].describe()
df['quack_mean'] = df[['arara', 'madz','storm','rune','cosmos','partner','partner 2']].mean(skipna=True, axis='columns')

df_nozero = df[df['mc'] != 0]

print('Pearson correlation coefficient: ')
print(df_nozero['quack_mean'].corr(df_nozero['mc']))

sns.set_style('darkgrid')
sns.set(font_scale=2)
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x='quack_mean', y='mc', data=df_nozero, s=150)

plt.plot([0, 10], [0, 10], linewidth=2, alpha=.5, color='green')

m, b = np.polyfit(df_nozero['quack_mean'], df_nozero['mc'], 1)
plt.plot(df_nozero['quack_mean'],b+m*df_nozero['quack_mean'], linewidth=2, alpha=.5, color='red')

ax.set(xlabel='Quack', ylabel='Metacritic',xlim=(-0.5,10.5),ylim=(-0.5,10.5))
df_nozero[df_nozero['quack_mean'] < 4][['title','mc','quack_mean']].sort_values(by=['mc'],ascending=False)
df[df['release_date'] == df['release_date'].min()][['title','release_date','quack_mean','mc']]
df[df['release_date'] == df['release_date'].max()][['title','release_date','quack_mean','mc']]
#We removed some rows while cleaning, so lets reset the index
df.reset_index(inplace=True)

print('Delta: ' + str((df['date'] - df['release_date']).max().days) + ' days\n')

date_max = (df['date'] - df['release_date']).idxmax()
print(df.iloc[[date_max]][['title','release_date','date','quack_mean','mc']])
print('Delta: ' + str((df['date'] - df['release_date']).min().days) + ' days\n')

date_min = (df['date'] - df['release_date']).idxmin()
print(df.iloc[[date_min]][['title','release_date','date','quack_mean','mc']])
df['delta_date'] = (df['date'] - df['release_date']).dt.days

plt.figure(figsize=(10, 10))
ax = sns.distplot(df['delta_date'], bins=20, kde=False, norm_hist=False)
ax.set(xlabel='Days', ylabel='Frequency')
plt.figure(figsize=(10, 10))
ax = sns.distplot(df[df['delta_date']<=1000.0]['delta_date'], bins=20, kde=False, norm_hist=False)
ax.set(xlabel='Days', ylabel='Frequency')
print('Games reviewed before one year delta: ' + str(df[df['delta_date']<=365.0]['delta_date'].count()))
print('Games reviewed after one year delta: ' + str(df[df['delta_date']>365.0]['delta_date'].count()))
import operator 
from functools import reduce


genre_list = [item for sublist in df['genre'].dropna() for item in sublist]
pd.Series(genre_list).value_counts(normalize=False).head(10)
from scipy import stats

host_list=['arara', 'madz','storm','rune','cosmos']

zs = pd.DataFrame(stats.zscore(df[host_list], nan_policy='omit', axis=0), columns=host_list)
zs['genre'] = df['genre']
zs
for host in host_list:
    print(host+' z-score mean: '+'%.4f' % (zs[host].abs().sum()/len(zs[host].dropna())))
zs_genre = pd.DataFrame(columns=host_list, index=set(genre_list))

zs_explode = zs.explode(column='genre')
for genre in set(genre_list):
    for host in host_list:
        #zs_genre[host][genre]=zs_explode[zs_explode['genre']==genre][host].dropna().abs().sum()
        zs_genre[host][genre]=zs_explode[zs_explode['genre']==genre][host].dropna().sum()
        if len(zs_explode[zs_explode['genre']==genre][host].dropna()) != 0:
            zs_genre[host][genre]=zs_genre[host][genre]/len(zs_explode[zs_explode['genre']==genre][host].dropna())
zs_genre
for host in host_list:
    g_idxmax = zs_genre[host].astype(float).idxmax(skipna=True)
    g_max = zs_genre[host].astype(float).max()
    g_idxmin = zs_genre[host].astype(float).idxmin(skipna=True)
    g_min = zs_genre[host].astype(float).min()
    print(host+' z-score result:')
    print('MAX:  '+'%.4f' % g_max+' '+g_idxmax)
    print('MIN: '+'%.4f' % g_min+' '+g_idxmin+'\n')
    
df_explode = df.explode(column='genre')

df_explode[df_explode['genre']=='Golf'][['title','arara', 'madz','storm','rune','cosmos']]
top_genre_list = df_explode['genre'].value_counts()[df_explode['genre'].value_counts() > len(df['ep'])*0.1]
top_zs_genre = zs_genre.loc[top_genre_list.index].astype(float)

top_zs_genre

f, ax = plt.subplots(figsize=(8, 8))

#sns.heatmap(top_zs_genre, linewidths=4, cmap="YlGnBu", vmin=-0.5, vmax=0.5, annot=True, fmt='.2f')
sns.heatmap(top_zs_genre, linewidths=4, cmap="YlGnBu", vmin=-0.5, vmax=0.5)
df.to_csv('/kaggle/working/quack_club_score_v2.csv',index = False)