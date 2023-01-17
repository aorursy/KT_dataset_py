#importing all the modules we will need
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import calendar
from IPython.display import Image
mpl.rcParams['figure.figsize'] = 15.0, 8
mpl.rcParams['font.size'] = 14
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
df = pd.read_csv('../input/nba-player-of-the-week/NBA_player_of_the_week.csv', index_col=0)
df = df.reset_index() #to get Age as a feature and not an index
df['id'] = df.index
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
df['n_awards'] = 1 #every line corresponds to one award. this is will be useful later for groupby methods
df.info()
df['date'] = df['date'].str.replace(', ', '-')
for k,v in enumerate(calendar.month_abbr): #the calendar module is useful to translate months as string to months as integers
    if(k!=0): #the first row is empty
        df['date'] = df['date'].str.replace(v + ' ', str(k)+'-')

df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y') #converting the data to the format that will be the most convenient
df['franchise'] = df['team'].apply(lambda x:x.split(' ')[-1]) #extracting the franchise only from the "city - franchise"

df.loc[df.franchise=='Bullets', 'franchise'] = 'Wizards'
df.loc[df.franchise=='SuperSonics', 'franchise'] = 'Thunder'
Image('../input/cha-nola/CHA_NOLA.png')
df.loc[(df.franchise=='Hornets') & (df.season_short>=2003) & (df.season_short<=2013), 'franchise'] = 'Pelicans' 
df.loc[df.franchise=='Bobcats', 'franchise'] = 'Hornets'
#extracting the teams' conferences during the last season
team_conferences = df.sort_values('season_short').drop_duplicates('franchise', keep='last')[['franchise', 'conference']]
team_conferences = team_conferences.rename(columns={'conference':'conf_in_2018'})

df = pd.merge(df, team_conferences, on='franchise') #attributing the conf_in_2018 to the original dataframe

df['conf_x'] = np.where(df.conf_in_2018=='West', 1, 2) #for later plot use
awards_by_player = df.groupby('player').agg({'n_awards':'count'}).reset_index() #grouping by players and converting the series to dataframe with reset_index(). As 1 line = 1 award, using 'count' or 'sum' would give the same results
awards_by_player.sort_values('n_awards', ascending=False).reset_index(drop=True)[:20]
#getting a dataframe with every team a player won awards with
players_best_teams = df.groupby(['player', 'franchise'])['n_awards'].count().reset_index() 

#sorting the dataframe by putting the teams with most awards at the end, and dropping the first teams if there's different teams for a given player
players_best_teams = players_best_teams.sort_values('n_awards').drop_duplicates('player', keep='last')
players_best_teams = players_best_teams.rename(columns={'franchise':'best_franchise'}) #being explicit

#attributing the best teams to our original ranking
dff1 = pd.merge(awards_by_player, players_best_teams[['player', 'best_franchise']], on='player')
#putting the data in a list of list
team_colors = [ 
['#E03A3E', '#C1D32F', 'Hawks'],
['#007a33', '#BA9653', 'Celtics'],
['#000000', '#000000', 'Nets'],
['#1D1160', '#00788c', 'Hornets'],
['#CE1141', '#000000', 'Bulls'],
['#6f263d', '#ffb81c', 'Cavaliers'],
['#00538C', '#B8C4CA', 'Mavericks'],
['#00285E', '#ffffff', 'Nuggets'],
['#ED174C', '#006BB6', 'Pistons'],
['#006BB6', '#fdb927', 'Warriors'],
['#ce1141', '#000000', 'Rockets'],
['#002D62', '#fdbb30', 'Pacers'],
['#ED174C', '#006bb6', 'Clippers'],
['#552583', '#ffc72c', 'Lakers'],
['#6189B9', '#00285E', 'Grizzlies'],
['#98002e', '#F9A01B', 'Heat'],
['#0c2340', '#236192', 'Timberwolves'],
['#002b5c', '#E31837', 'Pelicans'],
['#006BB6', '#f58426', 'Knicks'],
['#007AC1', '#ef3b24', 'Thunder'],
['#0057b8', '#c2ccd2', 'Magic'],
['#006BB6', '#ED174C', 'Sixers'],
['#1D1160', '#e56020', 'Suns'],
['#E03A3E', '#000000', 'Blazers'],
['#5A2D81', '#63727a', 'Kings'],
['#000000', '#C4CED4', 'Spurs'],
['#CE1141', '#000000', 'Raptors'],
['#002B5C', '#F9A01B', 'Jazz'],
['#002B5C', '#e31837', 'Wizards'],
['#00471b', '#eee1c6', 'Bucks']
]

#converting the list of list to a pandas dataframe
df_colors = pd.DataFrame(team_colors, columns=['primary_color', 'secondary_color', 'best_franchise'])

#attributing the right colors to rows of our best players ranking
dff = pd.merge(dff1, df_colors, on='best_franchise')
dff = dff.sort_values('n_awards', ascending=False)
fig, ax = plt.subplots(2, 1, figsize=(16, 10)) #one figure with two axes
dff['player'] = dff['player'].str.replace(' ', '\n') #replacing spaces by line breaks for better visualization on xticks

#dividing the dataframe for the two rows
dff_row1 = dff[:9]
dff_row2 = dff[10:19]

#barplotting with seaborn and attibuting the primary color for filling, the secondary color for border
sns.barplot('player', 'n_awards', data=dff_row1, ax=ax[0], palette=dff_row1.primary_color.tolist(), edgecolor = dff_row1.secondary_color.tolist(), linewidth=4)
sns.barplot('player', 'n_awards', data=dff_row2, ax=ax[1], palette=dff_row2.primary_color.tolist(), edgecolor = dff_row2.secondary_color.tolist(), linewidth=4)

for a in ax:
    a.set_xlabel('') #keeping it sober
    a.set_ylim(0, 65) #giving the same scale to both axes to keep in mind how Lebron also crushes the 11th to 20th
#getting number of awards by player, without forgetting the latter year the player got its award
dfg = df.groupby(['franchise', 'player']).agg({'n_awards':'count', 'season_short':'max'}).reset_index()

#if duplicates in team, keep player who got the last award
dfg = dfg.sort_values(['franchise', 'n_awards', 'season_short']).drop_duplicates('franchise', keep='last')
dfg = dfg.sort_values('n_awards', ascending=False).reset_index(drop=True)
dfg
dft = df[df.player=='LeBron James']
dft.groupby('franchise')['n_awards'].sum()
dff = df[df.season_short>2002]
#getting the number of awards by team, and the integer version of the conference which we computed earlier
dfg = dff.groupby('franchise').agg({'n_awards':'count', 'conf_x':'max'}).reset_index()

#in order to use the .shift() function later, we need to sort the dataframe based on number of awards
dfg = dfg.sort_values(['conf_x', 'n_awards'])
dfg['conf_x_adj'] = dfg['conf_x'] #we want to keep clean our conf_x column, we'll only work on the conf_x_adj column

#for teams that have same number of awards, we give them different conf_x_adj values so that they will not overlap on the plot
#note that this would not work if three different teams had same number of awards
dfg.loc[((dfg.n_awards==dfg.n_awards.shift(1)) & (dfg.conf_x==1)), 'conf_x_adj'] = 1.15 
dfg.loc[((dfg.n_awards==dfg.n_awards.shift(-1)) & (dfg.conf_x==1)), 'conf_x_adj'] = 0.85

dfg.loc[((dfg.n_awards==dfg.n_awards.shift(1)) & (dfg.conf_x==2)), 'conf_x_adj'] = 2.15
dfg.loc[((dfg.n_awards==dfg.n_awards.shift(-1)) & (dfg.conf_x==2)), 'conf_x_adj'] = 1.85
def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        pass

    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):    
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    return artists
fig, ax = plt.subplots(figsize=(14, 16)) #we want good height for the plot not to be too compact
ax.scatter(dfg.conf_x_adj, dfg.n_awards)
ax.set_xlabel('')

#organizing the plot
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(['West', 'East'])

#attributing each scatter point a logo
for i, r in dfg.iterrows():
    imscatter(r.conf_x_adj, r.n_awards, '../input/nba-logos/'+r.franchise+'.png', zoom=1)
    
plt.show()
df_king = dff[dff.franchise.isin(['Cavaliers', 'Heat'])]
df_king['player'].value_counts(normalize=True)
fig = plt.figure(figsize=(15, 8))
sns.boxplot(y='n_awards', x='conf_x', data=dfg)
dff = df.groupby('franchise')['player'].nunique().reset_index() #by using nunique, we consider a player only once if he won multiple awards
dff = dff.rename(columns={'player':'n_awards', 'index':'franchise'}) #being explicit
dff.sort_values('n_awards', ascending=False).reset_index(drop=True)[:5] #getting the top 5 teams
dfg = df[df.franchise=='Nets'].groupby('player').agg({'n_awards':'count', 'season_short':'max'}).reset_index()
dfg['player'] = dfg['player'].str.replace(' ', '\n') #replacing spaces by line breaks for better visualization on xticks
dfg = dfg.sort_values('n_awards', ascending=False) #sorting the dataframe for the plot
dfg['edgecolor'] = '#000000' #Nets color code

fig = plt.figure(figsize=(18, 8))
ax = sns.barplot('player', 'n_awards', data=dfg, edgecolor = dfg['edgecolor'].tolist(), linewidth=4, facecolor="None")
plt.xticks(fontsize=11)
ax.set_xlabel('')

ax2 = ax.twinx() #in order to get two different axis with different scales
ax2.scatter('player', 'season_short', data=dfg, color=dfg['edgecolor'].tolist())

ax.set_xticklabels(dfg.player.tolist())
ax.grid(False) #they looked really bad, trust me
ax2.grid(False)
#getting the data directly from Wikipedia, naming it dff for 'df_finals'
#Kaggle can't seem to handle scraping directly to Wikipedia
#we added the output of df_wiki = pd.read_html('https://en.wikipedia.org/wiki/List_of_NBA_champions', header=0)[5] to our private datasets
#below is the rest of the treatment
dffi = pd.read_csv('../input/nbafinals/nba_finals.csv', index_col=0)
#renaming columns for easier manipulation
dffi.columns = [x.lower().replace(' ', '_') for x in dffi.columns]
#getting rid of Wikipedia's references on some rows in order to convert year to int
for letter in ['\[a\]', '\[b\]', '\[c\]', '\[d\]', '\[e\]', '\[f\]']: 
    dffi['year'] = dffi['year'].str.replace(letter, '')
dffi['year'] = dffi['year'].astype(int)

#extracting the franchise names only
for conf in ['western_champion', 'eastern_champion']: 
    dffi[conf] = dffi[conf].apply(lambda x:x.split('(')[0])
    dffi[conf] = dffi[conf].apply(lambda x:x.rstrip().split(' ')[-1])
    dffi[conf] = dffi[conf].str.replace('SuperSonics', 'Thunder')

#adding a nba_champion column (eastern teams did win a few rings so we can't just dffi['nba_champion'] = dffi['western_champion'])
dffi['western_wins'] = dffi.result.apply(lambda x:x.split('–')[0])
dffi['eastern_wins'] = dffi.result.apply(lambda x:x.split('–')[-1])
dffi['nba_champion'] = np.where(dffi.western_wins > dffi.eastern_wins, dffi.western_champion, dffi.eastern_champion)

#inserting the match-up in an array
dffi['finals'] = dffi['western_champion'].map(str) + ' ' + dffi['eastern_champion'].map(str)
dffi['finals'] = dffi.finals.apply(lambda x:x.split(' '))

#getting only the data we need
dffi = dffi[dffi.year>1984]
dffi = dffi[['year', 'finals', 'nba_champion']]
dffi = dffi.rename(columns={'year':'season_short'})
dffi.tail()
dff1 = df.groupby(['season_short', 'franchise']).agg({'n_awards':'sum', 'date':'max'}).reset_index() #grouping by n_awards while getting the maximum date
dff1 = dff1.sort_values(['season_short', 'n_awards', 'date'])
dff1 = dff1.drop_duplicates('season_short', keep='last') #keeping the latest award given in case of equality
dff = pd.merge(dff1, dffi, on='season_short')
dff['franchise_was_finalist'] = dff.apply(lambda x:x['franchise'] in x['finals'], axis=1) #check if best_franchise was if the finals, returns boolean
dff['franchise_was_champion'] = dff.apply(lambda x:x['franchise']==x['nba_champion'], axis=1) #check if best_franchise was champion, returns boolean
dff
print('Best franchises (in terms of Player of the Week awards received) reaching the NBA Finals')
print(dff.franchise_was_finalist.value_counts(normalize=True))
print()
print('Best franchises (in terms of Player of the Week awards received) becoming NBA Champions')
print(dff.franchise_was_champion.value_counts(normalize=True))