# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk(''):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Create IPL winners dataframe

index = ['IPL-2008', 'IPL-2009', 'IPL-2010', 'IPL-2011', 'IPL-2012', 'IPL-2013', 'IPL-2014', 'IPL-2015', 'IPL-2016', 'IPL-2017', 'IPL-2018', 'IPL-2019']

data = {'Winner':['RR', 'SRH', 'CSK', 'CSK', 'KKR', 'MI', 'KKR', 'MI', 'SRH', 'MI', 'CSK', 'MI']}

iplwinners = pd.DataFrame(data, index)



import matplotlib.pyplot as plt



win = iplwinners.Winner.value_counts()



win_palette = ['royalblue', 'yellow', 'rebeccapurple', 'darkorange', 'deeppink']



plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(10,7))

ax.bar(win.index,win, color = win_palette)

ax.set(title='Which Team has the most IPL titles?')

ax.set(yticks = np.arange(0,6,1))

# Load IPL Kaggle dataset into df

df = pd.read_csv('../input/ipl-dataset-20082019/matches.csv', index_col='id')



# Clean IPL dataset

# Remove 'umpire3' column as it has too many NAs

df = df.drop('umpire3', axis = 1)



# Replace NAs in 'winner' and 'player of match' columns with 'no result' and '-'

df.winner = df.winner.fillna('no result')

df.player_of_match = df.player_of_match.fillna('-')



# All the matches with NAs for city were held in Dubai based on the entry for venue. Replace the NAs in 'city' column with 'Dubai'

df.city = df.city.fillna("Dubai")



# Adding consistent spelling for Bengaluru

df = df.replace('Bangalore', 'Bengaluru')



# Replace team names with their abbreviated forms for ease of data referencing

df = df.replace('Rising Pune Supergiants', 'RPS')

df = df.replace('Rising Pune Supergiant', 'RPS')

df = df.replace('Pune Warriors', 'RPS')

df = df.replace('Deccan Chargers', 'SRH')

df = df.replace('Sunrisers Hyderabad', 'SRH')

df = df.replace('Delhi Daredevils', 'DC')

df = df.replace('Delhi Capitals', 'DC')

df = df.replace('Chennai Super Kings', 'CSK')

df = df.replace('Gujarat Lions', 'GL')

df = df.replace('Kings XI Punjab', 'KXIP')

df = df.replace('Kochi Tuskers Kerala', 'KTK')

df = df.replace('Kolkata Knight Riders', 'KKR')

df = df.replace('Mumbai Indians', 'MI')

df = df.replace('Rajasthan Royals', 'RR')

df = df.replace('Royal Challengers Bangalore', 'RCB')



# Calculate win percentage for teams over different IPL seasons



# Step 1: Calculate matches played per season by team

matchmelt = pd.melt(df.reset_index(), id_vars=['id','winner','Season'], value_vars=['team1', 'team2'],value_name='Team')

matchesperseason = matchmelt.pivot_table(index='Team', columns='Season', values='id', aggfunc='count', fill_value=0)

matchesperseason['Total'] = matchesperseason.sum(axis=1)



# Step 2: Calculate wins played per season for each team

winsperseason = matchmelt.pivot_table(index = 'winner', columns='Season', values='id', aggfunc='count', fill_value=0) * 0.5

winsperseason['Total'] = winsperseason.sum(axis=1, numeric_only=True)



# Step 3: Calculate percentage win and drop defunct teams (GL, KTK and RPS) from the comparison

winpercent = round(winsperseason/matchesperseason * 100,2)

winpercent = winpercent.fillna(method='bfill') #Using 'bfill' to calculate win percent for CSK and RR for 2016 and 2017 seasons

winpercent = winpercent.drop(labels=['no result','GL', 'KTK', 'RPS'], axis = 0)

winpercent.index.name = 'Team'



# Step 4: Plot win percentage across all seasons for IPL teams



import matplotlib.pyplot as plt



palette = ['yellow', 'dodgerblue','rebeccapurple' ,'salmon', 'royalblue', 'firebrick','deeppink', 'darkorange']



plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(13,7))

ax.bar(winpercent.index,winpercent.Total, color = palette)

ax.set(title='Which Team has the Highest Percentage Wins Across all IPL Seasons?')

plt.ylim(top = 68)

ax.text(0,62.5,'1', fontsize = 20, horizontalalignment = 'center')

ax.text(4,59,'2', fontsize = 20, horizontalalignment = 'center')

ax.text(2,52,'3', fontsize = 20, horizontalalignment = 'center')
# Add Season Title Winner's win percen to iplwinner df

iplwinners['Highest Win Percent'] = winpercent.idxmax(axis = 0)

x = winpercent.loc[iplwinners.Winner, iplwinners.index]

iplwinners['Winner Win Percent'] = np.diag(x)



# Plot Win Percentage for Teams across all Seasons and 

winoverseason = winpercent.iloc[:,0:12].T

winoverseason.index.name = ''



plt.style.use('fivethirtyeight')

winoverseason.plot(figsize = (20,10), linewidth = 3, color = palette, xticks = np.arange(0,12,1), title = 'Have the Teams with the Highest Percentage Wins for the Season Won the IPL Title for that Season?')                          

season = iplwinners.iloc[:,2].plot(marker = "o", linewidth = 0, color = 'lime', markersize = 10)

season.legend(loc = 'best',ncol = 5)
# Calculate matches and wins at home and away



# Define a function to determine home and away match counts which accepts a df with teams with index: teams, columns: cities, values: counts of matches



def calc_home_away (homeaway_pt):

    homeaway_pt.loc[:,'Total'] = homeaway_pt.sum(axis=1)

    homeaway_pt.loc['CSK', 'Home'] = homeaway_pt.loc['CSK','Chennai']

    homeaway_pt.loc['DC', 'Home'] = homeaway_pt.loc['DC','Delhi'] + homeaway_pt.loc['DC','Raipur']

    homeaway_pt.loc['GL', 'Home'] = homeaway_pt.loc['GL','Ahmedabad'] + homeaway_pt.loc['GL','Rajkot'] + homeaway_pt.loc['GL','Kanpur']

    homeaway_pt.loc['KXIP', 'Home'] = homeaway_pt.loc['KXIP','Chandigarh'] + homeaway_pt.loc['KXIP','Dharamsala'] + homeaway_pt.loc['KXIP','Mohali']

    homeaway_pt.loc['KTK', 'Home'] = homeaway_pt.loc['KTK','Kochi'] 

    homeaway_pt.loc['KKR', 'Home'] = homeaway_pt.loc['KKR','Kolkata'] 

    homeaway_pt.loc['MI', 'Home'] = homeaway_pt.loc['MI','Mumbai']

    homeaway_pt.loc['RR', 'Home'] = homeaway_pt.loc['RR','Jaipur']  

    homeaway_pt.loc['RPS', 'Home'] = homeaway_pt.loc['RPS','Pune'] 

    homeaway_pt.loc['RCB', 'Home'] = homeaway_pt.loc['RCB','Bengaluru'] 

    homeaway_pt.loc['SRH', 'Home'] = homeaway_pt.loc['SRH','Hyderabad'] + homeaway_pt.loc['SRH','Visakhapatnam'] + homeaway_pt.loc['SRH','Nagpur']

    homeaway_pt.loc[:,'Away'] = homeaway_pt.loc[:,'Total'] - homeaway_pt.loc[:,'Home']

    rows = ['CSK', 'DC','KKR', 'KXIP', 'MI', 'RCB', 'RR', 'SRH']

    columns = ['Total', 'Home', 'Away']

    homeaway_pt = homeaway_pt.loc[rows,columns]

    homeaway_pt.index.name = 'Team'

    return homeaway_pt



# Pivot ipl dataset df in the format acceptable by calc_home_away function to determine wins at home and away

homeaway_pt = df.reset_index().pivot_table(values='id', index = 'winner', columns= 'city', aggfunc= 'count', fill_value=0)



# Calculate wins at home and away

wins = calc_home_away(homeaway_pt=homeaway_pt)

wins.columns.name = 'Win'

wins.columns = ['Total Wins', 'Home Wins', 'Away wins']



# Pivot ipl dataset df in the format acceptable by calc_home_away function to determine matches played at home and away

dfmelt = df.melt(id_vars='city',value_vars=['team1','team2'], value_name= 'team')

played_pt = dfmelt.pivot_table(values = 'variable', index = 'team', columns='city', aggfunc='count',fill_value=0)



# Calculate matches played at home and away

played = calc_home_away(homeaway_pt = played_pt)

played.columns.name = 'Played'

played.columns = ['Total Played', 'Played at Home', 'Played Away']



# Calculate percentage wins at home and away

homeawaypercent = pd.concat([wins,played],axis = 1)

homeawaypercent['Home Win Percent'] = round(homeawaypercent['Home Wins'] / homeawaypercent['Played at Home'] * 100,2)

homeawaypercent['Away Win Percent'] = round(homeawaypercent['Away wins'] / homeawaypercent['Played Away'] * 100,2)



# Plot the King Porus and Gengis Khans of IPL

import matplotlib.pyplot as plt



plt.style.use('fivethirtyeight')

fig, (ax1, ax2) = plt.subplots(1,2, sharey= True, figsize=(20,7))

ax1.bar(homeawaypercent.index,homeawaypercent.iloc[:,-2], color = palette)

ax1.set(ylabel='Percentage Home Wins Across all IPL Seasons', xlabel='',

       title='King Porus of IPL')

ax1.text(0,73,'1', fontsize = 20, horizontalalignment = 'center')

ax1.text(6,69,'2', fontsize = 20, horizontalalignment = 'center')

ax1.text(4,66,'3', fontsize = 20, horizontalalignment = 'center')

ax2.bar(homeawaypercent.index,homeawaypercent.iloc[:,-1], color = palette)

ax2.set(ylabel='Percentage Away Wins Across all IPL Seasons', xlabel='',

       title='Gengis Khans of IPL')

ax2.text(0,57,'1', fontsize = 20, horizontalalignment = 'center')

ax2.text(4,54,'2', fontsize = 20, horizontalalignment = 'center')

ax2.text(2,46,'3', fontsize = 20, horizontalalignment = 'center')

fig.tight_layout()
# Identify top 5 hunting grounds for each Team across all Seasons of IPL



# Melt the ipl dataset to gather the counts of matches played at each city for different teams 

played = dfmelt.groupby(by=['team','city']).count().sort_values(by = ['variable'],ascending = False).sort_index(level = 0, sort_remaining = False)

played.columns = ['played'] 



# Melt the ipl dataset to gather the counts of matches won at each city for different teams 

dfwin = df.melt(id_vars = 'city', value_vars = 'winner', value_name = 'team')

dfwin = dfwin[dfwin.team != 'no result']

won = dfwin.groupby(by = ['team', 'city']).count().sort_values(by=['variable'],ascending = False).sort_index(level = 0, sort_remaining = False)

won.columns = ['won']



# Merge played and won dfs 

citystats = pd.concat([played, won], axis = 1).fillna(0)

citystats['win_percent'] = round(citystats['won']/citystats['played']*100,2)

citystats = citystats.drop(index = ['KTK', 'RPS', 'GL'])



# Retain only the top 5 cities (by counts of total matches played)

citystats = citystats.sort_values(by=['played'],ascending = False).sort_index(level = 0, sort_remaining = False)

citystats = citystats.groupby(level=0).apply(lambda citystats: citystats[:5])

citystats.index = citystats.index.droplevel(level = 0)

citystats = citystats.reindex()



# Plot the Top 5 Hunting Grounds for the teams across all IPL Seasons

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib



plt.style.use('fivethirtyeight')

fig, axs = plt.subplots(2,4, sharey= True, figsize=(40,20))

fig.suptitle('Top 5 Hunting Grounds for IPL Teams Across All Seasons', fontsize = 40)

fig.add_subplot(111, frame_on=False)

plt.tick_params(labelcolor="none", bottom=False, left=False,grid_alpha = 0)

plt.ylabel('No of Matches', fontsize = 25)



teams = ['CSK', 'DC', 'KKR', 'KXIP', 'MI', 'RCB', 'RR', 'SRH']

palette = ['yellow', 'dodgerblue','rebeccapurple' ,'salmon', 'royalblue', 'firebrick','deeppink', 'darkorange']



i = 0

while i<8:

      for rows in range(0,2):

          for cols in range(0,4):

              dfplot = citystats.loc[teams[i]]

              axs[rows,cols].bar(dfplot.index, dfplot.iloc[:,0], color = palette[i], alpha = 0.4, label = 'Played')

              axs[rows,cols].bar(dfplot.index, dfplot.iloc[:,1], color = palette[i], label = 'Won')

              axs[rows,cols].legend(facecolor = 'white')

              i +=1

              del dfplot

            

patches = [ mpatches.Patch(color=palette[i], label="{:s}".format(teams[i]) ) for i in range(len(teams)) ]

fig.legend(handles=patches, loc= 'center right', fontsize = 25, borderaxespad = 0.2, labelspacing = 0.5)  
print ('\nThe top 3 cities in terms of the no. of matches hosted in IPL are:\n', df.city.value_counts()[0:3])

print ('\nThe different stadiums at Mumbai are:\n',df.loc[df['city'] == 'Mumbai'].venue.unique())
# Create analysis dataframe

index = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5']

data = {'Description':['Total No. of Titles Won', 'Percentage Wins Across all IPL Seasons', 'Do the Teams with Highest Win Percentages win the IPL Title?', 'King Porus and Gengis Khans of IPL', 'Top 5 Hunting Grounds for Teams Across IPL Seasons'],

       'Winner': ['MI', 'CSK', 'MI', 'CSK','CSK']}

analysis = pd.DataFrame(data, index)

analysis
print("\n The title for the Best IPL team goes to: ", analysis.Winner.value_counts().idxmax())
# Delete all objects and user defined functions



for name in dir():

    if not name.startswith('_'):

        del globals()[name]
