# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the libraries

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss



import seaborn as sns

from IPython.display import display


Wteams = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WTeams.csv')

Wteams.head()
# No of Teams



Wteams['TeamID'].nunique()
Wseason = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WSeasons.csv')

Wseason.tail()
# Total held seasons including the current

Wseason['Season'].count()
Wseeds = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneySeeds.csv')

Wseeds.head()
Wseeds = pd.merge(Wseeds, Wteams,on='TeamID')

Wseeds.head()

# Separating the regions from the Seeds



Wseeds['Region'] = Wseeds['Seed'].apply(lambda x: x[0][:1])

Wseeds['Seed'] = Wseeds['Seed'].apply(lambda x: int(x[1:3]))

print(Wseeds.head())

print(Wseeds.shape)
# Teams with maximum top seeds

fig = plt.gcf()

fig.set_size_inches(10, 6)

colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 



Wseeds[Wseeds['Seed'] ==1]['TeamName'].value_counts()[:10].plot(kind='bar',color=colors,linewidth=2,edgecolor='black')

plt.xlabel('Number of times in Top seeded positions')
# Teams with maximum lowest seeds

fig = plt.gcf()

fig.set_size_inches(10, 6)



Wseeds[Wseeds['Seed'] ==16]['TeamName'].value_counts()[:10].plot(kind='bar',color=colors,edgecolor='black',linewidth=1)

plt.xlabel('Number of times in bottom seeded positions')
rg_season_compact_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

rg_season_compact_results.head()
# Winning and Losing score Average over the years

x = rg_season_compact_results.groupby('Season')[['WScore','LScore']].mean()



fig = plt.gcf()

fig.set_size_inches(14, 6)

plt.plot(x.index,x['WScore'],marker='o', markerfacecolor='green', markersize=12, color='green', linewidth=4)

plt.plot(x.index,x['LScore'],marker=7, markerfacecolor='red', markersize=12, color='red', linewidth=4)

plt.legend()

tourney_compact_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

tourney_compact_results .tail()
games_played = tourney_compact_results.groupby('Season')['DayNum'].count().to_frame().merge(rg_season_compact_results.groupby('Season')['DayNum'].count().to_frame(),on='Season')

games_played.rename(columns={"DayNum_x": "Tournament Games", "DayNum_y": "Regular season games"})

ax = sns.countplot(x=tourney_compact_results['WLoc'])

ax.set_title("Win Locations")

ax.set_xlabel("Location")

ax.set_ylabel("Frequency");
tourney_detailed_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv')

tourney_detailed_results.head()
tourney_detailed_results.columns
ax = sns.countplot(x=tourney_detailed_results['WLoc'])

ax.set_title("Win Locations")

ax.set_xlabel("Location")

ax.set_ylabel("Frequency");



games_stats = []



for row in tourney_detailed_results.to_dict('records'):

    game = {}

    game['Season'] =  row['Season']

    game['DayNum'] = row['DayNum']

    game['TeamID'] = row['WTeamID']

    game['OpponentID'] = row['LTeamID']

    game['FGM'] = row['WFGM']

    game['Loc'] = row['WLoc']

    game['Won'] = 1

    game['Score'] = row['WScore']

    game['FGA'] = row['WFGA']

    game['FGM3'] = row['WFGM3']

    game['FGA3'] = row['WFGA3']

    game['FTM'] = row['WFTM']

    game['FTA'] = row['WFTA']

    game['OR'] = row['WOR']

    game['DR'] = row['WDR']

    game['AST'] = row['WAst']

    game['TO'] = row['WTO']

    game['STL'] = row['WStl']

    game['BLK'] = row['WBlk']

    game['PF'] = row['WPF']

    games_stats.append(game)

    game = {}

    game['Season'] = row['Season']

    game['DayNum'] = row['DayNum']

    game['TeamID'] = row['LTeamID']

    game['OpponentID'] = row['WTeamID']

    game['FGM'] = row['LFGM']

    game['Loc'] = row['WLoc']

    game['Won']= 0

    game['Score'] = row['LScore']

    game['FGA'] = row['LFGA']

    game['FGM3'] = row['LFGM3']

    game['FGA3'] = row['LFGA3']

    game['FTM'] = row['LFTM']

    game['FTA'] = row['LFTA']

    game['OR'] = row['LOR']

    game['DR'] = row['LDR']

    game['AST'] = row['LAst']

    game['TO'] = row['LTO']

    game['STL'] = row['LStl']

    game['BLK'] = row['LBlk']

    game['PF'] = row['LPF']

    games_stats.append(game)



    



# Separating winners and losers using Won Column which is set to 1 for winner and 0 for loser



tournament = pd.DataFrame(games_stats)

tournament.head()
tournament_df = pd.merge(tournament , Wseeds, on= ['Season','TeamID'])

tournament_df.rename(columns={'Seed': 'Team_Seed'}, inplace=True)

tournament_df[:2]
tournament_df2 = pd.merge(tournament_df , Wseeds.rename(columns={'TeamID':'OpponentID'}), on= ['Season','OpponentID'])

tournament_df2 .rename(columns={'Seed': 'OpponentSeed',

                                'TeamName_x':'Team',

                                'TeamName_y':'Opponents',

                                 'Region_x':'Team_Region',

                                 'Region_y':'Opponent_Region'}, inplace=True)

tournament_df2 .head()
# Winning_Teams



winning_Teams = tournament_df2[tournament_df2['Won'] == 1]



# Losing_Teams



losing_Teams = tournament_df2[tournament_df2['Won'] == 0]

winning_Teams.head().T
# Most successful teams

fig = plt.gcf()

fig.set_size_inches(10, 6)



colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 

winning_Teams['Team'].value_counts()[:10].plot(kind='bar',color=colors,edgecolor='black',linewidth=1 )



plt.title('Most successful Teams')

plt.tight_layout(h_pad=2)
# How seed ranking affect game stats for winners

sns.pairplot(winning_Teams[['FGA3','FGM3','AST','BLK','DR','FTA','FTM','OR','Team_Seed',]], hue='Team_Seed',kind="scatter",plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
# Features Correlated with Wins

f,ax = plt.subplots(figsize=(20,15))

corr = tournament_df2.corr()

sns.heatmap(corr, cmap='inferno', annot=True)

# Converting Loc column to numeric



tournament_df2 = pd.get_dummies(tournament_df2, columns=['Loc'])

tournament_df2.info()
train = tournament_df2[tournament_df2['Season'] < 2015]

validation = tournament_df2[tournament_df2['Season'] >= 2015]
train.columns

validation.columns


X = ['DayNum', 'TeamID', 'OpponentID', 'FGM','FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'AST', 'TO', 'STL',

       'BLK', 'PF', 'Team_Seed','Loc_A', 'Loc_H','Loc_N']

y = 'Won'



model = LogisticRegression(solver='liblinear',C=1.0)

model.fit(train[X],train[y])
predictions = mode.predict_proba(validation1[X])[:, 1]

validation.head()
# submit predicitions



submission = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')

submission.head()
validation['Pred'] = predictions

validation['ID'] = validation.apply(lambda row: '{}_{}_{}'.format(int(row['Season']), int(row['TeamID']), int(row['OpponentID'])), axis=1)


predictions = pd.merge(submission.drop('Pred', axis = 1), validation[['ID', 'Pred']], how='left', on=['ID']).fillna(0.5)
predictions.shape
predictions.to_csv('submision_baseline.csv',index=False)