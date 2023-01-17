#Import modules and initial setup

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import operator

from functools import reduce

from IPython.display import display, HTML, Image

color = sns.color_palette()

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
#Load files

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

main_df = pd.read_csv('../input/_LeagueofLegends.csv', index_col='MatchHistory')

gold_df = pd.read_csv('../input/goldValues.csv', index_col='MatchHistory')

gold_df = gold_df.loc[main_df.index]
#Number of matches per year, sesason, league

print("{} macthes in dataset".format(main_df.shape[0]))
#Pivot table league - year

league_table = pd.pivot_table(main_df, values=['Season'], index=['Year'],

                     columns=['League'], aggfunc='count').fillna("-")

display(league_table)
#Pivot table season - year

season_table = pd.pivot_table(main_df, values=['League'], index=['Year'],

                     columns=['Season'], aggfunc='count').fillna("-")

display(season_table)
#Graph matches

plt.figure(1, figsize=(18,4)) 

plt.subplot(131)

sns.countplot(x='Year', data=main_df, order=[2014, 2015, 2016, 2017], palette="Greens_d")

plt.xticks(rotation=30)

plt.ylabel('Number of matches')

plt.title('Matches per year')

plt.subplot(132)

sns.countplot(x='Season', data=main_df, palette="Greens_d")

plt.xticks(rotation=90)

plt.ylabel('Number of matches')

plt.title('Matches per season')

plt.subplot(133)

sns.countplot(x='League', data=main_df, palette="Greens_d")

plt.xticks(rotation=90)

plt.ylabel('Number of matches')

plt.title('Matches per league')



plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)



plt.show()
# Win Rates

plt.figure(2, figsize=(18,2))

items = ['Year', 'Season', 'League']

i=0

for item in items :

    win_rate_blue = {}

    win_rate_red = {}

    for value in main_df[item].unique():

        winBlue = sum(main_df[main_df[item]==value].bResult)/main_df[main_df[item]==value].shape[0]

        win_rate_blue[value]=winBlue

        win_rate_red[value]=1-winBlue

  

    i+=1

    plt.subplot(130+i)

    p1 = plt.bar(range(len(list(win_rate_blue.keys()))), win_rate_blue.values(), color='blue')

    p2 = plt.bar(range(len(list(win_rate_blue.keys()))), win_rate_red.values(), color='red',

             bottom=win_rate_blue.values())

    plt.ylabel('Win ratio')

    plt.title('Win ratio by team color and {}'.format(item))

    plt.xticks(range(len(list(win_rate_blue.keys()))), list(win_rate_blue.keys()), rotation=90)

    plt.yticks(np.arange(0, 1.0, 0.1))

    plt.axhline(y=0.5, color='black', linestyle='dashed', linewidth=2)

    plt.legend((p1[0], p2[0]), ('Blue', 'Red'))



plt.subplots_adjust(top=1.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)

plt.show()   
# Team assignments and wins

plt.figure(3, figsize=(18,6))

blue_teams = main_df[['blueTeamTag', 'bResult']]

blue_teams.columns=['team', 'win']

blue_teams['color'] = 'Blue'

red_teams = main_df[['redTeamTag', 'rResult']]

red_teams.columns=['team', 'win']

red_teams['color'] = 'red'

all_teams = pd.concat([blue_teams,red_teams])

all_teams.reset_index(drop=True, inplace=True)

mapping = {'Blue':1, 'red':0}

all_teams['blue']=all_teams.replace({'color':mapping})['color']

mapping = {'Blue':0, 'red':1}

all_teams['red']=all_teams.replace({'color':mapping})['color']

g=sns.factorplot(x='team', hue='color', col='win', data=all_teams

                 , kind="count", order=all_teams['team'].value_counts().index[:10]

                 , palette=['blue','red'],size=4, aspect=2)

g.set_axis_labels("Team tag", "Matches")

plt.show()
#Game lenght by year

plt.figure(4, figsize=(18,9))

i=0

for year in sorted(main_df['Year'].unique()):

    data = main_df[main_df['Year']==year]

    i += 1

    plt.subplot(420+i)

    sns.distplot(data['gamelength'], kde=False, bins=20, color='g')

    plt.axvline(data['gamelength'].mean(), color='g', linestyle='dashed', linewidth=2) #Mean line

    #sns.rugplot(data['gamelength'])

    plt.ylabel('Number of matches')

    plt.xlabel('Minutes')

    plt.title('Game lenght for {}'.format(year))

    plt.xlim([20,80])



plt.subplots_adjust(top=1.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)

plt.show()  
#Game lenght by season

plt.figure(5, figsize=(18,6))

i=0

for season in main_df['Season'].unique():

    data = main_df[main_df['Season']==season]

    i += 1

    plt.subplot(330+i)

    sns.distplot(data['gamelength'], kde=False, bins=20)

    plt.axvline(data['gamelength'].mean(), color='b', linestyle='dashed', linewidth=2) #Mean line

    #sns.rugplot(data['gamelength'])

    plt.ylabel('Number of matches')

    plt.xlabel('Minutes')

    plt.title('Game lenght for {}'.format(season))

    plt.xlim([20,80])



plt.subplots_adjust(top=1.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)

plt.show()
#Process gold data to put it in a workable format

gold_df_list = []



for gold_type in gold_df['NameType'].unique() :

    temp = gold_df[gold_df['NameType']==gold_type]

    temp.drop('NameType', axis=1, inplace=True)

    temp.columns = pd.MultiIndex.from_product([[gold_type], temp.columns])

    gold_df_list.append(temp)

    

gold_df_pivot = reduce(lambda x, y: pd.concat([x, y], axis=1), gold_df_list)



#fix types in Gold_df_pivot

gold_df_pivot=gold_df_pivot.apply(pd.to_numeric, errors='coerce')



win_blue_gold = gold_df_pivot.loc[main_df[main_df['bResult']==1].index]

win_red_gold = gold_df_pivot.loc[main_df[main_df['rResult']==1].index]



win_gold = win_blue_gold['goldblue'].append(win_red_gold['goldred'])

lose_gold = win_blue_gold['goldred'].append(win_red_gold['goldblue'])

win_gold['min_1'] = pd.to_numeric(win_gold['min_1'], errors='coerce', downcast='signed')

lose_gold['min_1'] = pd.to_numeric(lose_gold['min_1'], errors='coerce', downcast='signed')
#Plot mean gold per minute

plt.figure(6, figsize=(18,8))

p1=plt.plot(win_gold.mean().values, color='navy')

plt.fill_between(range(win_gold.shape[1])

    ,(win_gold.mean().values+win_gold.std().values)

    , (win_gold.mean().values-win_gold.std().values), alpha=0.30, color='navy')

#plt.errorbar(range(win_gold.shape[1]), win_gold.mean().values , win_gold.std().values, linestyle='None', marker='')

p2=plt.plot(lose_gold.mean().values, color='maroon')

plt.fill_between(range(lose_gold.shape[1])

    ,(lose_gold.mean().values+lose_gold.std().values)

    , (lose_gold.mean().values-lose_gold.std().values), alpha=0.30, color='tomato')

#plt.errorbar(range(lose_gold.shape[1]), lose_gold.mean().values , lose_gold.std().values, linestyle='None', marker='')

#plt.plot((win_gold.mean().values - lose_gold.mean().values), color='black')

plt.legend((p1[0], p2[0]), ('Winner', 'Loser'))

plt.xlabel('Minute')

plt.ylabel('Gold')

plt.title('Mean gold for winner and loser per minute')

plt.show()