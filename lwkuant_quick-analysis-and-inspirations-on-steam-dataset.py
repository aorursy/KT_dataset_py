## load the required packages

import numpy as np

import scipy as sp 

import pandas as pd 

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns
## load the dataset

df = pd.read_csv('../input/steam-200k.csv', header=None, encoding='utf-8')



print(df.shape)

print(df.head())



## assign the column names 

col_name = ['User', 'Game', 'Behavior', 'Behavior_value', 'Other']

df.columns = col_name

print()

print(df.head())

print(df.info())
## overview of the dataset 



# check if there are NAs

print('Number of NAs:')

print(np.sum(df.isnull()))



# chek if there are multiple vales for the column "Other"

print()

print(df['Other'].value_counts()) # since there are no various values for column "Other", I can remove this column



# remove the column "Other"

df.drop(['Other'], axis=1, inplace=True)

print()

print(df.columns)
# group by user and game to test

group_user_game = df.groupby(['User', 'Game'])

print(group_user_game['Game'].count().value_counts()) 
# check the 3 and 4 in the grouped table

print(group_user_game['Game'].count()[group_user_game['Game'].count().isin([3, 4])].head(10)) # take some examples



# display one example (for 3)

print()

print("for 3")

print(df.ix[(df['User'] == 1936551)&(df['Game'] == "Sid Meier's Civilization IV Colonization"), :])



# display one example (for 4)

print()

print("for 4")

print(df.ix[(df['User'] == 176261926)&(df['Game'] == "Sid Meier's Civilization IV"), :])



# extract the games in 3 and 4 group

print()

odd_games = [name[1] for name in list(dict(group_user_game['Game'].count()[group_user_game['Game'].count().isin([3, 4])]).keys())]

print(np.unique(odd_games))



print()

print('There are', len(np.unique(df['Game'])), 'Games')
# display some examples

print(group_user_game['Game'].count()[group_user_game['Game'].count().isin([1])].head(10))



# get the games' names

from collections import defaultdict

aban_games_dict = defaultdict(int)



# calculate the frequencies

for game in list(dict(group_user_game['Game'].count()[group_user_game['Game'].count().isin([1])]).keys()):

    aban_games_dict[game[1]] += 1



purchase_games_dict = defaultdict(int)



for game in list(dict(group_user_game['Game'].count()[group_user_game['Game'].count().isin([1, 2])]).keys()):

    purchase_games_dict[game[1]] += 1



ratio_dict = {}



for game in list(aban_games_dict.keys()):

    ratio_dict[game] = aban_games_dict[game]/purchase_games_dict[game]



aban_games_df = pd.Series(ratio_dict)

#print(aban_games_df.sort_values(ascending=False))



aban_games_df = pd.DataFrame(aban_games_df, columns=['Adandon_ratio'], index=list(aban_games_df.index))



purchase_list = []

for game in list(aban_games_df.index):

    purchase_list.append(purchase_games_dict[game])



aban_games_df['Purchase_times'] = purchase_list



print(aban_games_df.sort_values(by='Adandon_ratio', ascending=False))
print(aban_games_df.ix[aban_games_df['Purchase_times']>100, :].sort_values(by='Adandon_ratio', ascending=False))
# get the times played

played_games_dict = defaultdict(int)



for game in list(dict(group_user_game['Game'].count()[group_user_game['Game'].count().isin([2])]).keys()):

    played_games_dict[game[1]] += 1

    

# get the hours played

played_time = df.ix[(df['Game'].isin(list(played_games_dict.keys())))&(df['Behavior'] == 'play'), :].groupby(['Game'])

#print(played_time['Behavior_value'].sum())



play_df = pd.Series(dict(played_time['Behavior_value'].sum()))

play_df = pd.DataFrame(play_df, columns=['Time_played'], index=list(play_df.index))



play_list = []

for game in list(play_df.index):

    play_list.append(played_games_dict[game])



play_df['User_num'] = play_list



play_df['Avg_play_time'] = play_df['Time_played']/play_df['User_num']

print(play_df.sort_values(by=['Avg_play_time'], ascending=False).head(10))
## the distribution of number of users playing each game



num_users_list = play_df['User_num'].values



print(np.max(num_users_list))



fig, axes = plt.subplots(figsize=[5, 5])

sns.boxplot(data = num_users_list)

axes.set_title('The Distribution of Number of Users Playing Each Game')



print(np.percentile(num_users_list, q=[25, 50, 75]))
## The most-played games by total hours played

print(play_df.sort_values(by=['Time_played'], ascending=False).head())



play_df.sort_values(by=['Time_played'], ascending=True)[-10:]['Time_played'].plot.barh(color='#2E8B57', edgecolor='none')

plt.title('Top 10 Most-Played Games (by total time played)', fontsize=15)

plt.xlabel('Time played (in hours)')
## The most-played games by number of users playing 

print(play_df.sort_values(by=['User_num'], ascending=False).head())



play_df.sort_values(by=['User_num'], ascending=True)[-10:]['User_num'].plot.barh(color='#2E8B57', edgecolor='none')

plt.title('Top 10 Most-Played Games (by number of users playing)', fontsize=15)

plt.xlabel('Number of users playing')
## The most-played games by average of hours played and meeting certain number of users 

print(play_df.sort_values(by=['Avg_play_time'], ascending=False).head())



# use 100 as the criteria for hotness

play_df.ix[play_df['User_num']>100, :].sort_values(by=['Avg_play_time'], ascending=True)[-10:]['Avg_play_time'].plot.barh(color='#2E8B57', edgecolor='none')

plt.title('Top 10 Most-Played Games (by average time played)', fontsize=15)

plt.xlabel('Average time played (in hours)')