import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 100)



from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



men_folder_path = '/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/'

women_folder_path = '/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/'

stage1_folder_path = 'MDataFiles_Stage1/'
def load_section1():

    folder_path = men_folder_path

    prefix = 'M'

    team_section1 = pd.read_csv(folder_path+stage1_folder_path+prefix+'Teams.csv')

    season_section1 = pd.read_csv(folder_path+stage1_folder_path+prefix+'Seasons.csv')

    seed_section1 = pd.read_csv(folder_path+stage1_folder_path+prefix+'NCAATourneySeeds.csv')

    regular_section1 = pd.read_csv(folder_path+stage1_folder_path+prefix+'RegularSeasonCompactResults.csv')

    nacc_section1 = pd.read_csv(folder_path+stage1_folder_path+prefix+'NCAATourneyCompactResults.csv')

    regular_section1['is_regular'] = 1

    nacc_section1['is_regular'] = 0

    regular_detail_section1 = pd.read_csv(folder_path+stage1_folder_path+prefix+'RegularSeasonDetailedResults.csv')

    nacc_detail_section1 = pd.read_csv(folder_path+stage1_folder_path+prefix+'NCAATourneyDetailedResults.csv')

    regular_section1 = regular_section1.merge(regular_detail_section1)

    nacc_section1 = nacc_section1.merge(nacc_detail_section1)

    

    section1 = regular_section1.append(nacc_section1, ignore_index=True).sort_values(by=['Season','DayNum'])

    section1 = section1.merge(seed_section1,left_on=['Season','WTeamID'],right_on=['Season','TeamID']).drop('TeamID',axis=1)

    section1 = section1.merge(seed_section1,left_on=['Season','LTeamID'],right_on=['Season','TeamID'],suffixes=('_W', '_L')).drop('TeamID',axis=1)

    section1 = section1.merge(season_section1,left_on=['Season'],right_on=['Season'])

    section1 = section1.merge(team_section1,left_on=['WTeamID'],right_on=['TeamID']).drop('TeamID',axis=1)

    section1 = section1.merge(team_section1,left_on=['LTeamID'],right_on=['TeamID'],suffixes=('_W', '_L')).drop('TeamID',axis=1)

    section1 = section1.sort_values(by=['Season','DayNum'])

    

    return section1



mSection1 = load_section1()

mSection1
final_round = mSection1.query('DayNum == 154')



plt.subplots(figsize=(20, 8))

plt.subplots_adjust(wspace=0, hspace=.6)



plt.subplot(2,1,1)

plt.xticks(rotation=30)

sns.countplot(x="TeamName_W", data=final_round)



plt.subplot(2,1,2)

plt.xticks(rotation=30)

sns.countplot(x="TeamName_L", data=final_round)
team32 = mSection1.query('DayNum == 136 or DayNum == 137').groupby('TeamName_W')['Season'].count().rename('Count32').reset_index().sort_values(by='Count32',ascending=False).iloc[:10]

team16 = mSection1.query('DayNum == 138 or DayNum == 139').groupby('TeamName_W')['Season'].count().rename('Count16').reset_index().sort_values(by='Count16',ascending=False).iloc[:10]

team8 = mSection1.query('DayNum == 143 or DayNum == 144').groupby('TeamName_W')['Season'].count().rename('Count8').reset_index().sort_values(by='Count8',ascending=False).iloc[:10]

team4 = mSection1.query('DayNum == 145 or DayNum == 146').groupby('TeamName_W')['Season'].count().rename('Count4').reset_index().sort_values(by='Count4',ascending=False).iloc[:10]

team2 = mSection1.query('DayNum == 152').groupby('TeamName_W')['Season'].count().rename('Count2').reset_index().sort_values(by='Count2',ascending=False).iloc[:10]



plt.subplots(figsize=(20, 10))



plt.subplot(2,3,1)

plt.xticks(rotation=30)

sns.barplot(x="TeamName_W", y='Count32', data=team32)

plt.subplot(2,3,2)

plt.xticks(rotation=30)

sns.barplot(x="TeamName_W", y='Count16', data=team16)

plt.subplot(2,3,3)

plt.xticks(rotation=30)

sns.barplot(x="TeamName_W", y='Count8', data=team8)

plt.subplot(2,3,4)

plt.xticks(rotation=30)

sns.barplot(x="TeamName_W", y='Count4', data=team4)

plt.subplot(2,3,5)

plt.xticks(rotation=30)

sns.barplot(x="TeamName_W", y='Count2', data=team2)
tournament = mSection1.query('DayNum >= 136 and DayNum <= 154')

tournament['ScoreDiv'] = tournament['WScore'] - tournament['LScore']

tournament = tournament.groupby('Season')[['WScore','LScore','ScoreDiv']].mean().reset_index()



plt.subplots(figsize=(20, 5))

sns.barplot(x="Season", y='ScoreDiv', data=tournament)
final_round = mSection1.query('DayNum == 154')

final_round['Seed_W_num'] = final_round.Seed_W.apply(lambda seed:int(seed[1:]) if len(seed)==3 else int(seed[1:-1]))

final_round['Seed_L_num'] = final_round.Seed_L.apply(lambda seed:int(seed[1:]) if len(seed)==3 else int(seed[1:-1]))

final_round['Seed_W_code'] = final_round.Seed_W.apply(lambda seed:'Region'+seed[0])

final_round['Seed_L_code'] = final_round.Seed_L.apply(lambda seed:'Region'+seed[0])

final_round['Seed_W_region'] = final_round.apply(lambda row:row[row['Seed_W_code']], axis=1)

final_round['Seed_L_region'] = final_round.apply(lambda row:row[row['Seed_L_code']], axis=1)



plt.subplots(figsize=(20, 10))



plt.subplot(3,1,1)

sns.countplot(x="Seed_W_num", data=final_round)

plt.subplot(3,1,2)

sns.countplot(x="Seed_W_region", data=final_round)

plt.subplot(3,1,3)

sns.lineplot(x="Season", y="Seed_W_num", estimator=None, data=final_round)
cols = ['TeamID','Score','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']

cols = ['W'+col for col in cols] + ['L'+col for col in cols]



regular_data = mSection1.query('is_regular == 1')[cols].mean().rename('value')

tournament_data = mSection1.query('is_regular == 0 and DayNum < 154')[cols].mean().rename('value')

final_data = mSection1.query('DayNum == 154')[cols].mean().rename('value')



tmp = pd.DataFrame({})

tmp = tmp.append([['regular']+list(regular_data.values)],ignore_index=True)

tmp = tmp.append([['tournament']+list(tournament_data.values)],ignore_index=True)

tmp = tmp.append([['final']+list(final_data.values)],ignore_index=True)

tmp.columns = ['match_type']+list(regular_data.index)

tmp
plt.subplots(figsize=(20, 3*5))



for i,col in enumerate(['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']):

#     plt.subplot(5,3,i*3+1)

#     sns.barplot(x="match_type", y='W'+col, data=tmp)

#     plt.subplot(5,3,i*3+2)

#     sns.barplot(x="match_type", y='L'+col, data=tmp)

#     plt.subplot(5,3,i*3+3)

#     tmp['W'+col+'_L'+col] = tmp['W'+col] - tmp['L'+col]

#     sns.barplot(x="match_type", y='W'+col+'_L'+col, data=tmp)

    plt.subplot(5,3,i+1)

    tmp['W'+col+'_L'+col] = tmp['W'+col] - tmp['L'+col]

    sns.barplot(x="match_type", y='W'+col+'_L'+col, data=tmp)