import pandas as pd

import numpy as np

df_ipl = pd.read_csv("../input/ipl_dataset.csv")

df_ipl.shape
len(df_ipl['match_code'].unique())

# You can also use: 

df_ipl['match_code'].nunique()
len(set(df_ipl.iloc[:,0]))
# Corrected as Venues to Cities

print(df_ipl['city'].unique())

print(df_ipl['city'].nunique())
print(set(df_ipl['city']))

print(len(set(df_ipl['city'])))
df_ipl.isnull().sum()
vanues = df_ipl.groupby('venue')['match_code'].nunique().sort_values(ascending = False)
vanues.head()
print('Run count frequency table:\n\n',df_ipl['runs'].value_counts())
# Also you can try using pandas function crosstab

import pandas as pd

runs_count = pd.crosstab(df_ipl['runs'], columns='runs_count')

print('Runs count frequency table:\n\n',runs_count)
runs_counts = df_ipl['runs'].value_counts()

print('Runs count frequency table:\n\n',runs_counts.sum())#varified
df_ipl['year'] = df_ipl['date'].apply(lambda x : x[:4])



print('The no. of seasons that were played are :', len(df_ipl['year'].unique()))

print('Seasons played were in :', df_ipl['year'].unique())
df_ipl['date'] = pd.to_datetime(df_ipl['date'])

df_year = set(df_ipl['date'].dt.year)

print('The no. of seasons that were played are :',len(df_year))

print('Seasons played were in :',list(df_year))

matches_per_season = df_ipl.groupby('year')['match_code'].nunique()

print('Matches held per season are :\n\n', matches_per_season)

type(matches_per_season)
matches_per_season['2008']
runs_per_season = df_ipl.groupby('year')['total'].sum()
print('total runs scored across each season',runs_per_season)

print(type(runs_per_season))

print(runs_per_season['2015'])
high_scores=df_ipl.groupby(['match_code', 'inning','team1','team2'])['total'].sum().reset_index() 

type(high_scores)
high_scores = high_scores[high_scores['total'] >= 200]

type(high_scores)
high_scores.nlargest(10, 'total')   #.nlargest
high_scores1 = high_scores[high_scores['inning']==1]

high_scores2 = high_scores[high_scores['inning']==2]
high_scores1=high_scores1.merge(high_scores2[['match_code','inning', 'total']], on='match_code')

high_scores1.rename(columns={'inning_x':'inning_1','inning_y':'inning_2','total_x':'inning1_runs','total_y':'inning2_runs'},inplace=True)

high_scores1=high_scores1[high_scores1['inning1_runs']>=200]

high_scores1['is_score_chased']=1

high_scores1['is_score_chased'] = np.where(high_scores1['inning1_runs']<=high_scores1['inning2_runs'], 

                                           'yes', 'no')
chances = high_scores1['is_score_chased'].value_counts()

print(chances)
print('The chances of chasing a target of 200+ in 1st innings are : \n' , round(chances[1]/14*100))



#It seems to be clear that team batting first and scoring 200+ runs, has a very high probablity of winning the match.
# Lets see which team were performing outstanding with their respective seasons.

match_wise_data = df_ipl.drop_duplicates(subset = 'match_code', keep='first').reset_index(drop=True)
X=match_wise_data.groupby('year')['winner'].value_counts()

X
for year in range(2008,2017):

    print(year,X[str(year)].idxmax(), X[str(year)].max())
import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



data_ipl = pd.read_csv('../input/ipl_dataset.csv')

data_ipl.head()
data_ipl['year'] = data_ipl['date'].apply(lambda x : x[:4])

no_season=data_ipl['year'].nunique()

print('The number of season : ',no_season)

no_matches=data_ipl.groupby('year')['match_code'].nunique()

type(no_matches)

print('Number of matches:')

print(no_matches)


plt.figure(figsize = (14,8))

no_matches.plot.bar()

plt.title('IPL matches played in years')

plt.xlabel(str('Years').upper())

plt.ylabel(str('No. of matches').upper())

plt.xticks(rotation=45)

plt.show()
import pandas as pd

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

match_wise_data = data_ipl.drop_duplicates(subset='match_code',keep='first').reset_index(drop=True)

#print(match_wise_data)

sns.countplot(x='year',data=data_ipl)

plt.show()
venues = data_ipl.groupby('venue')['match_code'].nunique().sort_values(ascending = False)

type(venues)
plt.figure(figsize = (14,8))

venues.plot.barh() #Horizontal bar plot by  barh()

plt.title('Number of matches in each venue')

plt.xlabel(str('venues').upper())

plt.ylabel(str('No. of matches').upper())

#plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (14,8))

venues.plot.bar()

plt.title('Number of matches in each venue')

plt.xlabel(str('venues').upper())

plt.ylabel(str('No. of matches').upper())

plt.xticks(rotation=90)

plt.show()
sns.countplot(x='venue',data=match_wise_data)

plt.xticks(rotation='vertical')

plt.show()
wins_teams=data_ipl.groupby(['year','winner'])['match_code'].nunique()



wins_teams.unstack().plot(kind='bar', stacked=True, figsize=(15,10)) #for unstacked bar chart

plt.title(str('Number of matches wins by each teams in each seasons').upper())

plt.xlabel(str('Years').upper())

plt.ylabel(str('matches counts').upper())

plt.xticks(rotation=45)

plt.show()
match_wise_data = data_ipl.drop_duplicates(subset='match_code', keep='first').reset_index(drop=True)

total_wins = match_wise_data['winner'].value_counts() 

plot = total_wins.plot(kind='bar', title = "Total no. of wins across seasons 2008-2016", figsize=(7,5))

plt.xticks(fontsize =10 , rotation=90);
temp_data = pd.melt(match_wise_data, id_vars=['match_code', 'year'], value_vars= ['team1', 'team2'])

matches_played = temp_data.value.value_counts()

print(matches_played)
team1 = match_wise_data["team1"]

team2 = match_wise_data["team2"]

matches= pd.concat([team1,team2]).value_counts()

plt.figure(figsize = (14,8))

matches.plot(x= matches.index, y = matches, kind = 'bar')

plt.title(str('No. of matches played across 9 seasons').upper())

plt.xlabel(str('Teams').upper())

plt.ylabel(str('matches counts').upper())

plt.xticks(fontsize = 10, rotation=90)

plt.show()
# Bowlers performance can be judged by categories such 'bowled' and 'caught and bowled'

# subset the dataframe according to above categories

bowled = data_ipl[(data_ipl['wicket_kind']=='bowled')]

bowlers_wickets = bowled.groupby('bowler')['wicket_kind'].count()

#print(bowlers_wickets)

bowlers_wickets.sort_values(ascending = False, inplace = True)

blr=bowlers_wickets[:10]

plt.figure(figsize=(14,8))

plt.title(str('Bowlers with high ratings across 9 seasons').upper())

blr.plot(x= bowlers_wickets.index, y = bowlers_wickets, kind = 'barh', colormap = 'Accent_r');
score_per_venue = data_ipl.loc[:, ['match_code', 'venue', 'inning', 'total']]

average_score_per_venue = score_per_venue.groupby(['match_code', 'venue', 'inning']).agg({'total' : 'sum'}).reset_index()

average_score_per_venue = average_score_per_venue.groupby(['venue', 'inning'])['total'].mean().reset_index()

average_score_per_venue.head()
plt.figure(figsize=(14,8))

x1 = average_score_per_venue[average_score_per_venue['inning']==1]['venue']

y1 = average_score_per_venue[average_score_per_venue['inning']==1]['total']

x2 = average_score_per_venue[average_score_per_venue['inning']==2]['venue']

y2 = average_score_per_venue[average_score_per_venue['inning']==2]['total']

plt.plot( x1, y1, '-b',marker='o',ms=6,lw=2, label = 'inning1')

plt.plot( x2, y2, '-r',marker='o',ms=6,lw=2, label = 'inning2')

plt.legend(loc = 'lower right', fontsize = 15)

plt.title(str('average score per innings across all  stadium').upper())

plt.xticks(fontsize = 15, rotation=90)

plt.xlabel('Venues', fontsize=15)

plt.ylabel('Average runs scored on venues', fontsize=15)

plt.show()
dismissed = data_ipl.groupby(['wicket_kind']).count().reset_index()

dismissed = dismissed[['wicket_kind', 'delivery']].rename(columns={'delivery' : 'count'})

dismissed
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

f.suptitle("Top 5 Dismissal Kind", fontsize=15)



dismissed.plot.bar(ax=ax1, legend = False)

ax1.set_xticklabels(list(dismissed["wicket_kind"]), fontsize=8)



explode =[0.01,0.02,0.1,0.2,0.25,0.4,0.35,0.05,0.05]

properties = ax2.pie(dismissed["count"], labels=None, startangle=90, autopct='%1.1f%%', explode = explode)

ax2.legend(bbox_to_anchor=(1,1), labels=dismissed['wicket_kind'])
runs_data = data_ipl.loc[:,['runs','year']]

boundaries = runs_data[runs_data['runs']==4]

fours = boundaries.groupby('year')['runs'].count()

sixes = runs_data[runs_data['runs']==6]

sixer = sixes.groupby('year')['runs'].count()
plt.figure(figsize=(14,8))

plt.plot(fours.index, fours,'-b',marker='o',ms=6,lw=2, label = 'fours')

plt.plot(sixer.index, sixer,'-r',marker='o',ms=6,lw=2, label = 'sixes')

plt.title(str('Total 4s and 6s scored across seasons').upper())

plt.legend(loc = 'upper right', fontsize = 15)

plt.xticks(fontsize = 15, rotation=90)

plt.xlabel('IPL Seasons', fontsize=15)

plt.ylabel('scored', fontsize=15)

plt.show()
per_match_data = data_ipl.drop_duplicates(subset='match_code', keep='first').reset_index(drop=True)

total_runs_per_season = data_ipl.groupby('year')['total'].sum()

balls_delivered_per_season = data_ipl.groupby('year')['delivery'].count()

no_of_match_played_per_season = per_match_data.groupby('year')['match_code'].count()
avg_runs_per_match = total_runs_per_season/no_of_match_played_per_season

avg_balls_per_match = balls_delivered_per_season/no_of_match_played_per_season

avg_runs_per_ball = total_runs_per_season/balls_delivered_per_season
avg_data = pd.DataFrame([no_of_match_played_per_season, avg_runs_per_match, avg_balls_per_match, avg_runs_per_ball])

avg_data.index =['No.of Matches', 'Average Runs per Match', 'Average balls bowled per match', 'Average runs per ball']

avg_data
plt.figure(figsize=(14,8))

avg_data.T.plot(kind='bar', figsize = (12,10), colormap = 'Paired_r')

plt.title(str('average statistics across seasons').upper(), fontsize=15)

plt.xticks(fontsize = 15, rotation=90)

plt.xlabel('Season', fontsize=15)

plt.ylabel('Average', fontsize=15)

plt.legend(loc=9,ncol=4);
match_wise_data.year = pd.to_numeric(match_wise_data.year) 
import seaborn as sns

#No of wins by team and season in each city

x, y = 2008, 2016

while x <=y:

    wins_per_city = match_wise_data[match_wise_data['year'] == x].groupby(['winner', 'city'])['match_code'].count()

    plot = wins_per_city.unstack().plot(kind='bar', stacked=True, title="Team wins in different cities\nSeason "+str(x), figsize=(7, 5))

    sns.set_palette("Paired", len(match_wise_data['city'].unique()))

    plot.set_xlabel("Teams")

    plot.set_ylabel("No of wins")

    plot.legend(loc='best', prop={'size':8})

    x+=1