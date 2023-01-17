from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# Import libraries



import numpy as np

import pandas as pd

from IPython.display import display

pd.options.display.max_columns = None # Displays all columns and when showing dataframes

import sqlite3

import warnings

warnings.filterwarnings("ignore") # Hide warnings

import matplotlib.pyplot as plt

%matplotlib inline

import datetime as dt

import math

import time

from scipy.stats import poisson
# Import the data

'''

#For running on local machine, use:

path = ''   





'''

# For Kaggle kernels, use: 

path = "../input/"





with sqlite3.connect(path+'database.sqlite') as con:

    countries = pd.read_sql_query("SELECT * from Country", con)

    matches = pd.read_sql_query("SELECT * from Match", con)

    leagues = pd.read_sql_query("SELECT * from League", con)

    teams = pd.read_sql_query("SELECT * from Team", con)

    player = pd.read_sql_query("SELECT * from Player",con)

    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes",con)

    sequence = pd.read_sql_query("SELECT * from sqlite_sequence",con)

    team_attributes = pd.read_sql_query("SELECT * from Team_Attributes",con)
temp_df = matches.groupby('country_id').count()

temp_df = countries.merge(leagues,on='id').merge(temp_df,on='country_id')[['name_x','name_y','id_y']]

temp_df.columns = ['Country','League','Matches in Data']

temp_df
print('The number of matches included in the data is %i' % np.shape(matches)[0])
# Create summary table of number of matches in each league by year

temp_df = pd.merge(matches,leagues,on='country_id')

temp_df = temp_df.groupby(['season','name']).count()['id_x'].unstack()



#Create bar chart of summary

ax = temp_df.plot.bar(legend=False, figsize=(10,5))

ax.legend(bbox_to_anchor=(1,1))

ax.set_title('Matches within the dataset');

# Try to improve this by creating a stacked bar chart!!
temp_df
temp_df = pd.merge(matches,leagues,on='country_id')

#temp_df

temp_italy_14_15_df = temp_df[temp_df['name']=='Italy Serie A'][temp_df['season']=='2014/2015']

missing_home_team = temp_italy_14_15_df.groupby('home_team_api_id').count()['id_x'].sort_values()[:1].index[0]

missing_home_team = teams[teams['team_api_id']==missing_home_team]['team_long_name']

missing_home_team = missing_home_team[missing_home_team.index[0]]

missing_away_team = temp_italy_14_15_df.groupby('away_team_api_id').count()['id_x'].sort_values()[:1].index[0]

missing_away_team = teams[teams['team_api_id']==missing_away_team]['team_long_name']

missing_away_team = missing_away_team[missing_away_team.index[0]]

print('The missing match from Serie A\'s 2014/15 season was %s vs %s' % (missing_home_team,missing_away_team))
gw38_matches = len(temp_italy_14_15_df[temp_italy_14_15_df['stage']==38])

print('In Gameweek 38 of the 2014/15 Serie A season, the data includes %i matches' % gw38_matches)
temp_belgium_13_14_df = temp_df[temp_df['name']=='Belgium Jupiler League'][temp_df['season']=='2013/2014']

belgium_13_14_teams = temp_belgium_13_14_df.groupby('home_team_api_id').count().index

belgium_team_1 = teams[teams['team_api_id']==belgium_13_14_teams[0]]['team_long_name']

belgium_team_1 = belgium_team_1[belgium_team_1.index[0]]

belgium_team_2 = teams[teams['team_api_id']==belgium_13_14_teams[1]]['team_long_name']

belgium_team_2 = belgium_team_2[belgium_team_2.index[0]]

belgium_team_3 = teams[teams['team_api_id']==belgium_13_14_teams[2]]['team_long_name']

belgium_team_3 = belgium_team_3[belgium_team_3.index[0]]

belgium_team_4 = teams[teams['team_api_id']==belgium_13_14_teams[3]]['team_long_name']

belgium_team_4 = belgium_team_4[belgium_team_4.index[0]]

print('The four teams in the data from 2013/14 Belgian season are %s, %s, %s and %s.' % (belgium_team_1, belgium_team_2, belgium_team_3, belgium_team_4))
temp_italy_11_12_df = temp_df[temp_df['name']=='Italy Serie A'][temp_df['season']=='2011/2012']

temp_summary = temp_italy_11_12_df.groupby('stage').count()

for week in temp_summary.index[temp_summary['id_x']<10]:

    print('There were %i matches in Serie A 2011/12 Gameweek %i' % (temp_summary['id_x'][week],week))
temp_swiss_11_12_df = temp_df[temp_df['name']=='Switzerland Super League'][temp_df['season']=='2011/2012']

temp_summary = temp_swiss_11_12_df.groupby('away_team_api_id').count()

swiss_missing_team_11_12_count = temp_summary['id_x'].min() * 2

swiss_missing_team_11_12 = temp_summary['id_x'].sort_values().index[0]

swiss_missing_team_11_12 = teams[teams['team_api_id']==swiss_missing_team_11_12]['team_long_name']

swiss_missing_team_11_12 = swiss_missing_team_11_12[swiss_missing_team_11_12.index[0]]

print('In the 2011/12 Swiss season, %s only have %i matches in the data.' % (swiss_missing_team_11_12,swiss_missing_team_11_12_count))
# Create a dataframe with all home team and away team names and the result

temp_df = matches.copy() #for test runs, just take a sample of match data at this line

temp_df_2 = teams.copy()

temp_df_2['home_team_api_id'] = temp_df_2['team_api_id']

temp_df = temp_df.merge(temp_df_2,on='home_team_api_id',how='outer')

temp_df['home_team'] = temp_df['team_long_name']

temp_df_2['away_team_api_id'] = temp_df_2['team_api_id']

temp_df = temp_df.merge(temp_df_2,on='away_team_api_id',how='outer')

temp_df['away_team'] = temp_df['team_long_name_y']

temp_df_3 = temp_df.merge(leagues,on='country_id',how='outer')

temp_df['league_name'] = temp_df_3['name']





def points(goals_scored, goals_conceded):

    ''' (int, int) --> int

    

    Returns 3 points for a win, 1 for a draw and 0 for a loss.

    

    Pre-condition: Goals scored and conceded must be non-negative.

    

    >>> points(3,1)

    3

    

    >>> points(0,0)

    1

    

    >>> points(-1,2)

    None

    

    '''

    

    if goals_scored < 0 or goals_conceded < 0:

        return None

    elif goals_scored > goals_conceded:

        return 3

    elif goals_scored == goals_conceded:

        return 1

    else:

        return 0



# Add the points and result 

temp_df['home_points'] = temp_df.apply(lambda x: (points(x['home_team_goal'],x['away_team_goal'])),axis=1)

temp_df['away_points'] = temp_df.apply(lambda x: (points(x['away_team_goal'],x['home_team_goal'])),axis=1)

temp_df['scoreline'] = temp_df.apply(lambda x: (str(x['home_team_goal'])+'-'+str(x['away_team_goal'])),axis=1)

temp_df['total_goals'] = temp_df.apply(lambda x: (x['home_team_goal']+x['away_team_goal']),axis=1)



def result(home_points, total_goals):

    ''' (int) --> str

    

    Returns the result, based on the points won by the home team and the total goals

     

    >>> result(3,1)

    'Home win'

    

    >>> points(1,0)

    'No score draw'

    

    >>> points(1,2)

    'Score draw'

    

    >>> points(0,2)

    'Home loss'

    

    '''

    

    if home_points == 3:

        return 'Home win'

    elif home_points == 0:

        return 'Home loss'

    else:

        if total_goals == 0:

            return 'No score draw'

        else:

            return 'Score draw'





temp_df['result'] = temp_df.apply(lambda x: (result(x['home_points'],x['total_goals'])),axis=1)

#temp_df

match_results = temp_df
# Need to create player database in order to determine profile of starting lineups

temp_df = matches[['home_player_1',

                  'home_player_2',

                  'home_player_3',

                  'home_player_4',

                  'home_player_5',

                  'home_player_6',

                  'home_player_7',

                  'home_player_8',

                  'home_player_9',

                  'home_player_10',

                  'home_player_11',

                  'away_player_1',

                  'away_player_2',

                  'away_player_3',

                  'away_player_4',

                  'away_player_5',

                  'away_player_6',

                  'away_player_7',

                  'away_player_8',

                  'away_player_9',

                  'away_player_10',

                  'away_player_11'

                  ]]



temp_df_2 = pd.DataFrame(temp_df.apply(pd.value_counts).fillna(0).sum(axis=1),columns=['appearances'])

temp_df = player.copy()

temp_df = temp_df.set_index('player_api_id')

temp_df['appearances'] = 0

temp_df['appearances'][temp_df_2.index] = temp_df_2['appearances']

player_data = temp_df[['player_name','birthday','height','weight','appearances']]
# Add team player data for each match

# This section takes a long time to run (c25 minutes)

t0 = time.time()

home_age_dict = {}

away_age_dict = {}

home_height_dict= {}

away_height_dict = {}

home_weight_dict = {}

away_weight_dict = {}



for match in match_results.itertuples():

    match_api_id = match[7]

    date = pd.to_datetime(match[6])

    home_matched = 0

    away_matched = 0

    home_total_age = 0

    away_total_age = 0

    home_total_height = 0

    away_total_height = 0

    home_total_weight = 0

    away_total_weight = 0

    for i in range(22):

        if match[56+i] > 0:

            player_id = match[56+i]

            if i < 12:

                home_matched += 1

                home_total_age += (date - pd.to_datetime(player_data.ix[player_id]['birthday'])).days / 365.25

                home_total_height += player_data.ix[player_id]['height']

                home_total_weight += player_data.ix[player_id]['weight']

            else:

                away_matched += 1

                away_total_age += (date - pd.to_datetime(player_data.ix[player_id]['birthday'])).days / 365.25

                away_total_height += player_data.ix[player_id]['height']

                away_total_weight += player_data.ix[player_id]['weight']

    if home_matched > 0:

        home_age_dict[match_api_id] = home_total_age / home_matched

        home_height_dict[match_api_id] = home_total_height / home_matched

        home_weight_dict[match_api_id] = home_total_weight / home_matched

    if away_matched > 0:

        away_age_dict[match_api_id] = away_total_age / away_matched

        away_height_dict[match_api_id] = away_total_height / away_matched

        away_weight_dict[match_api_id] = away_total_weight / away_matched



match_results = match_results.set_index('match_api_id')

match_results['home_age'] = match_results.index.map(home_age_dict)

match_results['away_age'] = match_results.index.map(away_age_dict)

match_results['home_height'] = match_results.index.map(home_height_dict)

match_results['away_height'] = match_results.index.map(away_height_dict)

match_results['home_weight'] = match_results.index.map(home_weight_dict)

match_results['away_weight'] = match_results.index.map(away_weight_dict)

t1=time.time()

print('Profile of starting 11 created - time taken to run this step %i minutes and %i seconds' % ((t1-t0)//60,(t1-t0)%60))
# Add recent form data for each match

# This section takes a very long time to run (c2 hours)



t0 = time.time()

home_points_dict = {}

away_points_dict = {}

home_goals_for_dict= {}

away_goals_for_dict = {}

home_goals_against_dict = {}

away_goals_against_dict = {}



seasons = match_results['season'].unique()

leagues = match_results['league_id'].unique()



for season in seasons:

    for league in leagues:

        match_results_temp = match_results[(match_results['season']==season) & (match_results['league_id']==league)]



        for match in match_results_temp.itertuples():

            match_api_id = match[0]

            date = pd.to_datetime(match[6])

            season = match[4]

            home_team = match[7]

            away_team = match[8]

            home_team_recent = match_results_temp[(match_results_temp['season']==season) & 

                                                  (match_results_temp['date'].apply(pd.to_datetime)<date) &

                                                  ((match_results_temp['home_team_api_id_x']==home_team) | (match_results_temp['away_team_api_id']==home_team))

                                                 ].sort_values(by='date',ascending=False).head(6)

            home_points = home_team_recent['home_points'][home_team_recent['home_team_api_id_x']==home_team].sum() + home_team_recent['away_points'][home_team_recent['away_team_api_id']==home_team].sum()

            home_goals_for = home_team_recent['home_team_goal'][home_team_recent['home_team_api_id_x']==home_team].sum() + home_team_recent['away_team_goal'][home_team_recent['away_team_api_id']==home_team].sum()

            home_goals_against = home_team_recent['away_team_goal'][home_team_recent['home_team_api_id_x']==home_team].sum() + home_team_recent['home_team_goal'][home_team_recent['away_team_api_id']==home_team].sum()

            away_team_recent = match_results_temp[(match_results_temp['season']==season) &

                                                  (match_results_temp['date'].apply(pd.to_datetime)<date) &

                                                  ((match_results_temp['home_team_api_id_x']==away_team) | (match_results_temp['away_team_api_id']==away_team))

                                                 ].sort_values(by='date',ascending=False).head(6)

            away_points = away_team_recent['home_points'][away_team_recent['home_team_api_id_x']==away_team].sum() + away_team_recent['away_points'][away_team_recent['away_team_api_id']==away_team].sum()

            away_goals_for = away_team_recent['home_team_goal'][away_team_recent['home_team_api_id_x']==away_team].sum() + away_team_recent['away_team_goal'][away_team_recent['away_team_api_id']==away_team].sum()

            away_goals_against = away_team_recent['away_team_goal'][away_team_recent['home_team_api_id_x']==away_team].sum() + away_team_recent['home_team_goal'][away_team_recent['away_team_api_id']==away_team].sum()

    

            if len(home_team_recent)>2: #Do not include form for the first two matches of the season

                n = len(home_team_recent)

                home_points_dict[match_api_id] = home_points / n

                home_goals_for_dict[match_api_id] = home_goals_for / n

                home_goals_against_dict[match_api_id] = home_goals_against / n

            if len(away_team_recent)>2: #Do not include form for the first three matches of the season

                n = len(away_team_recent)

                away_points_dict[match_api_id] = away_points / n

                away_goals_for_dict[match_api_id] = away_goals_for / n

                away_goals_against_dict[match_api_id] = away_goals_against / n

        

match_results['home_team_form'] = match_results.index.map(home_points_dict)

match_results['away_team_form'] = match_results.index.map(away_points_dict)

match_results['home_team_goals_for'] = match_results.index.map(home_goals_for_dict)

match_results['away_team_goals_for'] = match_results.index.map(away_goals_for_dict)

match_results['home_team_goals_against'] = match_results.index.map(home_goals_against_dict)

match_results['away_team_goals_against'] = match_results.index.map(away_goals_against_dict)

t1=time.time()

print('Team form statistics created - time taken to run this step %i minutes and %i seconds' % ((t1-t0)//60,(t1-t0)%60))
# Create some additional columns



match_results['age_difference'] = match_results['home_age'] - match_results['away_age']

match_results['height_difference'] = match_results['home_height'] - match_results['away_height']

match_results['weight_difference'] = match_results['home_weight'] - match_results['away_weight']

match_results['average_age'] = (match_results['home_age'] + match_results['away_age']) / 2

match_results['average_height'] = (match_results['home_height'] + match_results['away_height']) / 2

match_results['average_weight'] = (match_results['home_weight'] + match_results['away_weight']) / 2

match_results['home_team_recent_goals'] = match_results['home_team_goals_for'] + match_results['home_team_goals_against']

match_results['away_team_recent_goals'] = match_results['away_team_goals_for'] + match_results['away_team_goals_against']

match_results['combined_recent_goals'] = match_results['home_team_recent_goals'] + match_results['away_team_recent_goals']

match_results['combined_form'] = (match_results['home_team_form'] + match_results['away_team_form']) / 2





match_results['home_team_avg_position_X'] = (match_results['home_player_X1'] + 

                                          match_results['home_player_X2'] + 

                                          match_results['home_player_X3'] + 

                                          match_results['home_player_X4'] + 

                                          match_results['home_player_X5'] + 

                                          match_results['home_player_X6'] + 

                                          match_results['home_player_X7'] + 

                                          match_results['home_player_X8'] + 

                                          match_results['home_player_X9'] + 

                                          match_results['home_player_X10'] + 

                                          match_results['home_player_X11'] 

                                          ) / 11

match_results['home_team_avg_position_Y'] = (match_results['home_player_Y1'] + 

                                          match_results['home_player_Y2'] + 

                                          match_results['home_player_Y3'] + 

                                          match_results['home_player_Y4'] + 

                                          match_results['home_player_Y5'] + 

                                          match_results['home_player_Y6'] + 

                                          match_results['home_player_Y7'] + 

                                          match_results['home_player_Y8'] + 

                                          match_results['home_player_Y9'] + 

                                          match_results['home_player_Y10'] + 

                                          match_results['home_player_Y11'] 

                                          ) / 11

match_results['away_team_avg_position_X'] = (match_results['away_player_X1'] + 

                                          match_results['away_player_X2'] + 

                                          match_results['away_player_X3'] + 

                                          match_results['away_player_X4'] + 

                                          match_results['away_player_X5'] + 

                                          match_results['away_player_X6'] + 

                                          match_results['away_player_X7'] + 

                                          match_results['away_player_X8'] + 

                                          match_results['away_player_X9'] + 

                                          match_results['away_player_X10'] + 

                                          match_results['away_player_X11'] 

                                          ) / 11

match_results['away_team_avg_position_Y'] = (match_results['away_player_Y1'] + 

                                          match_results['away_player_Y2'] + 

                                          match_results['away_player_Y3'] + 

                                          match_results['away_player_Y4'] + 

                                          match_results['away_player_Y5'] + 

                                          match_results['away_player_Y6'] + 

                                          match_results['away_player_Y7'] + 

                                          match_results['away_player_Y8'] + 

                                          match_results['away_player_Y9'] + 

                                          match_results['away_player_Y10'] + 

                                          match_results['away_player_Y11'] 

                                          ) / 11
avg_goals_per_game = match_results['total_goals'].mean()

print('The average number of goals per game for matches in the data is %3.2f' % avg_goals_per_game)



temp_df = match_results.groupby('total_goals').count()['home_team']

ax1 = temp_df.plot.bar(figsize=(10,5));

ax1.set_title('Total goals in the match');

ax1.set_ylabel('Matches');

pdf_poisson = poisson.pmf(temp_df.index,avg_goals_per_game)

preds = pdf_poisson * np.shape(matches)[0]

ax1 = plt.plot(temp_df.index,preds,color='orange', linewidth=3)
# Boxplots of various continuous variables



fig, ax = plt.subplots(4, 3, figsize=(15, 15))



match_results.boxplot(column='age_difference', by='total_goals', ax = ax[0,0]);

match_results.boxplot(column='weight_difference', by='total_goals', ax = ax[0,1]);

match_results.boxplot(column='height_difference', by='total_goals', ax = ax[0,2]);

match_results.boxplot(column='average_age', by='total_goals', ax = ax[1,0]);

match_results.boxplot(column='average_weight', by='total_goals', ax = ax[1,1]);

match_results.boxplot(column='average_height', by='total_goals', ax = ax[1,2]);

match_results.boxplot(column='home_team_recent_goals', by='total_goals', ax = ax[2,0]);

match_results.boxplot(column='away_team_recent_goals', by='total_goals', ax = ax[2,1]);

match_results.boxplot(column='combined_recent_goals', by='total_goals', ax = ax[2,2]);

match_results.boxplot(column='home_team_form', by='total_goals', ax = ax[3,0]);

match_results.boxplot(column='away_team_form', by='total_goals', ax = ax[3,1]);

match_results.boxplot(column='combined_form', by='total_goals', ax = ax[3,2]);



for i in range(4):

    for j in range(3):

        ax[i,j].set_xlabel('');
fig, ax = plt.subplots(1, 2, figsize=(16, 5))



temp_pt = match_results.pivot_table(values=['id_x'],

                      index='league_name',

                      columns='total_goals',

                      aggfunc='count')



temp_pt = 100 * temp_pt.div(temp_pt.sum(1), axis = 0);



ax1a = temp_pt.plot(kind='bar', stacked=True, legend=False, ax=ax[0]);

ax1a.set_title('Goals per game in each league');

ax1a.set_ylabel('Proportion of matches (%)');

ax1b = ax1a.twinx();

temp_df = match_results.groupby('league_name')['total_goals'].mean();

ax1b = temp_df.plot(kind='line', legend=False, color='black', linewidth=3);

ax1b.set_ylabel('Average goals per match');



count = match_results.groupby('league_name')['id_x'].count()

temp_995=[]

temp_005=[]

for number in count:

    temp_995.append(poisson.ppf(0.995,number*avg_goals_per_game)/number)

    temp_005.append(poisson.ppf(0.005,number*avg_goals_per_game)/number)



ax1b.plot(temp_995,linewidth=2,color='orange')

ax1b.plot(temp_005,linewidth=2,color='orange')





temp_pt_2 = match_results.pivot_table(values=['id_x'],

                      index='season',

                      columns='total_goals',

                      aggfunc='count')



temp_pt_2 = 100 * temp_pt_2.div(temp_pt_2.sum(1), axis = 0);



ax2a = temp_pt_2.plot(kind='bar', stacked=True, legend=False, ax=ax[1]);

ax2a.set_title('Goals per game in each season');

ax2a.set_ylabel('Proportion of matches (%)');

ax2b = ax2a.twinx();

temp_df_2 = match_results.groupby('season')['total_goals'].mean();

ax2b = temp_df_2.plot(kind='line', legend=False, color='black', linewidth=3);

ax2b.set_ylabel('Average goals per match');



count_2 = match_results.groupby('season')['id_x'].count()

temp_995_2=[]

temp_005_2=[]

for number in count_2:

    temp_995_2.append(poisson.ppf(0.995,number*avg_goals_per_game)/number)

    temp_005_2.append(poisson.ppf(0.005,number*avg_goals_per_game)/number)



ax2b.plot(temp_995_2,linewidth=2,color='orange');

ax2b.plot(temp_005_2,linewidth=2,color='orange');





plt.figure(figsize=(15,10))

temp_df2 = match_results.groupby('stage')['id_x'].count();

ax3b = temp_df2.plot(kind='bar', legend=False, alpha=0.2);

ax3b.set_ylabel('Number of matches');

temp_df = np.array(match_results.groupby('stage')['total_goals'].mean());

ax3a = ax3b.twinx();

ax3a.plot(temp_df, color='black', linewidth=3);

ax3a.set_ylabel('Average goals per match');

plt.title('Average number of goals over a season')

temp_995=[]

temp_005=[]

for number in temp_df2:

    temp_995.append(poisson.ppf(0.995,number*avg_goals_per_game)/number)

    temp_005.append(poisson.ppf(0.005,number*avg_goals_per_game)/number)



ax3a.plot(temp_995,linewidth=2,color='orange');

ax3a.plot(temp_005,linewidth=2,color='orange');

plt.xticks(range(38));

plt.xlim(-1,38);



gw8_prob = 1 -poisson.cdf(temp_df[7]*temp_df2[8],avg_goals_per_game*temp_df2[7])

gw38_prob = 1 - poisson.cdf(temp_df[37]*temp_df2[38],avg_goals_per_game*temp_df2[37])



print('The probability of the Gameweek 8 observation, based on a Poisson random variable, is %3.4f%%' % (gw8_prob * 100))

print('The probability of the Gameweek 38 observation, based on a Poisson random variable, is %3.4f%%' % (gw38_prob * 100))
# Determine probability of final GW average given average of all other gameweeks (based on Poisson)

# Will help illustrate whether impact of GW38 is specific to this GW or whether final stage of season is indicator

match_results['league_season'] = match_results['league_name'] + match_results['season']

stages = dict(match_results.groupby('league_season').max()['stage'])



match_results['stages_per_season'] = match_results['league_season'].map(stages)

match_results['last_day'] = match_results['stage']==match_results['stages_per_season']



avg_goals_leagues = match_results[match_results['last_day']==False].groupby('league_name').mean()['total_goals']

matches_last_day = match_results[match_results['last_day']==True].groupby('league_name').count()['id_x']

goals_last_day = match_results[match_results['last_day']==True].groupby('league_name').mean()['total_goals']

temp_df = pd.concat([avg_goals_leagues,matches_last_day,goals_last_day],axis=1)

temp_df.columns = ['avg_goal_season','matches_last_day','avg_goals_last_day']

temp_df['975_percentile'] = poisson.ppf(0.975,temp_df['matches_last_day']*temp_df['avg_goal_season'])/temp_df['matches_last_day']

temp_df['025_percentile'] = poisson.ppf(0.025,temp_df['matches_last_day']*temp_df['avg_goal_season'])/temp_df['matches_last_day']

temp_df[['avg_goal_season','975_percentile','025_percentile']].plot(color=['black','orange','orange'], linewidth=3, figsize=(15,10), legend=False);

temp_df['avg_goals_last_day'].plot(marker='*',linewidth=0,markersize=10,legend=True);

plt.title('Average goals on the last day of the season');

plt.fill_between(x=temp_df.index,y1=temp_df['025_percentile'],y2=temp_df['975_percentile'],color='orange',alpha=0.1);

plt.xticks(ticks=range(len(temp_df)),labels=temp_df.index,rotation=45);

plt.ylabel('Average goals per game');

plt.xlim((-0.5,len(temp_df)-0.5));
