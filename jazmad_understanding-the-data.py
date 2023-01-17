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
print('The number of teams included in the data is %i' % np.shape(teams)[0])
print('The number of players included in the data is %i' % np.shape(player)[0])
# Create summary table of number of matches in each league by year

temp_df = pd.merge(matches,leagues,on='country_id')

temp_df = temp_df.groupby(['season','name']).count()['id_x'].unstack()



#Create bar chart of summary

ax = temp_df.plot.bar(legend=False)

ax.legend(bbox_to_anchor=(1,1))

ax.set_title('Matches within the dataset');

# Try to improve this by creating a stacked bar chart!!
temp_df
min_date = pd.to_datetime(matches['date']).dt.date.min()

print('The earliest match in the dataset is %s/%s/%s' % (min_date.day,min_date.month,min_date.year))
min_date = pd.to_datetime(matches['date']).dt.date.max()

print('The latest match in the dataset is %s/%s/%s' % (min_date.day,min_date.month,min_date.year))
temp_df = matches.copy()

temp_df['month'] = pd.to_datetime(temp_df['date']).dt.month

temp_df = temp_df.groupby('month').count()['id']

ax2 = temp_df.plot.bar()

ax2.set_title('Matches played each month')

ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']);
# Create a dataframe with all home team and away team names and the result

temp_df = matches.copy()[['home_team_api_id','away_team_api_id','home_team_goal','away_team_goal','date']]

temp_df_2 = teams.copy()

temp_df_2['home_team_api_id'] = temp_df_2['team_api_id']

temp_df = temp_df.merge(temp_df_2,on='home_team_api_id',how='outer')

temp_df['home_team'] = temp_df['team_long_name']

temp_df_2['away_team_api_id'] = temp_df_2['team_api_id']

temp_df = temp_df.merge(temp_df_2,on='away_team_api_id',how='outer')

temp_df['away_team'] = temp_df['team_long_name_y']



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

match_results = temp_df[['date','home_team','away_team','result','scoreline','home_points','away_points','total_goals']]
#Check for teams not matched

print ('Out of %i matches (%i teams), there are %i unidentified teams' % (np.shape(matches)[0], 

                                                                          np.shape(matches)[0]*2, 

                                                                          sum(match_results['home_team'].isna()) + sum(match_results['away_team'].isna())))
ax3 = match_results.groupby('result').count()['home_team'].plot.pie(autopct='%i%%');

ax3.set_title('Match results');

ax3.set_ylabel('');
ax4 = match_results.groupby('total_goals').count()['home_team'].plot.bar();

ax4.set_title('Total goals in the match');

ax4.set_ylabel('Matches');
temp_df = match_results.copy()



#Replace infrequent scorelines with Other for readability

not_top20 = temp_df.groupby('scoreline').count().sort_values(by='date',ascending=False).index[20:]

temp_df = temp_df.replace(not_top20,'Other')



ax5 = temp_df.groupby('scoreline').count()['home_team'].sort_values(ascending=False).plot.bar();

ax5.set_title('Most common scorelines');

ax5.set_ylabel('Matches');
temp_df = match_results.copy()

temp_df_2a = temp_df.groupby('home_team').sum()['home_points']

temp_df_2b = temp_df.groupby('home_team').count()['home_points']

temp_df_3a = temp_df.groupby('away_team').sum()['away_points']

temp_df_3b = temp_df.groupby('away_team').count()['home_points']

temp_df_4a = pd.DataFrame(temp_df_2a + temp_df_3a)

temp_df_4b = pd.DataFrame(temp_df_2b + temp_df_3b)

temp_df_4a = temp_df_4a.rename(columns={0:'points'})

temp_df_4b = temp_df_4b.rename(columns={'home_points':'matches_played'})

temp_df_5 = pd.concat([temp_df_4b,temp_df_4a],axis=1)

temp_df_5['avg_points'] = temp_df_5['points'] / temp_df_5['matches_played']



avg_points = temp_df_5.mean()['avg_points']

print('The average number of points earned by a team per match is %3.2f' % avg_points)



top_20 = temp_df_5.loc[(temp_df_5.sort_values(by='avg_points',ascending = False).index[:20])]

fig1 = plt.figure()

ax6a = top_20['matches_played'].plot.bar(alpha=0.3)

ax6b = ax6a.twinx()

ax6b = top_20['avg_points'].plot.line(color='orange')

ax6a.set_title('20 most successful teams');

ax6a.set_xlabel('Team');

ax6a.set_ylabel('Total matches played');

ax6b.set_ylabel('Average points per match');



bottom_20 = temp_df_5.loc[(temp_df_5.sort_values(by='avg_points',ascending = True).index[:20])].sort_values(by='avg_points',ascending = False)

fig2 = plt.figure()

ax7a = bottom_20['matches_played'].plot.bar(alpha=0.3)

ax7b = ax7a.twinx()

ax7b = bottom_20['avg_points'].plot.line(color='orange')

ax7a.set_title('20 least successful teams');

ax7a.set_xlabel('Team');

ax7a.set_ylabel('Total matches played');

ax7b.set_ylabel('Average points per match');
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
temp_df = matches.copy()

temp_dict = {}

for match in temp_df.itertuples():

    for i in range(22):

        if match[56+i] > 0:

            player_id = match[56+i]

            if player_id in temp_dict:

                temp_dict[player_id] = (temp_dict[player_id][0] + match[12+i] , temp_dict[player_id][1] + match[34+i])

            else:

                temp_dict[player_id] = (match[12+i],match[34+i])

                

player_data['co_ords'] = player_data.index.map(temp_dict)

player_data['avg_position'] = player_data.apply(lambda x: (x['co_ords'][0]/x['appearances'],x['co_ords'][1]/x['appearances']),axis=1)

player_data.drop(['co_ords'],axis=1)



def position(co_ords):

    ''' (list of floats) --> string

    

    Returns the position on the field in terms of defender/midfield/attacker

    

    The parameters can be refined.

    

    '''

    

    if math.isnan(co_ords[0]):

        return None

    

    if co_ords == (1.0, 1.0):

        position = 'GK'

    elif co_ords[0] < 4:

        position = 'Def'

    elif co_ords[0] <7:

        position = 'Mid'

    else:

        position = 'Att'

    

    if co_ords[1] < 3:

        side = 'Left'

    elif co_ords[1] > 6:

        side = 'Right'

    else:

        side = 'Centre'

    

    return position



def side(co_ords):

    ''' (list of floats) --> string

    

    Returns the position on the field in terms of left/centre/middle

    

    The parameters can be refined.

    

    '''

    if math.isnan(co_ords[0]):

        return None

    

    if co_ords == (1.0, 1.0):

        side = 'None'  

    elif co_ords[1] < 3:

        side = 'Left'

    elif co_ords[1] > 6:

        side = 'Right'

    else:

        side = 'Centre'

    

    return side



player_data['position'] = player_data['avg_position'].apply(position)

player_data['side'] = player_data['avg_position'].apply(side)



print('The 20 players who have the most appearances (within the data) are:')

player_data.sort_values(by='appearances', ascending=False)[:20][['player_name','appearances', 'position']]
