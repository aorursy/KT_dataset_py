# Libraries Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# in order to show plots within this notebook
%matplotlib inline

import seaborn as sb
sb.set_style('whitegrid')

# to import data, which is stored in a sqlite3 database
import sqlite3

# database file's name
database = '../input/database.sqlite'

#from IPython.display import Image
#logo = Image(filename='https://i.imgur.com/ydwTfM1.png')
# opening a connection to the database
conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables
leagues = pd.read_sql('SELECT * FROM league where name = \'Italy Serie A\' ', conn)
leagues
serie_a = pd.read_sql('SELECT * FROM Match WHERE league_id = 10257 '
                               ' ORDER BY season, date', conn)
serie_a.head()
possession_col = serie_a.columns.get_loc('possession')
serie_a = serie_a.iloc[:, np.r_[0, 3:possession_col]]
serie_a.head()
# retrieving the team's unique ID
cagliari_id = pd.read_sql('SELECT * FROM Team WHERE team_long_name = \'Cagliari\' ', conn)
cagliari_id
# filtering all Cagliari's matches
cagliari_matches = serie_a.query('home_team_api_id == 8529 or away_team_api_id == 8529')
cagliari_matches.shape
cagliari_matches.info()
cagliari_matches.loc[: , 'date'] =  pd.to_datetime(cagliari_matches['date'], format='%Y-%m-%d')
cagliari_matches.head()
cols_to_correct = ['away_player_X11', 
 'away_player_Y11',                 
'away_player_Y11',
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
'away_player_2',
'away_player_3',
'away_player_4',
'away_player_5',
'away_player_6',
'away_player_7',
'away_player_8',
'away_player_9',
'away_player_10',
'away_player_11']

cagliari_matches.fillna(0, inplace=True)
# float columns to int
for col in cols_to_correct:
    cagliari_matches.loc[:, col] = cagliari_matches[col].astype(int)
def matches_and_scores(team_matches, team_id):
    # list with all the H (home) or A (away) matches
    match_played = [] 
    # list with the scores for the team in that match (3 points for a win, 1 for a draw and 0 for a loss)
    score = []
    for index, row in team_matches.iterrows(): 
        if row['home_team_api_id'] == team_id: 
            match_played.append('H') 
            if row['home_team_goal'] > row['away_team_goal']:
                # home win
                score.append(3)
            elif row['home_team_goal'] == row['away_team_goal']:
                # draw match
                score.append(1)
            else:
                # home defeat
                score.append(0)
        else: 
            match_played.append('A') 
            if row['home_team_goal'] > row['away_team_goal']:
                # away defeat
                score.append(0)
            elif row['home_team_goal'] == row['away_team_goal']:
                # draw
                score.append(1)
            else:
                # away win
                score.append(3)
    return match_played, score
# Cagliari Matches
cagliari_matches = cagliari_matches.assign(home_away = matches_and_scores(cagliari_matches, cagliari_id['team_api_id'].item())[0], score = matches_and_scores(cagliari_matches,  cagliari_id['team_api_id'].item())[1])
# Y => role: goalkeeper, defender, midfielder, forward
y_position = ['home_player_Y1',
 'home_player_Y2',
 'home_player_Y3',
 'home_player_Y4',
 'home_player_Y5',
 'home_player_Y6',
 'home_player_Y7',
 'home_player_Y8',
 'home_player_Y9',
 'home_player_Y10',
 'home_player_Y11',
 'away_player_Y1',
 'away_player_Y2',
 'away_player_Y3',
 'away_player_Y4',
 'away_player_Y5',
 'away_player_Y6',
 'away_player_Y7',
 'away_player_Y8',
 'away_player_Y9',
 'away_player_Y10',
 'away_player_Y11']

# X => area: left, center, right
x_postion = ['home_player_X1',
 'home_player_X2',
 'home_player_X3',
 'home_player_X4',
 'home_player_X5',
 'home_player_X6',
 'home_player_X7',
 'home_player_X8',
 'home_player_X9',
 'home_player_X10',
 'home_player_X11',
 'away_player_X1',
 'away_player_X2',
 'away_player_X3',
 'away_player_X4',
 'away_player_X5',
 'away_player_X6',
 'away_player_X7',
 'away_player_X8',
 'away_player_X9',
 'away_player_X10',
 'away_player_X11']

# player_id for the starting 11
starters = ['home_player_1',
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
 'away_player_11']
import xml.etree.ElementTree as ET
import math

def player_position(match_id, player_id):
    '''
        translating the players coordinates X and Y into categories
        goalkeeper
        defender
        midfielder
        forward
        and position on the field
        right
        center
        left    
    
    '''
    
    player_info = ""
    try:
        if int(player_id):
            # info about that match
            match = serie_a[serie_a.match_api_id == match_id]
            # checking if the player was a starter: in that case we can extract details
            for i in starters:
                player_pos = ""
                starter_num = ""
                if int(match[i]) == int(player_id):
                    player_pos = i
                    starter_num = player_pos.split('_')[2]
                    home_away = player_pos.split('_')[0]
                    x_pos = int(match[player_pos.split('_')[0]+ '_player_X'+ player_pos.split('_')[2]])
                    y_pos = int(match[player_pos.split('_')[0]+ '_player_Y'+ player_pos.split('_')[2]])
                    if x_pos >= 1 and x_pos <= 4:
                        x_pos = 'left'
                    elif x_pos >= 5 and x_pos <= 7:
                        x_pos = 'center'
                    else:
                        x_pos = 'right'
                    if y_pos == 1:
                        y_pos = 'goalkeeper'
                    elif y_pos >= 2 and y_pos <= 5:
                        y_pos = 'defender'
                    elif y_pos >= 6 and y_pos <= 8:
                        y_pos = 'midfielder'
                    else:
                        y_pos = 'forward'
                    break

            # if that player wasn't among the starters, I can only set a generic attribute
            if player_pos == "":
                player_pos = "substitute"
                starter_num = "substitute"
                home_away = player_pos.split('_')[0]
                x_pos = "substitute"
                y_pos = "substitute"

            player_info = [{'player_position': starter_num, 'home_away': home_away, 'x_pos': x_pos, 'y_pos': y_pos}]

    except:
        player_info = ""
        
    return player_info

def goals_details(cag_matches):
    ind = 0
    goals_info = []
    for i, v in cag_matches.goal.iteritems():
        try:
            id_match = cag_matches.match_api_id.iloc[ind]
            e = ET.fromstring(v)   
            for goal in e.findall('value'):
                player_info = ""
                goal_type = goal.find('comment').text
                team = goal.find('team').text
                if goal.find('player1') is not None:
                    player = goal.find('player1').text
                elif goal.find('player') is not None:
                    player = goal.find('player').text
                else:
                    player = np.NaN
                elapsed = goal.find('elapsed').text
                if goal.find('elapsed_plus') is not None:
                    elapsed_plus = goal.find('elapsed_plus').text
                else:
                    elapsed_plus = 0
                if goal.find('subtype') is not None:
                    subtype = goal.find('subtype').text
                else:
                    subtype = np.NaN

                player_info = player_position(id_match, player)                            
                if (len(player_info) == 1):
                    goals_info.append({'match_api_id': id_match, 'goal_type': goal_type, 'team_id': team, 'player_id': player, \
                                  'elapsed_time': elapsed, 'elapsed_plus': elapsed_plus, 'subtype': subtype, \
                                    'x_position': player_info[0]['x_pos'], \
                                      'y_position': player_info[0]['y_pos'], 'home_away': player_info[0]['home_away']})        

        except TypeError as error:
            pass

        ind += 1
        
    return goals_info
# gathering data and saving it in a DataFrame
goals_det = goals_details(cagliari_matches)
goals_cag = pd.DataFrame(data=goals_det)
goals_cag.head()
goals_cag.info()
cols_to_correct = ['elapsed_time', 'elapsed_plus', 'player_id']
for col in cols_to_correct:
    goals_cag[col] = goals_cag[col].astype(int)
season_points = cagliari_matches.groupby('season').score.sum()
lbl_points = np.arange(0, season_points.max()+ 5, 5)
positioning = ['9th place', '10th place', '11th place', '12th place', '13th place', '14th place', '15th place']

plt.figure(figsize=(18, 9));
sb.lineplot(season_points.index, season_points.values);
plt.title('Points at the end of the Season');
plt.xlabel('Season');
plt.yticks(lbl_points, lbl_points);
sb.despine(left=True, bottom=False, right=True, top=True)
score = ['loss', 'drws', 'wins']
yticks = np.arange(0, 22, 1)

# creating the plot
plt.figure(figsize=(18.5, 9));

colors = ['#6E75A8', '#FCD581', '#B0413E']
avg = np.arange(0, 2.2, 0.2)

with sb.color_palette("RdBu", 4):
    sb.countplot(data = cagliari_matches, x = 'season', hue='score')

plt.title('Results Distribution (number of Lost, Draw, Won Matches)')
plt.ylabel('Season', fontsize=12)
plt.ylabel('Matches', fontsize=12)
plt.grid(False);
sb.despine(left=False, bottom=False, right=True)
plt.yticks(yticks, yticks)

# legend
plt.legend(
    labels = score,
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(0.52, 0.92)
);
# to better focus on home and away matches, I've created this function which takes the list of matches for a team
# and returns the count of wins, draws and lost matches along with the share for each of the result types
# and the season for those results

home = cagliari_matches.query('home_away ==\'H\'')
away = cagliari_matches.query('home_away == \'A\'')


def final_result(matches, home_away, season):
    '''
        input:
            dataframe with matches
            a variable specifying whether the team appears as home or away team
            the season those matches were played
        output:
            a dataframe containing
                the number of wins, draws, lost matches
                the share of wins, draws, lost matches compared to the total of matches played
                the season those matches were played
    '''
    results = []
    wins, loss, drws = 0, 0, 0
    for i, v in matches.iterrows():
        if (v.home_team_goal > v.away_team_goal):
            if home_away == 'h':
                wins += 1
            else:
                loss += 1
        elif v.home_team_goal == v.away_team_goal:
            drws += 1
        else:
            if home_away == 'h':
                loss += 1
            else:
                wins += 1
                
    results = pd.DataFrame([['wins', wins, round(wins/matches.shape[0]* 100, 2), season], \
                            ['drws', drws, round(drws/matches.shape[0]* 100, 2), season], \
                            ['loss', loss, round(loss/matches.shape[0]* 100, 2), season]], \
                           columns=['result', 'total', 'share', 'season'])    
    
    
    return results
# splitting matches between home and away
home_results = final_result(home, 'h', 'overall')
away_results = final_result(away, 'a', 'overall')

# creating the plot
fig_wins_loss, ax_wins_loss = plt.subplots(figsize=(18.5, 7.5));
colors = ['#6E75A8', '#FCD581', '#B0413E']
results = ["%s %9s%%" % (lab, siz) for lab, siz in zip(['Wins', 'Drws', 'Loss'], home_results.share)]
explode = (0, 0, 0)

fig_wins_loss.add_subplot(1,2,1)
ax_wins_loss.grid(False);
ax_wins_loss.get_xaxis().set_visible(False)
ax_wins_loss.get_yaxis().set_visible(False)

sb.despine(left=True, bottom=True, right=True)

# pie chart for home matches
plt.pie(home_results.share, explode = explode, shadow=True, startangle=0, colors=colors, radius=1);

plt.legend(
    title = "Home Matches",
    labels = results,
    loc = 'upper right',
    prop = {'size': 13},
    bbox_to_anchor = (0.52, 0.85),
    bbox_transform = fig_wins_loss.transFigure
);

fig_wins_loss.add_subplot(1,2,2)
results = ["%s %9s%%" % (lab, siz) for lab, siz in zip(['Wins', 'Drws', 'Loss'], away_results.share)]

# pie chart for away matches
plt.pie(away_results.share, explode = explode, shadow=True, startangle=0, colors=colors, radius=1);

plt.legend(
    title = "Away Matches",
    labels = results,
    loc = 'upper right',
    prop = {'size': 13},
    bbox_to_anchor = (0.94, 0.85),
    bbox_transform = fig_wins_loss.transFigure
);
yticks = np.arange(0, 13, 1)

# creating the plot
plt.figure(figsize=(18.5, 9));

colors = ['#6E75A8', '#FCD581', '#B0413E']
avg = np.arange(0, 2.2, 0.2)

with sb.color_palette("RdBu", 4):
    sb.countplot(data = home, x = 'season', hue='score')

plt.title('Results Distribution by Season - Home Matches')
plt.grid(False);
sb.despine(left=False, bottom=False, right=True)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Matches', fontsize=12)
plt.yticks(yticks, yticks)

# legend
plt.legend(
    labels = score,
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(0.52, 0.92)
);
yticks = np.arange(0, 13, 1)

# creating the plot
plt.figure(figsize=(18.5, 9));

colors = ['#6E75A8', '#FCD581', '#B0413E']
avg = np.arange(0, 2.2, 0.2)

with sb.color_palette("RdBu", 4):
    sb.countplot(data = away, x = 'season', hue='score')

plt.title('Results Distribution by Season - Away Matches')
plt.grid(False);
sb.despine(left=False, bottom=False, right=True)
plt.yticks(yticks, yticks)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Matches', fontsize=12)

# legend
plt.legend(
    labels = score,
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(0.92, 1)
);
home_avg_pts = home.groupby('season').score.mean()
away_avg_pts = away.groupby('season').score.mean()
avg_points = cagliari_matches.groupby('season').score.mean()

lbl_points = np.arange(0, 3.2, 0.2)

# creating the plot
plt.figure(figsize=(18.5, 9));

sb.lineplot(home_avg_pts.index, home_avg_pts.values, color='b', marker='o', alpha=0.3);
sb.lineplot(away_avg_pts.index, away_avg_pts.values, color='r', marker='o', alpha=0.3);
sb.lineplot(avg_points.index, avg_points.values, color = 'g', marker='o');
plt.title('Average Points per Season');
plt.xlabel('Season');
plt.yticks(lbl_points, np.round(lbl_points, 2));
sb.despine(left=True, bottom=False, right=True, top=True)

# legend
plt.legend(
    labels = ['Home Avg', 'Away Avg', 'Season Avg'],
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(1, 1)
);
home_goals_made = home.groupby('season').home_team_goal.mean()
home_goals_taken = home.groupby('season').away_team_goal.mean()
away_goals_taken = away.groupby('season').home_team_goal.mean()
away_goals_made = away.groupby('season').away_team_goal.mean()

# creating the plot
plt.figure(figsize=(18.5, 7.5));

colors = ['#6E75A8', '#FCD581', '#B0413E']
avg = np.arange(0, 2.2, 0.2)

plt.subplot(1, 2, 1);

sb.lineplot(home_goals_made.index, home_goals_made.values, color='b', marker='o', alpha=0.3);
sb.lineplot(home_goals_taken.index, home_goals_taken.values, color='r', marker='o', alpha=0.3);
plt.yticks(avg, np.round(avg, 2));
plt.xlabel('Season', fontsize=12)
plt.title('Home Matches Goals by Season');
sb.despine(left=False, bottom=False, right=True, top=True);

# legend
plt.legend(
    labels = ['Goals Made', 'Goals Taken'],
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(1, 0.2)
);

plt.subplot(1,2,2);
sb.despine(left=True, bottom=True, right=True);
sb.lineplot(away_goals_made.index, away_goals_made.values, color='b', marker='o', alpha=0.3);
sb.lineplot(away_goals_taken.index, away_goals_taken.values, color='r', marker='o', alpha=0.3);

plt.yticks(avg, np.round(avg, 2));
plt.xlabel('Season', fontsize=12)
plt.title('Away Matches Goals by Season');
goals_info = pd.merge(goals_cag, cagliari_matches[['match_api_id', 'season']], on='match_api_id', how='left')
goals_made_by_pos = goals_info.query('team_id ==\'8529\'')

# filtering out substitute because it doesn't give much of information
goals_made_by_pos = goals_made_by_pos[goals_made_by_pos.y_position !='substitute']

positions = ['forward', 'midfielder', 'defender', 'goalkeeper', 'substitute']

# creating the plot
plt.figure(figsize=(18.5, 7.5));

plt.title('Goals made by role - Overall', fontsize=14);

with sb.color_palette("RdBu", 4):
    sb.countplot(data = goals_made_by_pos, x = 'season', hue='y_position')
    
# legend
plt.legend(
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(0.88, 0.88)
);
plt.grid(axis='x', alpha= 0);
plt.grid(axis='y', alpha= 0.3);
plt.ylabel('Goals Made', fontsize=12);
plt.xticks(fontsize=12);

sb.despine(left=False, bottom=False, right=True)    
goals_info_home_away = goals_info[(goals_info.home_away != 'substitute') & (goals_info.team_id == '8529')]
yticks = np.arange(0, 17, 1)

# creating the plot
plt.figure(figsize=(28.5, 10));
plt.rcParams["axes.labelsize"] = 18

with sb.color_palette("RdBu", 4):
    g = sb.catplot(x="season", hue="y_position", col="home_away", data=goals_info_home_away, kind="count", height=10, aspect=1, legend = False);
    g.set_titles('{col_name}')
    
plt.yticks(yticks, yticks, fontsize=20);


#legend
plt.legend(
   loc = 'best',
   prop={'size': 18},
   bbox_to_anchor=(0.5, 0.88)
);
goals_made_by_area = goals_info.query('team_id ==\'8529\'')
goals_made_by_area = goals_made_by_area[goals_made_by_area.x_position != 'substitute']

#palette = ["#5d8437", "#76d74d","#cbce46"]
yticks = np.arange(0, 40, 2)

# creating the plot
plt.figure(figsize=(18.5, 7.5));
plt.rcParams["axes.labelsize"] = 13


with sb.color_palette("RdBu", 4):
    sb.countplot(data = goals_made_by_area, x = 'season', hue='x_position');

#legend
plt.legend(
   loc = 'best',
   prop={'size': 14},
   bbox_to_anchor=(0.5, 0.92)
);

plt.title('Goals by Position on the field - Overall', fontsize=14)
plt.yticks(yticks, yticks, fontsize=12);
plt.xticks(fontsize=12);
plt.ylabel('Goals Made');
plt.grid(axis='x', alpha= 0);
plt.grid(axis='y', alpha= 0.3);
sb.despine(left=False, bottom=False, right=True)
yticks = np.arange(0, 22, 1)

# creating the plot
plt.figure(figsize=(28.5, 10));
plt.rcParams["axes.labelsize"] = 18

with sb.color_palette("RdBu", 4):
    g = sb.catplot(x="season", hue="x_position", col="home_away", data=goals_made_by_area, kind="count", height=10, aspect=1,\
                   legend = False);
    g.set_titles('{col_name}')

plt.yticks(yticks, yticks);

#legend
plt.legend(
   loc = 'best',
   prop={'size': 18},
   bbox_to_anchor=(0.1, 0.95)
);
goals_taken_by_pos = goals_info.query('team_id !=\'8529\'')

# filtering out substitute because it doesn't give much of information
goals_taken_by_pos = goals_taken_by_pos[goals_taken_by_pos.y_position !='substitute']

positions = ['forward', 'midfielder', 'defender', 'goalkeeper', 'substitute']

# creating the plot
plt.figure(figsize=(18.5, 7.5));

plt.title('Goals taken by role - Overall', fontsize=14);

with sb.color_palette("RdBu", 4):
    sb.countplot(data = goals_taken_by_pos, x = 'season', hue='y_position')
    
# legend
plt.legend(
    #labels = positions,
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(0.55, 1)
);
plt.grid(axis='x', alpha= 0);
plt.grid(axis='y', alpha= 0.3);
plt.ylabel('Goals Taken', fontsize=13);
plt.xlabel('Season', fontsize=13)
sb.despine(left=False, bottom=False, right=True)  
yticks = np.arange(0, 18, 1)

# creating the plot
plt.figure(figsize=(28.5, 10));
plt.rcParams["axes.labelsize"] = 18

with sb.color_palette("RdBu", 4):
    g = sb.catplot(x="season", hue="y_position", col="home_away", data=goals_made_by_pos, kind="count",\
                   height=10, aspect=1, legend = False);
    g.set_titles('{col_name}')

plt.yticks(yticks, yticks);

#legend
plt.legend(
   loc = 'best',
   prop={'size': 18},
   bbox_to_anchor=(0.1, 0.95)
);
goals_taken_by_pos = goals_info.query('team_id !=\'8529\'')

# filtering out substitute because it doesn't give much of information
goals_taken_by_pos = goals_taken_by_pos[goals_taken_by_pos.y_position !='substitute']

positions = ['forward', 'midfielder', 'defender', 'goalkeeper', 'substitute']

# creating the plot
plt.figure(figsize=(18.5, 7.5));

plt.title('Goals taken by position - Overall', fontsize=14);

with sb.color_palette("RdBu", 4):
    sb.countplot(data = goals_taken_by_pos, x = 'season', hue='x_position')
    
# legend
plt.legend(
    loc = 'best',
    prop={'size': 14},
    bbox_to_anchor=(0.55, 1)
);
plt.grid(axis='x', alpha= 0);
plt.grid(axis='y', alpha= 0.3);
plt.ylabel('Goals Taken', fontsize=13);
plt.xlabel('Season', fontsize=13);
sb.despine(left=False, bottom=False, right=True)  
yticks = np.arange(0, 25, 1)

# creating the plot
plt.figure(figsize=(28.5, 10));
with sb.color_palette("RdBu", 4):
    g = sb.catplot(x="season", hue="x_position", col="home_away", data=goals_taken_by_pos, kind="count",\
                   height=10, aspect=1, legend = False);
    g.set_titles('{col_name}')

plt.yticks(yticks, yticks);

#legend
plt.legend(
   loc = 'best',
   prop={'size': 18},
   bbox_to_anchor=(0.1, 1.1)
);
goals_info_copy = goals_info.copy()

# filtering all Cagliari Goals and excluding the "substitute" ones
goals_info_copy = goals_info_copy[(goals_info_copy.x_position != 'substitute') & (goals_info_copy.team_id == '8529')]

xpos = []
ypos = []
for i, v in goals_info_copy.iterrows():
    
    # xpos = 0 left, 1 center, 2 right
    # ypos = 0 GK, 1 DE, 2 MI, 3 FW
    
    if v.x_position == 'center':
        ypos.append(1)
    elif v.x_position == 'right':
        ypos.append(0)
    elif v.x_position == 'left':
        ypos.append(2)
    else:
        ypos.append(3)
    if v.y_position == 'goalkeeper':
        xpos.append(0)
    elif v.y_position == 'defender':
        xpos.append(1)
    elif v.y_position == 'midfielder':
        xpos.append(2)
    elif v.y_position == 'forward':
        xpos.append(3)
    else:
        xpos.append(4)
        

goals_info_copy['xpos'] = xpos
goals_info_copy['ypos'] = ypos
# creating the plot
plt.figure(figsize=(12.5, 6));

pos_y = ['right', 'center', 'left']
pos_x = ['goalkeeper', 'defender', 'midfielder', 'forward']

bins_y = np.arange(0, 3+1, 1)
bins_x = np.arange(0, 4+1, 1)
lbl_y = np.arange(0.3, 3+0.3, 1)
lbl_x = np.arange(0.5, 4+0.3, 1)
plt.hist2d(data=goals_info_copy, x='xpos', y='ypos', cmin = 0.5, cmap = 'viridis_r', bins = [bins_x, bins_y]);
plt.xticks(lbl_x, pos_x, ha = 'left', fontsize=12, position=(-0.3,0));
plt.yticks(lbl_y, pos_y, ha = 'left', fontsize=12, position=(-0.1,0));

plt.colorbar();
cagliari_goals = goals_info.query('team_id ==\'8529\'')
cagliari_goals_time= cagliari_goals.elapsed_time + cagliari_goals.elapsed_plus
bin_edges = np.arange(0, 105, 5)

# creating the plot
plt.figure(figsize=(28.5, 10));
plt.hist(cagliari_goals_time, bin_edges)
plt.xlabel('Minute in the match', fontsize=14)
plt.ylabel('Number of goals', fontsize=14)
plt.xticks(bin_edges, bin_edges, fontsize=14)
plt.yticks(fontsize=14)
plt.axvline(x=45, linewidth=0.5, color='r');
plt.axvline(x=90, linewidth=0.5, color='r');
cagliari_goals_home = cagliari_goals[cagliari_goals.home_away == 'home']
cagliari_goals_away = cagliari_goals[cagliari_goals.home_away == 'away']
cagliari_goals_home_by_time = cagliari_goals_home.elapsed_time + cagliari_goals_home.elapsed_plus
yticks = np.arange(0, 15, 1)
xticks = np.arange(0, 105, 5)

# creating the plot
plt.figure(figsize=(28.5, 10));

plt.hist(cagliari_goals_home_by_time, bin_edges)
plt.title('Goals distribution over time - Home Matches', fontsize=14)
plt.xlabel('Minute in the match', fontsize=12)
plt.ylabel('Number of goals', fontsize=12)
plt.xticks(xticks, xticks)
plt.yticks(yticks, yticks)
plt.axvline(x=45, linewidth=0.5, color='r');
plt.axvline(x=90, linewidth=0.5, color='r');
cagliari_goals_away_by_time = cagliari_goals_away.elapsed_time + cagliari_goals_away.elapsed_plus

# creating the plot
plt.figure(figsize=(28.5, 10));

plt.hist(cagliari_goals_away_by_time, bin_edges)
plt.title('Goals distribution over time - Away Matches', fontsize=14)
plt.xlabel('Minute in the match', fontsize=12)
plt.ylabel('Number of goals', fontsize=12)
plt.xticks(xticks, xticks)
plt.yticks(yticks, yticks)
plt.axvline(x=45, linewidth=0.5, color='r');
plt.axvline(x=90, linewidth=0.5, color='r');
# recovering details about players
players = pd.read_sql('SELECT * FROM Player', conn)

# creating the plot
plt.figure(figsize=(20, 10));
scorers_list = pd.merge(goals_info_copy, players, how='inner', left_on='player_id', right_on='player_api_id')
topscorers = scorers_list.player_name.value_counts()[:8].reset_index()
sb.barplot(data=topscorers, y='index', x='player_name', palette = "rocket");

plt.ylabel('Top Scorers');
plt.xlabel('Goals')
plt.yticks(fontsize=14);
plt.xticks(fontsize=14);