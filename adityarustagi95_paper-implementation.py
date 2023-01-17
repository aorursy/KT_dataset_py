# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import timedelta

from datetime import datetime

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/cricsheet/ball_by_ball_player_data.csv')

odi = data[data.ODI == 1]

odi.index = range(len(odi))

index = list(odi[['Team 1', 'Team 2', 'Date']].drop_duplicates().index)
odi_matches = {}

for items in tqdm(range(len(index) - 1)) :

    temp_df = odi.loc[index[items]:index[items + 1] - 1, :]

    odi_matches[items] = {}

    odi_matches[items]['Team 1'] = temp_df['Team 1'].unique()[0]

    odi_matches[items]['Team 2'] = temp_df['Team 2'].unique()[0]

    odi_matches[items]['Gender'] = temp_df['Gender'].unique()[0]

    odi_matches[items]['Date'] = temp_df['Date'].unique()[0]

    odi_matches[items]['Venue'] = temp_df['Venue'].unique()[0]

    odi_matches[items]['City'] = temp_df['City'].unique()[0]

    odi_matches[items]['Toss Decision'] = temp_df['Toss Decision'].unique()[0]

    odi_matches[items]['Toss Winner'] = temp_df['Toss Winner'].unique()[0]

    odi_matches[items]['Winner'] = temp_df['Winner'].unique()[0]

    odi_matches[items]['Primary Umpire'] = temp_df['Primary Umpire'].unique()[0]

    odi_matches[items]['Secondary Umpire'] = temp_df['Secondary Umpire'].unique()[0]

    odi_matches[items]['TV Umpire'] = temp_df['TV Umpire'].unique()[0]

    odi_matches[items]['DF'] = temp_df.drop(['Team 1', 'Team 2', 'Gender', 'Season', 'Date', 'Venue', 'City',

                                             'Toss Decision', 'Toss Winner', 'Primary Umpire', 'Secondary Umpire',

                                             'TV Umpire', 'Winner', 'Winner Runs'], axis = 1)
for index, d in tqdm(odi_matches.items()):

    team_1 = d['Team 1']

    team_2 = d['Team 2']

    temp_df = d['DF']

    j = 1

    for items in [team_1, team_2] :

        li = []

        for idx, row in temp_df[temp_df['Batting Team'] == items].iterrows():

            li_temp = row[['Primary Player', 'Secondary Player']].tolist()

            for i in li_temp :

                if i not in li :

                    li.append(i)

        for idx, row in temp_df[temp_df['Batting Team'] != items].iterrows():

            li_temp = row[['Bowler']].tolist()

            for i in li_temp :

                if i not in li :

                    li.append(i)

        odi_matches[index]['Team ' + str(j) + ' Lineup'] = li

        j += 1
player_dict = {}

for index, d in tqdm(odi_matches.items()):

    a = d['Team 1']

    b = d['Team 2']

    team1 = d['Team 1 Lineup']

    team2 = d['Team 2 Lineup']

    df = d['DF']

    for team in [team1, team2] :

        for players in team :

            li = [index]

            #Team

            li.append(a if players in team1 else b)

            

            balls_played = df[df['Primary Player'] == players]['Run'].count()

            #Batting Innings

            li.append(1 if balls_played > 0 else 0)

            #Runs Scored

            li.append(df[df['Primary Player'] == players]['Run'].sum())

            #Balls Played

            li.append(balls_played)

            #Extras Played

            li.append(df[df['Primary Player'] == players][df['No Ball'] > 0].count().Run)

            

            balls_bowled = df[df['Bowler'] == players].count().Run

            #Bowling_innings

            li.append(1 if balls_bowled > 0 else 0)

            #Wickets Taken

            li.append(df[df['Bowler'] == players]['Wicket'].sum())

            #Runs Given

            li.append(df[df['Bowler'] == players]['Run'].sum())

            #Balls Bowled

            li.append(balls_bowled)

            #Extras Given

            li.append(df[df['Bowler'] == players]['No Ball'].sum())

            #venue

            li.append(d['Venue'])

            #date

            li.append(d['Date'])

            #against

            li.append(a if not players in team1 else b)

            

            if players in player_dict :

                player_dict[players].append(li)

            else :

                player_dict[players] = [li]
columns = ['Match_ID', 'Team', 'Batting_Innings', 'Runs_Scored', 'Balls_Played', 'Extras_Played', 'Bowling_Innings', 'Wickets_Taken', 'Runs_Given',

           'Balls_Bowled', 'Extras_Given', 'Venue', 'Date', 'Against']

for key, _ in tqdm(player_dict.items()) :

    player_dict[key] = pd.DataFrame(player_dict[key], columns=columns)

    player_dict[key]['Date'] = pd.to_datetime(player_dict[key]['Date'])

    player_dict[key] = player_dict[key].sort_values('Date')

    player_dict[key].index = range(len(player_dict[key]))
raw_dfs = {}

lineups = {}

for key, values in odi_matches.items():

    raw_dfs[key] = values['DF']

    lineups[key] = [values['Team 1 Lineup'], values['Team 2 Lineup']]

    del(values['Team 1 Lineup'])

    del(values['Team 2 Lineup'])

    del(values['DF'])
def encode(stat, attrib_type) :

    

    #  FOR TOTAL {} INNINGS

    for items in ['batting', 'balling'] :

        if attrib_type == 'consistency' :

            if stat['Total_{}_innings'.format(items)] >= 1 and stat['Total_{}_innings'.format(items)] <= 49 :  stat['Total_{}_innings'.format(items)] = 1

            elif stat['Total_{}_innings'.format(items)] >= 50 and stat['Total_{}_innings'.format(items)] <= 99 :  stat['Total_{}_innings'.format(items)] = 2

            elif stat['Total_{}_innings'.format(items)] >= 100 and stat['Total_{}_innings'.format(items)] <= 124 :  stat['Total_{}_innings'.format(items)] = 3

            elif stat['Total_{}_innings'.format(items)] >= 125 and stat['Total_{}_innings'.format(items)] <= 149 :  stat['Total_{}_innings'.format(items)] = 4

            elif stat['Total_{}_innings'.format(items)] >= 150 : stat['Total_{}_innings'.format(items)] = 5





        elif attrib_type == 'form' :

            if stat['Total_{}_innings'.format(items)] >= 1 and stat['Total_{}_innings'.format(items)] <= 4 :  stat['Total_{}_innings'.format(items)] = 1

            elif stat['Total_{}_innings'.format(items)] >= 5 and stat['Total_{}_innings'.format(items)] <= 9 :  stat['Total_{}_innings'.format(items)] = 2

            elif stat['Total_{}_innings'.format(items)] >= 10 and stat['Total_{}_innings'.format(items)] <= 11 :  stat['Total_{}_innings'.format(items)] = 3

            elif stat['Total_{}_innings'.format(items)] >= 12 and stat['Total_{}_innings'.format(items)] <= 14 :  stat['Total_{}_innings'.format(items)] = 4

            elif stat['Total_{}_innings'.format(items)] >= 15 : stat['Total_{}_innings'.format(items)] = 5



        elif attrib_type == 'opposition' :

            if stat['Total_{}_innings'.format(items)] >= 1 and stat['Total_{}_innings'.format(items)] <= 2 :  stat['Total_{}_innings'.format(items)] = 1

            elif stat['Total_{}_innings'.format(items)] >= 3 and stat['Total_{}_innings'.format(items)] <= 4 :  stat['Total_{}_innings'.format(items)] = 2

            elif stat['Total_{}_innings'.format(items)] >= 5 and stat['Total_{}_innings'.format(items)] <= 6 :  stat['Total_{}_innings'.format(items)] = 3

            elif stat['Total_{}_innings'.format(items)] >= 7 and stat['Total_{}_innings'.format(items)] <= 9 :  stat['Total_{}_innings'.format(items)] = 4

            elif stat['Total_{}_innings'.format(items)] >= 10 : stat['Total_{}_innings'.format(items)] = 5



        elif attrib_type == 'venue' :

            if stat['Total_{}_innings'.format(items)] == 1 :  stat['Total_{}_innings'.format(items)] = 1

            elif stat['Total_{}_innings'.format(items)] == 2 :  stat['Total_{}_innings'.format(items)] = 2

            elif stat['Total_{}_innings'.format(items)] == 3 :  stat['Total_{}_innings'.format(items)] = 3

            elif stat['Total_{}_innings'.format(items)] == 4 :  stat['Total_{}_innings'.format(items)] = 4

            elif stat['Total_{}_innings'.format(items)] >= 5 : stat['Total_{}_innings'.format(items)] = 5

                

    # FOR BATTING AVG

    if stat['batting_avg'] >= 0 and stat['batting_avg'] <= 9.99 :  stat['batting_avg'] = 1

    elif stat['batting_avg'] >= 10 and stat['batting_avg'] <= 19.99 :  stat['batting_avg'] = 2

    elif stat['batting_avg'] >= 20 and stat['batting_avg'] <= 29.99 :  stat['batting_avg'] = 3

    elif stat['batting_avg'] >= 30 and stat['batting_avg'] <= 39.99 :  stat['batting_avg'] = 4

    elif stat['batting_avg'] >= 40 : stat['batting_avg'] = 5

            

    # FOR BATTING SR

    if stat['batting_SR'] >= 0 and stat['batting_SR'] <= 49.99 :  stat['batting_SR'] = 1

    elif stat['batting_SR'] >= 50 and stat['batting_SR'] <= 59.99 :  stat['batting_SR'] = 2

    elif stat['batting_SR'] >= 60 and stat['batting_SR'] <= 79.99 :  stat['batting_SR'] = 3

    elif stat['batting_SR'] >= 80 and stat['batting_SR'] <= 100 :  stat['batting_SR'] = 4

    elif stat['batting_SR'] >= 100 : stat['batting_SR'] = 5

            

    # FOR CENTURIES

    if attrib_type == 'consistency' :

        if stat['Centuries'] >= 1 and stat['Centuries'] <= 4 :  stat['Centuries'] = 1

        elif stat['Centuries'] >= 5 and stat['Centuries'] <= 9 :  stat['Centuries'] = 2

        elif stat['Centuries'] >= 10 and stat['Centuries'] <= 14 :  stat['Centuries'] = 3

        elif stat['Centuries'] >= 15 and stat['Centuries'] <= 19 :  stat['Centuries'] = 4

        elif stat['Centuries'] >= 20 : stat['Centuries'] = 5



    elif attrib_type == 'form' :

        if stat['Centuries'] == 1 :  stat['Centuries'] = 1

        elif stat['Centuries'] == 2 :  stat['Centuries'] = 2

        elif stat['Centuries'] == 3 :  stat['Centuries'] = 3

        elif stat['Centuries'] == 4 :  stat['Centuries'] = 4

        elif stat['Centuries'] == 5 : stat['Centuries'] = 5



    elif attrib_type == 'opposition' :

        if stat['Centuries'] == 1 :  stat['Centuries'] = 3

        elif stat['Centuries'] == 2 :  stat['Centuries'] = 4

        elif stat['Centuries'] >= 3 :  stat['Centuries'] = 5



    elif attrib_type == 'venue' :

        if stat['Centuries'] == 1 :  stat['Centuries'] = 4

        elif stat['Centuries'] >= 2 :  stat['Centuries'] = -5

            

    # FOR FIFTIES

    if attrib_type == 'consistency' :

        if stat['Fifties'] >= 1 and stat['Fifties'] <= 9 :  stat['Fifties'] = 1

        elif stat['Fifties'] >= 10 and stat['Fifties'] <= 19 :  stat['Fifties'] = 2

        elif stat['Fifties'] >= 20 and stat['Fifties'] <= 29 :  stat['Fifties'] = 3

        elif stat['Fifties'] >= 30 and stat['Fifties'] <= 39 :  stat['Fifties'] = 4

        elif stat['Fifties'] >= 40 : stat['Fifties'] = 5





    elif attrib_type == 'form' or attrib_type == 'opposition' :

        if stat['Fifties'] >= 1 and stat['Fifties'] <= 2 :  stat['Fifties'] = 1

        elif stat['Fifties'] >= 3 and stat['Fifties'] <= 4 :  stat['Fifties'] = 2

        elif stat['Fifties'] >= 5 and stat['Fifties'] <= 6 :  stat['Fifties'] = 3

        elif stat['Fifties'] >= 7 and stat['Fifties'] <= 9 :  stat['Fifties'] = 4

        elif stat['Fifties'] >= 10 : stat['Fifties'] = 5



    elif attrib_type == 'venue' :

        if stat['Fifties'] == 1 :  stat['Fifties'] = 4

        elif stat['Fifties'] >= 2 :  stat['Fifties'] = -5

            

    # FOR ZEROS

    if attrib_type == 'consistency' :

        if stat['Zeros'] >= 1 and stat['Zeros'] <= 4 :  stat['Zeros'] = 1

        elif stat['Zeros'] >= 5 and stat['Zeros'] <= 9 :  stat['Zeros'] = 2

        elif stat['Zeros'] >= 10 and stat['Zeros'] <= 14 :  stat['Zeros'] = 3

        elif stat['Zeros'] >= 15 and stat['Zeros'] <= 19 :  stat['Zeros'] = 4

        elif stat['Zeros'] >= 20 : stat['Zeros'] = 5



    elif attrib_type == 'form' or attrib_type == 'opposition' :

        if stat['Zeros'] == 1 :  stat['Zeros'] = 1

        elif stat['Zeros'] == 2 :  stat['Zeros'] = 2

        elif stat['Zeros'] == 3 :  stat['Zeros'] = 3

        elif stat['Zeros'] == 4 :  stat['Zeros'] = 4

        elif stat['Zeros'] >= 5 : stat['Zeros'] = 5

            

    # FOR HIGHEST SCORE

    if stat['Highest_score'] >= 1 and stat['Highest_score'] <= 24 :  stat['Highest_score'] = 1

    elif stat['Highest_score'] >= 25 and stat['Highest_score'] <= 49 :  stat['Highest_score'] = 2

    elif stat['Highest_score'] >= 50 and stat['Highest_score'] <= 99 :  stat['Highest_score'] = 3

    elif stat['Highest_score'] >= 100 and stat['Highest_score'] <= 150 :  stat['Highest_score'] = 4

    elif stat['Highest_score'] >= 150 : stat['Highest_score'] = 5

    

    # FOR OVERS

    if attrib_type == 'form' or attrib_type == 'opposition' :

        if stat['Total_overs_bowled'] >= 1 and stat['Total_overs_bowled'] <= 9 :  stat['Total_overs_bowled'] = 1

        elif stat['Total_overs_bowled'] >= 10 and stat['Total_overs_bowled'] <= 24 :  stat['Total_overs_bowled'] = 2

        elif stat['Total_overs_bowled'] >= 25 and stat['Total_overs_bowled'] <= 49 :  stat['Total_overs_bowled'] = 3

        elif stat['Total_overs_bowled'] >= 50 and stat['Total_overs_bowled'] <= 100 :  stat['Total_overs_bowled'] = 4

        elif stat['Total_overs_bowled'] >= 100 : stat['Total_overs_bowled'] = 5



    elif attrib_type == 'consistency' :

        if stat['Total_overs_bowled'] >= 1 and stat['Total_overs_bowled'] <= 99 :  stat['Total_overs_bowled'] = 1

        elif stat['Total_overs_bowled'] >= 100 and stat['Total_overs_bowled'] <= 249 :  stat['Total_overs_bowled'] = 2

        elif stat['Total_overs_bowled'] >= 250 and stat['Total_overs_bowled'] <= 499 :  stat['Total_overs_bowled'] = 3

        elif stat['Total_overs_bowled'] >= 500 and stat['Total_overs_bowled'] <= 1000 :  stat['Total_overs_bowled'] = 4

        elif stat['Total_overs_bowled'] >= 1000 : stat['Total_overs_bowled'] = 5



    elif attrib_type == 'venue' :

        if stat['Total_overs_bowled'] >= 1 and stat['Total_overs_bowled'] <= 9 :  stat['Total_overs_bowled'] = 1

        elif stat['Total_overs_bowled'] >= 10 and stat['Total_overs_bowled'] <= 19 :  stat['Total_overs_bowled'] = 2

        elif stat['Total_overs_bowled'] >= 20 and stat['Total_overs_bowled'] <= 29 :  stat['Total_overs_bowled'] = 3

        elif stat['Total_overs_bowled'] >= 30 and stat['Total_overs_bowled'] <= 39 :  stat['Total_overs_bowled'] = 4

        elif stat['Total_overs_bowled'] >= 40 : stat['Total_overs_bowled'] = 5

    

    # FOR BALLING AVG

    if stat['balling_avg'] >= 0 and stat['balling_avg'] <= 24.99 :  stat['balling_avg'] = 5

    elif stat['balling_avg'] >= 25 and stat['balling_avg'] <= 29.99 :  stat['balling_avg'] = 4

    elif stat['balling_avg'] >= 30 and stat['balling_avg'] <= 34.99 :  stat['balling_avg'] = 3

    elif stat['balling_avg'] >= 35 and stat['balling_avg'] <= 49.99 :  stat['balling_avg'] = 2

    elif stat['balling_avg'] >= 50 : stat['balling_avg'] = 1

            

    # FOR BALLING SR

    if stat['balling_SR'] >= 0 and stat['balling_SR'] <= 29.99 :  stat['balling_SR'] = 5

    elif stat['balling_SR'] >= 30 and stat['balling_SR'] <= 39.99 :  stat['balling_SR'] = 4

    elif stat['balling_SR'] >= 40 and stat['balling_SR'] <= 49.99 :  stat['balling_SR'] = 3

    elif stat['balling_SR'] >= 50 and stat['balling_SR'] <= 59.99 :  stat['balling_SR'] = 2

    elif stat['balling_SR'] >= 60 : stat['balling_SR'] = 1

        

    # FOR FOUR FIVE WICKET HAUL

    if attrib_type == 'consistency' :

        if stat['FFWH'] >= 1 and stat['FFWH'] <= 2 :  stat['FFWH'] = 3

        elif stat['FFWH'] >= 3 and stat['FFWH'] <= 4 :  stat['FFWH'] = 4

        elif stat['FFWH'] >= 5 : stat['FFWH'] = 5



    elif attrib_type == 'form' or attrib_type == 'opposition' or attrib_type == 'venue' :

        if stat['FFWH'] >= 1 and stat['FFWH'] <= 2 :  stat['FFWH'] = 4

        elif stat['FFWH'] >= 3 : stat['FFWH'] = 5

            

    return stat



def encode_label(run):

    if run >= 1 and run <= 24 : run = 1

    elif run >= 25 and run <= 49 : run = 2

    elif run >= 50 and run <= 74 : run = 3

    elif run >= 75 and run <= 99 : run = 4

    elif run >= 100 : run = 5

    return run



def transform(stats, attrib_type) :

    if attrib_type == 'consistency' :

        batting_attrib = 0.4262*stats['batting_avg'] + 0.2566*stats['Total_batting_innings'] + 0.1510*stats['batting_SR'] + 0.0787*stats['Centuries'] + 0.0556*stats['Fifties'] - 0.0328*stats['Zeros']

        balling_attrib = 0.4174*stats['Total_overs_bowled'] + 0.2634*stats['Total_balling_innings'] + 0.1602*stats['balling_SR'] + 0.0975*stats['balling_avg'] + 0.0615*stats['FFWH']

        return batting_attrib, balling_attrib

        

    elif attrib_type == 'form' :

        batting_attrib = 0.4262*stats['batting_avg'] + 0.2566*stats['Total_batting_innings'] + 0.1510*stats['batting_SR'] + 0.0787*stats['Centuries'] + 0.0556*stats['Fifties'] - 0.0328*stats['Zeros']

        balling_attrib = 0.3269*stats['Total_overs_bowled'] + 0.2846*stats['Total_balling_innings'] + 0.1877*stats['balling_SR'] + 0.1210*stats['balling_avg'] + 0.0798*stats['FFWH']

        return batting_attrib, balling_attrib

        

    elif attrib_type == 'opposition' :

        batting_attrib = 0.4262*stats['batting_avg'] + 0.2566*stats['Total_batting_innings'] + 0.1510*stats['batting_SR'] + 0.0787*stats['Centuries'] + 0.0556*stats['Fifties'] - 0.0328*stats['Zeros']

        balling_attrib = 0.3177*stats['Total_overs_bowled'] + 0.3177*stats['Total_balling_innings'] + 0.1933*stats['balling_SR'] + 0.1465*stats['balling_avg'] + 0.0943*stats['FFWH']

        return batting_attrib, balling_attrib

    

    elif attrib_type == 'venue' :

        batting_attrib = 0.4262*stats['batting_avg'] + 0.2566*stats['Total_batting_innings'] + 0.1510*stats['batting_SR'] + 0.0787*stats['Centuries'] + 0.0556*stats['Fifties'] - 0.0328*stats['Highest_score']

        balling_attrib = 0.3018*stats['Total_overs_bowled'] + 0.2783*stats['Total_balling_innings'] + 0.1836*stats['balling_SR'] + 0.1391*stats['balling_avg'] + 0.0972*stats['FFWH']

        return batting_attrib, balling_attrib
training_data = {}

def updated_stats(player) :

    '''

    Update No. of innings, Sum of Scores, Balls Faced, Centuries, Zeros, Highest Score, No. Of Innings as a bowler, overs bowled,

    Runs Conceded, Balls Bowled, Wickets Taken, 4/5 wickets haul

    '''

    li = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    full = []

    player = player.copy(deep=True)

    player.index = range(len(player))

    for index in range(len(player)) :

        #Updating total batting innings

        li[0] += player.loc[index, 'Batting_Innings']



        #Updating sum of scores

        li[1] += player.loc[index, 'Runs_Scored']



        #Updating Total Balls Faced

        li[2] += player.loc[index, 'Balls_Played']



        #Updating Centuries

        li[3] += 1 if player.loc[index, 'Runs_Scored'] >= 100 else 0



        #Updating Zeros

        li[4] += 1 if player.loc[index, 'Runs_Scored'] == 0 else 0



        #Updating Highest Score

        li[5] = player.loc[index, 'Runs_Scored'] if player.loc[index, 'Runs_Scored'] > li[5] else li[5]



        #Updating total balling innings

        li[6] += player.loc[index, 'Bowling_Innings']



        #updating overs bowled

        li[7] += player.loc[index, 'Balls_Bowled']/6



        #updating runs conceded

        li[8] += player.loc[index, 'Runs_Given']



        #updating total balls bowled

        li[9] += player.loc[index, 'Balls_Bowled']



        #updating total wickets taken

        li[10] += player.loc[index, 'Wickets_Taken']



        #updating FFWH

        li[11] += 1 if player.loc[index, 'Wickets_Taken'] >= 4 else 0

        

        #updating 50s

        li[12] += 1 if player.loc[index, 'Runs_Scored'] >= 50 else 0

        

    return {

        'Total_batting_innings' : li[0], 'batting_avg' : li[1]/li[0], 'batting_SR' : 100*(li[1]/li[2]), 'Centuries' : li[3], 'Fifties' : li[12], 'Zeros' : li[4], 'Highest_score' : li[5],

        'Total_balling_innings' : li[6], 'Total_overs_bowled' : li[7], 'balling_avg' : li[8]/li[10], 'balling_SR' : li[9]/li[10],

        'FFWH' : li[11]

    }
# MAKING THE MODELS OF DIFFERENT PLAYERS

player_models = {}

derived_attributes = {}
def train_player_model(player, model) :

    data = derived_attributes[player][:-1]

    inputs = [[items[0], items[2], items[4], items[8]] for items in data]

    label = [items[-1] for items in data]

    model.fit(np.array(inputs).reshape(-1, 4), label)

    return(model)
def predict_player_runs(team_1, team_2, team1_lineup, team2_lineup, date, match_id, venue, model, mode):

    global player_models, total_accruacy

    denominator = 0

    numerator = 0

    for lineup in [team1_lineup, team2_lineup] :

        li = []

        for players in lineup :

            df = player_dict[players][player_dict[players].Date < pd.to_datetime(date)]

            

            # Calculating Consistency

            consistency_stats = updated_stats(df)

            consistence_stats = encode(consistency_stats, 'consistency')

            consistence_batting, consistence_balling = transform(consistence_stats, 'consistency')



            # Calculating Form

            form_stats = updated_stats(df[(df.Date < pd.to_datetime(date)) & (df.Date >= pd.to_datetime(date) - timedelta(days=365))])

            form_stats = encode(form_stats, 'form')

            form_batting, form_balling = transform(consistence_stats, 'form')



            # Calculating Opposition

            oppn = team_2 if players in team1_lineup else team_1

            opposition_stats = updated_stats(df[df.Against == oppn])

            opposition_stats = encode(opposition_stats, 'opposition')

            opposition_batting, opposition_balling = transform(opposition_stats, 'opposition')



            # Calculating Venue

            venue_stats = updated_stats(df[df.Venue == venue])

            venue_stats = encode(venue_stats, 'venue')

            venue_batting, venue_balling = transform(venue_stats, 'venue')

            

            # Calculating Label

            label = encode_label(player_dict[players][player_dict[players]['Match_ID'] == match_id]['Runs_Scored'].to_list()[0])

            

            if players in derived_attributes:

                derived_attributes[players].append([consistence_batting, consistence_balling, form_batting, form_balling, opposition_batting, opposition_balling,

                                                   venue_batting, venue_balling, label])

            else :

                derived_attributes[players] = [[consistence_batting, consistence_balling, form_batting, form_balling, opposition_batting, opposition_balling,

                                                   venue_batting, venue_balling, label]]

            if not mode == 'Collect' :

                try :

                    player_models[players] = train_player_model(players, model)

                    attrib = derived_attributes[players][-1]

                    predicted_output = player_models[players].predict(np.array([attrib[0], attrib[2], attrib[4], attrib[8]]).reshape(1, -1))

                    print(team_1 if players in team1_lineup else team_2, players, predicted_output[0], attrib[-1])

                    if predicted_output[0] == attrib[-1] :

                        numerator += 1

                    denominator += 1

                

                except Exception as e:

                    print('No Derived Attributes Present')

                    print(e)

    

    if not mode == 'Collect' :

        total_accruacy += (numerator/denominator)

        return numerator/denominator
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
def get_match_accuracy(match_id, mode = 'Collect') :

    date = odi_matches[match_id]['Date']

    label = odi_matches[match_id]['Winner']

    team1_lineups = lineups[match_id][0]

    team2_lineups = lineups[match_id][1]

    team1 = odi_matches[match_id]['Team 1']

    team2 = odi_matches[match_id]['Team 2']

    venue = odi_matches[match_id]['Venue']

    clf = tree.DecisionTreeClassifier()

    predicted_runs = predict_player_runs(team1, team2, team1_lineups, team2_lineups, date, match_id, venue, clf, mode)

    return predicted_runs
train_match_ids = [key for key, values in odi_matches.items() if int(values['Date'][:4]) < 2018]

ml_train_dataset = []

for match_ids in tqdm(train_match_ids) :

    get_match_accuracy(match_ids, mode = 'Collect')
test_match_ids = [key for key, values in odi_matches.items() if int(values['Date'][:4]) > 2018]

total_accruacy = 0

ml_train_dataset = []

for match_ids in test_match_ids :

    try:

        print('------------------------------------------------------')

        accuracy = get_match_accuracy(match_ids, mode = 'Test')

        print('Accruacy Till Now :- {}'.format(accuracy))

    except:

        pass

print('----------------------------------------------')

print('Final_accuracy is :- ')

print(total_accruacy/len(test_match_ids))
len(player_model['V Kohli'])