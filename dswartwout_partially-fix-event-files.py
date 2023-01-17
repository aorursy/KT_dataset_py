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
import re

import time
home = '/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament'
# this regular expression recognizes the point-scoring events, including the number of points scored

point_pat = re.compile('^made([1-3])$')



# to help in computing the current scores, this function determines whether an event provides points for a given team

# the row is expected to have three columns: WTeamID or LTeamID, EventTeamID, and EventType

def event_points(row):

    if row[0] == row[1]:

        made = point_pat.match(str(row[2]))

        if made is None:

            points = 0

        else:

            points = int(made.groups()[0])

    else:

        points = 0

    return points



# df should be a DataFrame with WTeamID, LTeamID, EventTeamID and EventType columns

# returns two Series, one for the winner's event points and the other for the loser's

def compute_event_points(df):

    return (df.loc[:,['WTeamID','EventTeamID','EventType']].apply(event_points,axis=1),

            df.loc[:,['LTeamID','EventTeamID','EventType']].apply(event_points,axis=1))
# df should be a DataFrame with Season, DayNum, WTeamId, LTeamID, EventTeamID, and EventType columns as described

# in the Data section of this competition

# this function adds columns for the winner's and loser's current score

# the column names are optional parameters

def add_current_scores(df,WCurScore='WCurScore',LCurScore='LCurScore'):

    # compute the event points - note that this can be done without regard to game boundaries

    WEventPoints, LEventPoints = compute_event_points(df)

    

    # next, find the game boundaries

    # iterrows() returns a 2-tuple consisting of the index and the column values

    # because of drop_duplicates(), the index is the index of the first event of the game 

    games = df.loc[:,['Season','DayNum','WTeamID','LTeamID']].drop_duplicates()

    starts = [game[0] for game in games.iterrows()]

    ends = [game[0]-1 for game in games.iterrows()][1:] + [df.shape[0]-1] 

    

    # now make the current score columns, going one game at a time

    df.loc[:,'WCurScore'] = pd.concat([WEventPoints.loc[starts[n]:ends[n]].cumsum() for n in range(games.shape[0])]).to_numpy()

    df.loc[:,'LCurScore'] = pd.concat([LEventPoints.loc[starts[n]:ends[n]].cumsum() for n in range(games.shape[0])]).to_numpy()

    

# a small number of games yield a mismatch between their final score and the final values of their current scores

# for example, there are 11 such games in 2019 and 29 in 2018

# df should be an events DataFrame

def find_score_issue_games(df):

    games = df.loc[:,['Season','DayNum','WTeamID','LTeamID']].drop_duplicates()

    starts = [game[0] for game in games.iterrows()]

    ends = [game[0]-1 for game in games.iterrows()][1:] + [df.shape[0]-1] 

    endgames = df.loc[ends,:]

    oops = endgames[(endgames['WFinalScore']!=endgames['WCurScore'])|(endgames['LFinalScore']!=endgames['LCurScore'])]

    print(f'{oops.shape[0]} games with final score/event consistency issues ({oops.shape[0]*100/games.shape[0]:.2f}%)')
for year in [2015, 2016, 2017, 2018, 2019]:

    print(f'working on {year}')

    

    # read an event dataset

    read_start = time.time()

    events = pd.read_csv(f'{home}/MEvents{year}.csv')

    read_end = time.time()

    read_elapsed = read_end - read_start

    print(f'{read_elapsed:.2f} sec to read {events.shape[0]} rows ({events.shape[0]/read_elapsed:.2f} rows/sec)')

    

    # fill in the current score columns

    fill_start = time.time()

    add_current_scores(events)

    fill_end = time.time()

    fill_elapsed = fill_end - fill_start

    print(f'{fill_elapsed:.2f} sec to process {events.shape[0]} events ({events.shape[0]/fill_elapsed:.2f} events/sec)')

    

    # write the results

    write_start = time.time()

    events.to_csv(f'Mevents_reg_season_{year}.csv',header=True,index=False)

    write_end = time.time()

    write_elapsed = write_end - write_start

    print(f'{write_elapsed:.2f} sec to write {events.shape[0]} rows ({events.shape[0]/write_elapsed:.2f} rows/sec)')

    

    find_score_issue_games(events)

    print('')