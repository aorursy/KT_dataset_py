## This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import numpy.random as npr # random stuff

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This list defines the garbage time parameters used here. Each entry in the list should be a (minutes,points) pair.

# If the first pair is (n,p), it means that during the last n minutes of the game, a lead of p points or more is

# considered insurmountable. If the next pair is (m,q), it means that during the preceding m minutes, a lead of q points or more

# is considered insurmountable. And so on. Garbage time begins the first time the winning team acquires an insurmountable lead,

# provided it keeps an insurmountable lead for the remainder of the game.

#

# Given this rule, if the winning team makes a shot that increases their lead from 13 points to 15 points with 1.5 min left,

# and they maintain a 15+ point lead for the rest of the game, then garbage time starts at ElapsedSeconds = 38.5 * 60 + 1 = 2311.

# If the winning team reaches a 20-point lead with 3 min left, maintains a 20+ point lead from then until the 2 min mark,

# and maintains a 15+ point lead through the last 2 min of the game, then garbage time starts at ElapsedSeconds = 37 *60 + 1 = 2221.

# On the other hand, if a team acquires a 15-point lead with 2 min left but wins the game by 14, then there is no garbage time.

garbage_time_rule = [(2,15),(3,20)]
import re

import time
home = '/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament'

event_files = '/kaggle/input/partially-fix-event-files'
# validate a garbage-time rule

# 1. the time range must be entirely within the second half

# 2. the points threshhold must be non-decreasing 

# times are converted to seconds for ease of use

def validate_gbg_rule(rule,game_length=40):

    thresh = 0

    end_of_part = game_length

    halftime = game_length/2

    by_sec = []

    for i,r in enumerate(rule):

        if r[1] < thresh:

            raise ValueError(f'lead threshhold {r[1]} in part {i} must be at least {thresh}')

        start_of_part = end_of_part - r[0]

        if start_of_part < halftime:

            raise ValueError(f'part {i} begins at {start_of_part}, which is in the first half')

        by_sec += [(start_of_part*60,r[1])]

        end_of_part = start_of_part

        thresh = r[1]

    return by_sec
# quick and dirty unit test

ve = None

try:

    validate_gbg_rule([(10,12),(10.5,15)])

except ValueError as e:

    ve = e

assert ve is not None



ve2 = None

try:

    validate_gbg_rule([(3,12),(3,11)])

except ValueError as e:

    ve2 = e

assert ve2 is not None



assert validate_gbg_rule([(4,13)]) == [(2160,13)]

assert validate_gbg_rule([(2,15),(3,20)]) == [(2280,15),(2100,20)]

assert validate_gbg_rule([(2,12),(2,15),(2,18)]) == [(2280,12),(2160,15),(2040,18)]
# if rule hasn't been validated, you get what you deserve :-)

def is_insurmountable(row,rule):

    es = row['ElapsedSeconds']

    lead = row['WCurScore'] - row['LCurScore']

    for r in rule:

        if es >= r[0] and lead >= r[1]:

            return 1

    return 0



# game_frame should be an events DataFrame for a single game

# rule should be a validated garbage time rule

def flag_garbage_time(df,rule):

    insurm = list(df.apply(lambda row: is_insurmountable(row,rule),axis=1))

    last_before_garbage = np.max([n for n in range(df.shape[0]) if insurm[n]==0])

    df.loc[:,'gbg_time'] = [int(n>last_before_garbage) for n in range(df.shape[0])]
gtr = validate_gbg_rule(garbage_time_rule)



for year in [2015, 2016, 2017, 2018, 2019]:

    print(f'working on {year}')

    

    # read an event dataset

    read_start = time.time()

    events = pd.read_csv(f'{event_files}/Mevents_reg_season_{year}.csv')

    read_end = time.time()

    read_elapsed = read_end - read_start

    print(f'{read_elapsed:.2f} sec to read {events.shape[0]} rows ({events.shape[0]/read_elapsed:.2f} rows/sec)')

    

    # find the games in the event file

    games = events.loc[:,['Season','DayNum','WTeamID','LTeamID']].drop_duplicates()

    starts = [game[0] for game in games.iterrows()]

    ends = [game[0]-1 for game in games.iterrows()][1:] + [events.shape[0]-1]

    print(f'found {len(games)} games in {year}')

    

    # fill in the garbage time column

    process_start = time.time()



    game_frames = []

    for n in range(len(games)):

        gm = events.loc[starts[n]:ends[n],:].copy()

        flag_garbage_time(gm,gtr)

        game_frames += [gm]

    pd.concat(game_frames).to_csv(f'Mevents_reg_season_{year}_gt.csv',header=True,index=False)

    

    process_end = time.time()

    process_elapsed = process_end - process_start

    print(f'{process_elapsed:.2f} sec to process {events.shape[0]} events ({events.shape[0]/process_elapsed:.2f} events/sec)')

    print('')