# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #import matplotlib
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import package with helper functions 
from google.cloud import bigquery
import bq_helper


# create a helper object for this dataset
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")
baseball.list_tables()
# Your Code Goes Here
#`bigquery-public-data.baseball.schedules`
# all finished atbats,
atbat_query = """SELECT gameId, startTime, attendance, dayNight, durationMinutes, awayTeamName, homeTeamName, venueName, venueSurface, outcomeDescription, pitcherId, pitcherFirstName, pitcherLastName, pitcherThrowHand, hitterBatHand, is_ab, is_ab_over, is_hit, is_on_base, is_bunt, is_bunt_shown, is_double_play, is_triple_play, is_wild_pitch, is_passed_ball 
              FROM `bigquery-public-data.baseball.games_wide`
              WHERE is_ab_over = 1
              """
at_bat = baseball.query_to_pandas_safe(atbat_query)
at_bat.head()
# all game level info
game_query = """SELECT distinct gameId, startTime, attendance, dayNight, durationMinutes, awayTeamName, homeTeamName, venueName, venueSurface, homeFinalRuns, homeFinalHits, homeFinalErrors, awayFinalRuns, awayFinalHits, awayFinalErrors
        FROM `bigquery-public-data.baseball.games_wide`
        order by gameId
        """
game_info = baseball.query_to_pandas_safe(game_query)
game_info.head()

# all game level info
gamestats_query = """SELECT distinct gameId, max(attendance) as Attendance, max(durationMinutes) as Duration, max(homeFinalRuns) as HomeRuns, max(homeFinalHits) as HomeHits, max(homeFinalErrors) as HomeErrors, max(awayFinalRuns) as AwayRuns, max(awayFinalHits) as AwayHits, max(awayFinalErrors) as AwayErrors, count(distinct pitcherId) as PitchingChanges, sum(is_ab_over) as PlateAppearances, sum(is_hit) as TotalHits, sum(is_on_base) as TotalOnBase, sum(is_bunt) as TotalBunts, sum(is_bunt_shown) as TotalBuntShown, sum(is_wild_pitch) as TotalWildPitch, sum(is_passed_ball) as TotalPassedBall
        FROM `bigquery-public-data.baseball.games_wide`
        group by gameID
        order by gameID
        """
game_stats = baseball.query_to_pandas_safe(gamestats_query)
game_stats.head()

game_stats.describe()
game_stats_plot = game_stats.plot(x='PitchingChanges',y='Duration',kind='scatter')
game_stats_plot.set_xlabel('Pitching Changes')
game_stats_plot.set_ylabel('Duration (m)')
game_stats_plot.set_xticks(range(0,22))
sns.lmplot(x='PitchingChanges',y='Duration',data=game_stats,fit_reg=True) 
plt.show()
#game_stats_hist = pd.DataFrame.hist(data=game_stats, column ='PitchingChanges', bins = 20)
game_stats_hist = game_stats.PitchingChanges.plot('hist', bins = 11, xticks = range(0,22))
plt.show()