# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import bq_helper
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")

baseball.list_tables()
# look at some of the columns
baseball.table_schema('games_wide')[:30]
# get just the max runs for each home team
query = """WITH runs_scored AS
                (
                    SELECT homeTeamName, MAX(homeFinalRuns) max_runs
                    FROM `bigquery-public-data.baseball.games_wide`
                    GROUP BY homeTeamName
                )
            SELECT homeTeamName, max_runs
            FROM runs_scored
            ORDER BY homeTeamName
            """
baseball.estimate_query_size(query)
baseball.query_to_pandas_safe(query)
# get all the game info for when the home teams scored their max number of runs
query = """WITH runs_scored AS
                (
                    SELECT DISTINCT startTime, homeTeamName, homeFinalRuns, awayTeamName, awayFinalRuns, 
                        MAX(homeFinalRuns) OVER (PARTITION BY homeTeamName) max_runs
                    FROM `bigquery-public-data.baseball.games_wide`
                )
            SELECT EXTRACT(DATE FROM startTime) date, homeTeamName, homeFinalRuns, awayTeamName, awayFinalRuns
            FROM runs_scored
            WHERE homeFinalRuns = max_runs
            ORDER BY homeTeamName
            """
baseball.estimate_query_size(query)
baseball.query_to_pandas_safe(query)
# let's start by getting the wins for just one team, the KC Royals
query = """WITH games AS 
                (
                    SELECT DISTINCT startTime, homeTeamName, awayTeamName, homeFinalRuns, awayFinalRuns
                    FROM `bigquery-public-data.baseball.games_wide`
                    WHERE (homeTeamName='Royals' OR awayTeamName='Royals')
                )
            SELECT EXTRACT(DATE FROM startTime) date, 
                IF ((homeFinalRuns>awayFinalRuns AND homeTeamName='Royals' OR homeFinalRuns<awayFinalRuns AND awayTeamName='Royals'), 'W', 'sad!') win, 
                homeTeamName, homeFinalRuns, awayTeamName, awayFinalRuns
            FROM games
            ORDER BY date
            """
baseball.estimate_query_size(query)
baseball.query_to_pandas_safe(query)[-10:]




