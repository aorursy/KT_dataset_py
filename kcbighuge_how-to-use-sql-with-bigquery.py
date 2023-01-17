import bq_helper

# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema('crime')
chicago_crime.head('crime')
print(type(chicago_crime.table_schema('crime')[-1]))
chicago_crime.table_schema('crime')[-1]
chicago_crime.head('crime', selected_columns='location', num_rows=10)
query = """SELECT district 
            FROM `bigquery-public-data.chicago_crime.crime`
            WHERE year = 2015"""

chicago_crime.estimate_query_size(query)
districts_2015 = chicago_crime.query_to_pandas_safe(query, max_gb_scanned=0.1)
districts_2015.head()
districts_2015.district.mean()
import bq_helper

baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")
baseball.list_tables()
baseball.head('games_wide')
baseball.table_schema('games_wide')
# find scores and venue of royals games
query = """SELECT gameId, awayTeamName, homeTeamName, awayFinalRuns, homeFinalRuns, venueName
            FROM `bigquery-public-data.baseball.games_wide`
            WHERE awayTeamName='Royals' or homeTeamName='Royals'"""

baseball.estimate_query_size(query)
# query_to_pandas_safe only returns result if < 1GB
kc_games = baseball.query_to_pandas_safe(query)
print(kc_games.shape)
kc_games.head()
# who did royals play most as the home team?
kc_games.homeTeamName.value_counts()
# who did they most as the away team?
kc_games.awayTeamName.value_counts()
import bq_helper
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")
query = """SELECT COUNT(gameId)
            FROM `bigquery-public-data.baseball.games_wide`
            """

baseball.query_to_pandas_safe(query).head()
# group by whoever was the visiting team
query = """SELECT awayTeamName, COUNT(gameId)
            FROM `bigquery-public-data.baseball.games_wide`
            GROUP BY awayTeamName
            """
baseball.query_to_pandas_safe(query).head()
query = """SELECT gameId, awayTeamName, homeTeamName, awayFinalRuns, homeFinalRuns, venueName
            FROM `bigquery-public-data.baseball.games_wide`
            WHERE awayTeamName='Royals' or homeTeamName='Royals'
            GROUP BY gameId"""

baseball.estimate_query_size(query)
# let's select fewer columns
query = """SELECT homeTeamName, COUNT(gameId)
            FROM `bigquery-public-data.baseball.games_wide`
            WHERE awayTeamName='Royals'
            GROUP BY homeTeamName"""
baseball.estimate_query_size(query)
baseball.query_to_pandas_safe(query).head()
# let's select fewer columns
query = """SELECT homeTeamName, COUNT(gameId)
            FROM `bigquery-public-data.baseball.games_wide`
            WHERE awayTeamName='Royals'
            GROUP BY homeTeamName
            HAVING COUNT(gameId)>2000"""
baseball.estimate_query_size(query)
# look for games with most common opponents
baseball.query_to_pandas_safe(query).head()
# how many different venues are there?
query = """SELECT venueName, COUNT(gameId)
            FROM `bigquery-public-data.baseball.games_wide`
            GROUP BY venueName
            """
baseball.query_to_pandas_safe(query)
# how many games played at the venues?
query = """SELECT venueName, COUNT(DISTINCT gameId)
            FROM `bigquery-public-data.baseball.games_wide`
            GROUP BY venueName
            """
baseball.query_to_pandas_safe(query)
# what is attendance for the game at Fort Bragg?
query = """SELECT venueName, attendance
            FROM `bigquery-public-data.baseball.games_wide`
            WHERE venueName='Fort Bragg'
            """
baseball.query_to_pandas_safe(query).head()
# what is total attendance for each venue?
# https://stackoverflow.com/questions/34706740/how-to-get-count-of-distinct-following-group-by-in-sql
query = """SELECT venueName, COUNT(DISTINCT gameId) AS game_count, SUM(attendance) AS total_attendance
            FROM (SELECT DISTINCT gameId, venueName, attendance FROM `bigquery-public-data.baseball.games_wide`)
            GROUP BY venueName
            HAVING game_count > 0
            ORDER BY total_attendance DESC
            """
baseball.query_to_pandas_safe(query)
# is there a correlation between attendance and runs scored?
# https://cloudplatform.googleblog.com/2013/09/introducing-corr-to-google-bigquery.html
query = """SELECT venueName, homeTeamName, COUNT(DISTINCT gameId) AS game_count, SUM(attendance) AS total_attendance,
                SUM(CASE WHEN homeFinalRuns>awayFinalRuns then 1 else 0 end) as home_wins, 
                CORR(attendance, homeFinalRuns) corr
            FROM (SELECT DISTINCT gameId, venueName, homeTeamName, attendance, homeFinalRuns, awayFinalRuns 
                FROM `bigquery-public-data.baseball.games_wide`)
            GROUP BY venueName, homeTeamName
            HAVING game_count > 1
            ORDER BY total_attendance DESC
            """
baseball.query_to_pandas_safe(query)
import bq_helper
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")

# total attendance for each home team ordered by team name
query = """SELECT homeTeamName, COUNT(DISTINCT gameId) AS game_count, SUM(attendance) AS total_attendance
            FROM (SELECT DISTINCT gameId, homeTeamName, attendance FROM `bigquery-public-data.baseball.games_post_wide`)
            GROUP BY homeTeamName
            ORDER BY homeTeamName DESC
            """
baseball.query_to_pandas_safe(query)
# let's go back to the chicago crime dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")

query = """SELECT case_number, date
            FROM `bigquery-public-data.chicago_crime.crime`
            ORDER BY date
            """
chicago_crime.estimate_query_size(query)
chicago_crime.query_to_pandas_safe(query).head()
# get the day of each date
query = """SELECT EXTRACT(DAYOFWEEK FROM date)
            FROM `bigquery-public-data.chicago_crime.crime`
            ORDER BY date
            """
chicago_crime.estimate_query_size(query)
chicago_crime.query_to_pandas_safe(query).head(10)
# get the first 2 words from primary_type
query = """SELECT REGEXP_EXTRACT(primary_type, r"\w+[^\w]+\w+")
            FROM `bigquery-public-data.chicago_crime.crime`
            ORDER BY date
            """
chicago_crime.estimate_query_size(query)
chicago_crime.query_to_pandas_safe(query).head(10)
# get the number of arrests for each day of week
query = """SELECT COUNT(unique_key) num, EXTRACT(DAYOFWEEK FROM date) day
            FROM (SELECT * FROM `bigquery-public-data.chicago_crime.crime` WHERE arrest=True)
            GROUP BY day
            ORDER BY num DESC
            """
chicago_crime.estimate_query_size(query)
arrests_by_day = chicago_crime.query_to_pandas_safe(query)
arrests_by_day
import matplotlib.pyplot as plt

# plot the arrest frequencies
plt.bar(arrests_by_day.day, arrests_by_day.num)
plt.title("Number of arrests by day of week\n(Sunday = 1)")
plt.xlabel('Day of week');
# get the number of arrests for each hour of the day
query = """SELECT COUNT(unique_key) num, EXTRACT(HOUR FROM date) hour
            FROM (SELECT * FROM `bigquery-public-data.chicago_crime.crime` WHERE arrest=True)
            GROUP BY hour
            ORDER BY num DESC
            """
chicago_crime.estimate_query_size(query)
arrests = chicago_crime.query_to_pandas_safe(query)
arrests
plt.bar(arrests.hour, arrests.num)
plt.title("Number of arrests by hour of the day")
plt.xlabel('Hour of day');
import bq_helper
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")

# get attendance for royals games
query = """WITH Royals AS 
                (
                    SELECT DISTINCT gameId,  startTime, homeTeamName, awayTeamName, attendance
                    FROM `bigquery-public-data.baseball.games_wide`
                    WHERE homeTeamName='Royals' OR awayTeamName='Royals'
                )
            SELECT  EXTRACT(DATE FROM startTime) date, awayTeamName, homeTeamName, attendance
            FROM Royals
            ORDER BY startTime
            """
baseball.estimate_query_size(query)
royals_games = baseball.query_to_pandas_safe(query)
print(royals_games[:5])
royals_games.tail()
# get the weekly attendance for royals games
query = """WITH Royals AS 
                (
                    SELECT DISTINCT gameId,  startTime, attendance
                    FROM `bigquery-public-data.baseball.games_wide`
                    WHERE homeTeamName='Royals' OR awayTeamName='Royals'
                )
            SELECT EXTRACT(WEEK FROM startTime) week, COUNT(gameId) games, SUM(attendance) attendance
            FROM Royals
            GROUP BY week
            ORDER BY week
            """
baseball.estimate_query_size(query)
royals = baseball.query_to_pandas_safe(query)
royals.head()
# import plotting library
import matplotlib.pyplot as plt

# plot weekly attendance
plt.plot(royals.attendance)
plt.title("Weekly attendance at Royals games");
# get the daily attendance for all mlb games
query = """WITH mlb AS 
                (
                    SELECT DISTINCT gameId, startTime, attendance
                    FROM `bigquery-public-data.baseball.games_wide`
                )
            SELECT EXTRACT(DATE FROM startTime) date, COUNT(gameId) games, SUM(attendance) attendance
            FROM mlb
            GROUP BY date
            ORDER BY date
            """
baseball.estimate_query_size(query)
majors = baseball.query_to_pandas_safe(query)
print(majors.head())

# plot weekly attendance
plt.plot(majors.attendance)
plt.title("Daily attendance at MLB games");
import bq_helper
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")
baseball.list_tables()
baseball.head('schedules')
# what was the "game number" for regular season games at wrigley
query = """SELECT DISTINCT EXTRACT(DATE FROM g.startTime) date, g.gameId, s.gameNumber, g.startTime, g.awayTeamName, g.homeTeamName, g.venueName, g.attendance
            FROM `bigquery-public-data.baseball.games_wide` as g
            INNER JOIN `bigquery-public-data.baseball.schedules` as s ON g.gameId = s.gameId
            WHERE g.venueName='Wrigley Field'
            ORDER BY g.startTime
            """
baseball.estimate_query_size(query)
baseball.query_to_pandas_safe(query).head()