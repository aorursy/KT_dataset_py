import bq_helper

# CREATE OBJECT 

baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")
baseball.list_tables() #TABLES REVIEW
baseball.head('games_wide') # TABLE PREVIEW
query = """SELECT homeTeamName,awayTeamName,homeFinalRuns
            FROM `bigquery-public-data.baseball.games_wide`
            WHERE homeFinalRuns <10
             """
Number_team = baseball.query_to_pandas_safe(query)
Number_team
query = """SELECT homeTeamName,venueMarket
            FROM `bigquery-public-data.baseball.games_wide`
            GROUP BY homeTeamName,venueMarket;
             """  
market_by_team = baseball.query_to_pandas_safe(query)

market_by_team
