# Getting Started with SQL and BigQuery
# BigQuery dataset contains data about NCAA basketball with play-by-play and box scores since 2009
# Final scores back to 1996
# Wins and losses back to 1894-5 for some teams

# Imports
import bq_helper  # Helper functions for putting BigQuery results in Pandas DataFrames
# Create a BigQueryhelper object pointing to a specific dataset
ncaa_basketball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                          dataset_name="ncaa_basketball")
# Print a list of all the tables in the ncaa_basketball dataset
ncaa_basketball.list_tables()
# Print information on all the columns in the table in the ncaa_basketball dataset
ncaa_basketball.table_schema("mbb_teams_games_sr")
# Preview the first couple of lines of the "mbb_teams_games_sr" table
ncaa_basketball.head("mbb_teams_games_sr")
# Preview the first ten entries in the 'three_points_made' in the "mbb_teams_games_sr" table
ncaa_basketball.head("mbb_teams_games_sr", selected_columns='three_points_made', num_rows=10)
# This query looks in the "mbb_teams_games_sr" table and gets the 'three_points_made' column
query = """SELECT three_points_made 
            FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`"""

# Check how big this query will be in GB
ncaa_basketball.estimate_query_size(query)
# Run the query only if it is below the specified upper limit
threes_made = ncaa_basketball.query_to_pandas_safe(query, max_gb_scanned=0.001)
# Average three pointers made since 2014
threes_made.three_points_made.mean()