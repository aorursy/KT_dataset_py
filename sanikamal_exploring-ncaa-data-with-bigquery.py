import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.cloud import bigquery

client = bigquery.Client()
dataset_ref = client.dataset('ncaa_basketball', project='bigquery-public-data')
type(dataset_ref)
ncaa_dataset = client.get_dataset(dataset_ref)
type(ncaa_dataset)
[x.table_id for x in client.list_tables(ncaa_dataset)]
ncaa_team_colors = client.get_table(ncaa_dataset.table('team_colors'))
type(ncaa_team_colors)
# dir(ncaa_team_colors)
[command for command in dir(ncaa_team_colors) if not command.startswith('_')]
ncaa_team_colors.schema
schema_subset = [col for col in ncaa_team_colors.schema if col.name in ('code_ncaa', 'color')]

results = [x for x in client.list_rows(ncaa_team_colors, start_index=100, selected_fields=schema_subset, max_results=10)]
results
for i in results:

    print(dict(i))
BYTES_PER_GB = 2**30

ncaa_team_colors.num_bytes / BYTES_PER_GB
def estimate_gigabytes_scanned(query, bq_client):

    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun

    my_job_config = bigquery.job.QueryJobConfig()

    my_job_config.dry_run = True

    my_job = bq_client.query(query, job_config=my_job_config)

    BYTES_PER_GB = 2**30

    return my_job.total_bytes_processed / BYTES_PER_GB
estimate_gigabytes_scanned("SELECT id FROM `bigquery-public-data.ncaa_basketball.team_colors`", client)
estimate_gigabytes_scanned("SELECT * FROM `bigquery-public-data.ncaa_basketball.team_colors`", client)

ncaa_mbb_teams_games_sr = client.get_table(ncaa_dataset.table('mbb_teams_games_sr'))

ncaa_mbb_pbp_sr = client.get_table(ncaa_dataset.table('mbb_pbp_sr'))
ncaa_mbb_teams_games_sr.schema
ncaa_mbb_pbp_sr.schema
#standardSQL

query="""SELECT

  event_type,

  COUNT(*) AS event_count

FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`

GROUP BY 1

ORDER BY event_count DESC"""



# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

events_type = query_job.to_dataframe()

events_type
#standardSQL

#most three points made

query="""SELECT

  scheduled_date,

  name,

  market,

  alias,

  three_points_att,

  three_points_made,

  three_points_pct,

  opp_name,

  opp_market,

  opp_alias,

  opp_three_points_att,

  opp_three_points_made,

  opp_three_points_pct,

  (three_points_made + opp_three_points_made) AS total_threes

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

WHERE season > 2010

ORDER BY total_threes DESC

LIMIT 5"""



# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

most_three_points = query_job.to_dataframe()

most_three_points
#standardSQL

query="""SELECT

  venue_name, venue_capacity, venue_city, venue_state

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

GROUP BY 1,2,3,4

ORDER BY venue_capacity DESC

LIMIT 5"""





# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

highest_seating_cap = query_job.to_dataframe()

highest_seating_cap
#standardSQL

#highest scoring game of all time

query="""SELECT

  scheduled_date,

  name,

  market,

  alias,

  points_game AS team_points,

  opp_name,

  opp_market,

  opp_alias,

  opp_points_game AS opposing_team_points,

  points_game + opp_points_game AS point_total

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

WHERE season > 2010

ORDER BY point_total DESC

LIMIT 5"""



# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

highest_scoring_game = query_job.to_dataframe()

highest_scoring_game
#standardSQL

#biggest point difference in a championship game

query="""SELECT

  scheduled_date,

  name,

  market,

  alias,

  points_game AS team_points,

  opp_name,

  opp_market,

  opp_alias,

  opp_points_game AS opposing_team_points,

  ABS(points_game - opp_points_game) AS point_difference

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

WHERE season > 2014 AND tournament_type = 'National Championship'

ORDER BY point_difference DESC

LIMIT 5"""







# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

biggest_diff = query_job.to_dataframe()

biggest_diff