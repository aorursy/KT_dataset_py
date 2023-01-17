import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.cloud import bigquery


client = bigquery.Client()
dataset_ref = client.dataset("new_york", project = 'bigquery-public-data') # Create reference to new york dataset
myDataset = client.get_dataset(dataset_ref) # stores dataset into myDataset

tables = list(client.list_tables(myDataset))
for table in tables:
    print(table.table_id)
print("There are {} tables" .format(len(tables)))
table_ref_tree = dataset_ref.table('tree_species')  # takes table named 'tree_species' and creates a reference for it
treeSpecies = client.get_table(table_ref_tree) # stores tree_species table into treeSpecies
client.list_rows(treeSpecies, max_results = 6).to_dataframe() # prints first 6 rows of the treeSpecies table


# query selects all collumns from the tree_species table
query1 = """
    SELECT *
    FROM bigquery-public-data.new_york.tree_species
    """
dry_run_config = bigquery.QueryJobConfig(dry_run = True)
dry_run_query_job = client.query(query1, job_config = dry_run_config)


print("This query will process {} bytes." .format(dry_run_query_job.total_bytes_processed))
OneHundMB = 1000*1000*100  # one hundred megabytes
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB) # limits number of bytes that can be pulled
# lists name and datatype for all collumns in the treeSpecies table
treeSpecies.schema

# Code creates a reference and views first 6 lines of the tree_census_2015 table
table_ref_tree2015 = dataset_ref.table('tree_census_2015')
treeCensus2015 = client.get_table(table_ref_tree2015)
client.list_rows(treeCensus2015, max_results = 6).to_dataframe()


# List collumns for the treeCensus2015 data table
treeCensus2015.schema
# Query that selects and lists the total number of each tree type. Tree named based off thier scientific and common names
query2 = """
    SELECT
        COUNT(tree_id) AS Tree_Count,
        spc_latin AS Scientific_Name,
        spc_common AS Common_Name
    FROM `bigquery-public-data.new_york.tree_census_2015`
    GROUP BY 
        spc_latin,
        spc_common
    ORDER BY
        Tree_Count DESC
        
"""
# Make sure the data doesn't exceed our byte limit, then view the query as a table
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)
treeCensus2015 = client.query(query2, job_config = safe_config)
treeCensus2015.to_dataframe()


# View collumns in nypd_my_collisions table
table_ref_collision = dataset_ref.table("nypd_mv_collisions")
myAccidents = client.get_table(table_ref_collision)
myAccidents.schema
# Query that shows the total number of vehicle accidents that happened in NYC. Organized by year.
query3 = """
    SELECT 
        EXTRACT(YEAR FROM timestamp) AS Year,
        COUNT(unique_key) AS Total_Accidents,
        SUM(number_of_persons_injured) AS Injuries,
        SUM(number_of_persons_killed) AS Deaths
    FROM `bigquery-public-data.new_york.nypd_mv_collisions`
    GROUP BY Year
    ORDER BY Year
    """
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)
collisionData = client.query(query3, job_config=safe_config)
collisionData.to_dataframe()
# View first 6 rows in citibike_stations table

table_ref_citibike = dataset_ref.table("citibike_stations")
bikeStations = client.get_table(table_ref_citibike)
client.list_rows(bikeStations,max_results = 6).to_dataframe()
# Query that shows each bike station name with its cooresponding location in NYC
query4 = """
    SELECT name, latitude, longitude
    FROM `bigquery-public-data.new_york.citibike_stations`
    
    """
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)
collisionData = client.query(query4, job_config=safe_config)
collisionData.to_dataframe()
# View collumns in treeSpecies table

treeSpecies.schema
# View collumns in tree_census_2015 table

table_ref_tree2015 = dataset_ref.table('tree_census_2015')
treeCensus2015 = client.get_table(table_ref_tree2015)
treeCensus2015.schema
# query that attempts to find the location of all the trees in NYC that have yellow leaves in the fall.
query5 = """

    SELECT cs.spc_latin AS Common_Name, cs.latitude, cs.longitude
    FROM `bigquery-public-data.new_york.tree_census_2015` AS cs INNER JOIN `bigquery-public-data.new_york.tree_species` AS ts
    ON cs.spc_latin = ts.species_scientific_name
    WHERE fall_color LIKE "yellow"
        
"""

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)
treeCensus2015 = client.query(query5, job_config = safe_config)
treeCensus2015.to_dataframe()

table_ref_tree2015 = dataset_ref.table('tree_census_2015')
treeCensus2015 = client.get_table(table_ref_tree2015)
client.list_rows(treeCensus2015, max_results = 7).to_dataframe()
# Query that checks to see if the two datasets tree_census_2015 and tree_species have matching names for trees
query6 = """
    SELECT spc_latin AS Scientific_Name
    FROM `bigquery-public-data.new_york.tree_census_2015` AS cs INNER JOIN `bigquery-public-data.new_york.tree_species` AS ts
    ON cs.spc_latin = ts.species_scientific_name 
    GROUP BY spc_latin
    ORDER BY spc_latin DESC
"""

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)
treeCensus2015 = client.query(query6, job_config = safe_config)
treeCensus2015.to_dataframe()

