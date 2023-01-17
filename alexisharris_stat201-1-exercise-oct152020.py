# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from google.cloud import bigquery

client = bigquery.Client()  # import Kaggle's bigquery data

dataset_ref = client.dataset("new_york",project = "bigquery-public-data") # Call new_york dataset from bigquery data

nycDataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(nycDataset)) # create a list to view the tables within the new_york dataset

for table in tables:

    print(table.table_id)
print("Number of tables:",format(len(tables))) # print tables
table_ref_tree = dataset_ref.table('tree_species') # look at tree_species table

treeSpecies = client.get_table(table_ref_tree)

client.list_rows(treeSpecies, max_results = 6).to_dataframe() # look at 6 rows of the tree_species data
# create a query of the tree_species dataset. Will select everything from this table and see how many bytes this query will process

query1 = """

    SELECT *

    FROM bigquery-public-data.new_york.tree_species

"""

dry_run_config = bigquery.QueryJobConfig(dry_run = True) #dry run

dry_run_query_job = client.query(query1, job_config = dry_run_config)



print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed)) #line of code that views total bytes processed
OneHundMB = 1000*1000*100 # create a bytes size number that will process a query less than or equal to 100MB

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB) #set a safe congifuration number so that we don't go over 100MB
treeSpecies.schema #look at type of data in treeSpecies
#observe data from tree_census_2015

table_ref_tree2015 = dataset_ref.table('tree_census_2015')

treeCensus2015 = client.get_table(table_ref_tree2015)

client.list_rows(treeCensus2015, max_results = 6).to_dataframe() 
treeCensus2015.schema #look at type of data in treeCensus2015
# Create a query with a list of the most popular trees from highest number to lowest.

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



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB) #need to re-run safe_config line everytime you creat a new query

treeCensus2015 = client.query(query2, job_config=safe_config) #set safe config

treeCensus2015.to_dataframe() #output
table_ref_collision = dataset_ref.table('nypd_mv_collisions') #get the table for motor vehicle accidents

myAccidents = client.get_table(table_ref_collision)

myAccidents.schema # look at type of data inside table
# This query will output the number of motor vehicle accidents by year as well as injuries and deaths

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
#task 5

table_ref_citibike = dataset_ref.table('citibike_stations') #retrieve bike stations in NY data

bikeStations = client.get_table(table_ref_citibike)

client.list_rows(bikeStations, max_results = 6).to_dataframe()

bikeStations.schema #look at the type of data in this table
# This query creates a list of the Bike Stations in New York City along with their longitude and latitude.

query4 = """



         SELECT name AS Name, 

                longitude AS Longitude, 

                latitude AS Latitude

         FROM `bigquery-public-data.new_york.citibike_stations`

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)

bikeStationsData = client.query(query4, job_config=safe_config).to_dataframe()

bikeStationsData
#task 6

treeSpecies.schema #look at type of data in this table
# look at first few (6) entries of data within this table

table_ref_tree2015 = dataset_ref.table("tree_census_2015")

treeCensus2015 = client.get_table(table_ref_tree2015)

client.list_rows(treeCensus2015, max_results = 6).to_dataframe()
# Create a query that joins two tables, matching entries together by the trees latin names

query5 = """

         SELECT spc_common AS Common_Name,

                longitude AS Longitude,

                latitude AS Latitude

         FROM `bigquery-public-data.new_york.tree_census_2015` AS cs

         INNER JOIN `bigquery-public-data.new_york.tree_species` AS ts

         ON cs.spc_latin = ts.species_scientific_name

         WHERE fall_color LIKE "yellow"



"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)

yellowLeavesData = client.query(query5, job_config=safe_config).to_dataframe()

yellowLeavesData #output data

#Unfortunately doesn't work. Did not figure out how to do this or why it didn't work :(.
# Make a query that looks at the latin names from one of the tables 'tree_census_2015'.

query6 = """

         SELECT spc_latin AS S_Name

         FROM `bigquery-public-data.new_york.tree_census_2015`

         GROUP BY spc_latin

         ORDER BY spc_latin ASC

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)

tree2015 = client.query(query6, job_config=safe_config).to_dataframe()

tree2015
# Make a query that looks at the latin names from one of the tables 'tree_species'.

query7 = """

         SELECT species_scientific_name AS S_Name

         FROM `bigquery-public-data.new_york.tree_species`

         GROUP BY species_scientific_name

         ORDER BY species_scientific_name ASC

"""



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)

tree2015_2 = client.query(query7, job_config=safe_config).to_dataframe()

tree2015_2
