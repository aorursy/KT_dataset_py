#Author: Skyler Prickett and Joseph Reid

#Assignment: Live Code Assignment

#Overview: This program will handle data from a new york data set and fufill taks as they are given, relating to the data set
#These tasks will primarily involve creating references to the data, fetching the data, and preforming operations to display the data in a data efficient manner
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.cloud import bigquery
client = bigquery.Client() #creating Client object
dataset_ref = client.dataset("new_york", project = "bigquery-public-data") #constructing a reference to "new_york"
nycDataset = client.get_dataset(dataset_ref)#API request -  fetching dataset

tables = list(client.list_tables(nycDataset))#listing all tables in nycDataset

for table in tables:#This will print names of all tables in the dataset
    print(table.table_id)

print("There appear to be {} tables".format(len(tables)))#displaying number of tables
table_ref_tree = dataset_ref.table("tree_species")#creating a reference to the "tree_species" table
treeSpecies = client.get_table(table_ref_tree)#fetching table
client.list_rows(treeSpecies, max_results = 6).to_dataframe()#listing the first 6 rows of the "tree_species" table

#Creating query to select all the items from all columns from the dataset
query1 = """
    SELECT *
    FROM `bigquery-public-data.new_york.tree_species`
    """

dry_run_config = bigquery.QueryJobConfig(dry_run = True)#Creating a QueryJobConfig object to estimate the size of the query without running it
dry_run_query_job = client.query(query1, job_config = dry_run_config)#Estimating costs of query by dry running it

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
OneHundMB = 1000*1000*100#only run query if it is less than one hundred MB
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)
treeSpecies.schema#Printing information of all the columns in the "tree_species" table of the dataset
table_ref_tree2015 = dataset_ref.table("tree_census_2015")#creating a reference to the "tree_census_2015" table
treeCensus2015 = client.get_table(table_ref_tree2015)
client.list_rows(treeCensus2015, max_results = 6).to_dataframe()#printing first 6 rows of given table

treeCensus2015.schema
#query to select the number of trees and their scientific and common name, grouped by names and ordered by descending tree count
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
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)#creating parameters for a dry run to estimate costs of running given query
treeCensus2015 = client.query(query2, job_config=safe_config)
treeCensus2015.to_dataframe()

table_ref_collision = dataset_ref.table("nypd_mv_collisions")#creating a reference to the "nypd_mv_collisions"
mvAccidents = client.get_table(table_ref_collision)#fetching table data
mvAccidents.schema
#query showing year, count of accidents, number of injuries and number of death in new york, grouped and ordered by year
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
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)#creating dry run parameters to estimate data costs
collisionData = client.query(query3, job_config=safe_config)
collisionData.to_dataframe()
table_ref_citibike = dataset_ref.table("citibike_stations")#creating a reference to the "citibike_stations" table
bikeStations = client.get_table(table_ref_citibike)
#client.list_rows(bikeStations, max_results = 6).to_dataframe()
bikeStations.schema
#creating query to show coordinates of bike stations, shown as longitude and latitude
query4 = """
    SELECT name AS Name, longitude AS Longitude, latitude AS Latitude
    FROM `bigquery-public-data.new_york.citibike_stations`
    """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)#creating dry run parameters to estimate costs
bikeStationsData = client.query(query4, job_config=safe_config).to_dataframe()
bikeStationsData
treeSpecies.schema

table_ref_tree2015 = dataset_ref.table("tree_census_2015")#creating reference to the "tree_census_2015 table"
treeCensus2015 = client.get_table(table_ref_tree2015)
client.list_rows(treeCensus2015, max_results = 6).to_dataframe()#printing first 6 rows in table
#creating query that will show the locations of trees with yellow leaves during the fall time
#requires joing data from different data sets in nyc to show the correlation of tree species and leaf color in fall to estimate locations
query5 = """
    SELECT 
        spc_common AS Common_Name,
        longitude AS Longitude,
        latitude AS Latitude
    FROM `bigquery-public-data.new_york.tree_census_2015` AS cs INNER JOIN `bigquery-public-data.new_york.tree_species` AS ts
    ON cs.spc_latin = ts.species_scientific_name
    WHERE fall_color LIKE "yellow"
    """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)#creating dry run parameters to estimate data cost
yellowLeaves= client.query(query5, job_config=safe_config).to_dataframe()
yellowLeaves


#query listing scientific names of trees in new york in 2015
query6 = """
    SELECT spc_latin AS S_Name
    FROM `bigquery-public-data.new_york.tree_census_2015`
    GROUP BY spc_latin
    ORDER BY spc_latin DESC
    """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)#creating dry run parameters to estimate data cost
trees2015 = client.query(query6, job_config=safe_config).to_dataframe()
trees2015
#creating query to show the scientific names of tree species in new york descending
query7 = """
    SELECT species_scientific_name AS S_Name
    FROM `bigquery-public-data.new_york.tree_species`
    GROUP BY species_scientific_name
    ORDER BY species_scientific_name DESC
    """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = OneHundMB)#creating dry run parameters to estimate data cost
treesSpecies = client.query(query7, job_config=safe_config).to_dataframe()
treesSpecies