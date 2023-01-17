from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "openaq" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there's only one!)

for table in tables:  

    print(table.table_id)
# creating ref to the table

table_ref = dataset_ref.table("global_air_quality")



# API - fetch the table

table = client.get_table(table_ref)



#retrive the first 5 rows of the table



client.list_rows(table=table,max_results=5).to_dataframe()
# SELECTING ALL THE CITIES IN THE CITY COLUMENS WHERE THE COUNTRY NAME IS "US"

# In the query the bigquery-public-data is where all the kaggle data exists

# openaq is the dataset 

# global_air_quality is the name of the specific table

query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """

# sending the messenger to the table to perform the query

query_job = client.query(query=query)



us_cities = query_job.to_dataframe()



# print the us cities



us_cities.head()
# counting the number of unique cities in the us_cities dataframe

#using the pandas syntax. retrieve only the top cities

# count values retrieve the number of the cities name repeated in decreasing order and head returns the top 5

top_five_cities =us_cities.city.value_counts()

top_five_cities.head()
# it is possible to query multiple columens at the same time



query = """

        SELECT city, pollutant

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """



query_job = client.query(query= query)

city_pollutant_us = query_job.to_dataframe()

city_pollutant_us.head()
# size of the query with out running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)

dry_run_query_job = client.query(query, job_config=dry_run_config)

print("this query can process {} bytes of data at the same time".format(dry_run_query_job.total_bytes_processed))
# limiting the size of the query data retrieval for 1 MB

TEN_MB = 1000*1000

safe_config= bigquery.QueryJobConfig(maximum_bytes_billed=TEN_MB)

safe_query_job = client.query(query, job_config=safe_config)



safe_query_job.to_dataframe()