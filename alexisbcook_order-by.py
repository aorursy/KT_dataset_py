from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "nhtsa_traffic_fatalities" dataset

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "accident_2015" table

table_ref = dataset_ref.table("accident_2015")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "accident_2015" table

client.list_rows(table, max_results=5).to_dataframe()
# Query to find out the number of accidents for each day of the week

query = """

        SELECT COUNT(consecutive_number) AS num_accidents, 

               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week

        ORDER BY num_accidents DESC

        """
# Set up the query

query_job = client.query(query)



# API request - run the query, and convert the results to a pandas DataFrame

accidents_by_day = query_job.to_dataframe()



# Print the DataFrame

accidents_by_day