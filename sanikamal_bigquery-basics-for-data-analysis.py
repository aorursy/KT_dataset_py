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
# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
table.schema
result = ["{0} {1}".format(schema.name,schema.field_type) for schema in table.schema]

result
query="""

    SELECT city FROM `bigquery-public-data.openaq.global_air_quality`

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:10]
# What five cities have the most measurements?

cities.city.value_counts().head()
query="""

    SELECT * FROM `bigquery-public-data.openaq.global_air_quality` WHERE averaged_over_in_hours>0.25

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:15]
query="""

    SELECT city FROM `bigquery-public-data.openaq.global_air_quality`

    GROUP BY 

    city

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:10]
query="""

    SELECT city,COUNT(*) FROM `bigquery-public-data.openaq.global_air_quality`

    GROUP BY 

    city

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:15]
query="""

    SELECT city,COUNT(*)  AS total_city FROM `bigquery-public-data.openaq.global_air_quality`

    GROUP BY 

    city

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:15]
query="""

    SELECT city,COUNT(*)  AS total_city FROM `bigquery-public-data.openaq.global_air_quality`

    GROUP BY city

    ORDER BY city

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:15]
query="""

    SELECT city,COUNT(*)  AS total_city FROM `bigquery-public-data.openaq.global_air_quality`

    GROUP BY city

    ORDER BY total_city

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:10]
query="""

    SELECT city,COUNT(*)  AS total_city FROM `bigquery-public-data.openaq.global_air_quality`

    GROUP BY city

    ORDER BY total_city DESC

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities[:10]
query="""

    SELECT city,COUNT(*)  AS total_city FROM `bigquery-public-data.openaq.global_air_quality`

    GROUP BY city

    ORDER BY total_city DESC

    """;

# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

cities = query_job.to_dataframe()

cities.to_csv('cityname_and_totalcity.csv',encoding='utf-8', index=False)