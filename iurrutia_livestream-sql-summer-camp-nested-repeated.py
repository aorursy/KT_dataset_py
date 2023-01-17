from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "google_analytics_sample" dataset

dataset_ref = client.dataset("google_analytics_sample", project="bigquery-public-data")



# Construct a reference to the "ga_sessions_20170801" table

table_ref = dataset_ref.table("ga_sessions_20170801")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the table

client.list_rows(table, max_results=5).to_dataframe()
print("SCHEMA field for the 'totals' column:\n")

print(table.schema[5])



print("\nSCHEMA field for the 'device' column:\n")

print(table.schema[7])
# Query to count the number of transactions per browser

query = """

        SELECT device.browser AS device_browser,

            SUM(totals.transactions) as total_transactions

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`

        GROUP BY device_browser

        ORDER BY total_transactions DESC

        """



# Run the query, and return a pandas DataFrame

result = client.query(query).result().to_dataframe()

result.head()
table.schema[10]

# Query to determine most popular landing point on the website

query = """

        SELECT hits.page.pagePath as path,

            COUNT(hits.page.pagePath) as counts

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`, 

            UNNEST(hits) as hits

        WHERE hits.type="PAGE" and hits.hitNumber=1

        GROUP BY path

        ORDER BY counts DESC

        """



# Run the query, and return a pandas DataFrame

result = client.query(query).result().to_dataframe()

result.head()