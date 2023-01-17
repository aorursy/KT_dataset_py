# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex4 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "world_bank_intl_education" dataset

dataset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))

#for table in tables:

    #print(table.table_id)



# Construct a reference to the "international_education" table

table_ref = dataset_ref.table("international_education")



# API request - fetch the table

table = client.get_table(table_ref)



#table.schema



# Preview the first five lines of the "international_education" table

client.list_rows(table, max_results=5).to_dataframe()
# Your code goes here

country_spend_pct_query = """

                          SELECT country_name, AVG(value) AS avg_ed_spending_pct

                          FROM `bigquery-public-data.world_bank_intl_education.international_education`

                          WHERE indicator_code = "SE.XPD.TOTL.GD.ZS" and year>=2010 and year<=2017

                          GROUP BY country_name

                          ORDER BY AVG(value) DESC

                          """

#dry_run_config = bigquery.QueryJobConfig(dry_run=True)

#dry_run_query_job = client.query(country_spend_pct_query, job_config=dry_run_config)

#print("{}".format(dry_run_query_job.total_bytes_processed))

# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

#DRY RUN GAVE ME BYTES LESS THAN 232000000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=232*1000*1000)

country_spend_pct_query_job = client.query(country_spend_pct_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

country_spending_results = country_spend_pct_query_job.to_dataframe()



# View top few rows of results

print(country_spending_results.head())



# Check your answer

q_1.check()
#q_1.hint()

#q_1.solution()
# Your code goes here

code_count_query = """

                    SELECT indicator_code, indicator_name, count(1) AS num_rows

                    FROM `bigquery-public-data.world_bank_intl_education.international_education`

                    WHERE year=2016

                    GROUP BY indicator_code, indicator_name

                    HAVING count(1)>=175

                    ORDER BY num_rows DESC

                    """

                    



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

code_count_query_job = client.query(code_count_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

code_count_results = code_count_query_job.to_dataframe()



# View top few rows of results

print(code_count_results.head())



# Check your answer

q_2.check()
#q_2.hint()

#q_2.solution()