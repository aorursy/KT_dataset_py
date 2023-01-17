# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex4 import *



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

education_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                          dataset_name="world_bank_intl_education")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "world_bank_intl_education" dataset

dataset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "international_education" table

table_ref = dataset_ref.table("international_education")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "international_education" table

client.list_rows(table, max_results=5).to_dataframe()
education_data.head('international_education')
# Your Code Here



country_spend_pct_query = """

                          SELECT country_name, AVG(value) AS avg_ed_spending_pct

                          FROM `bigquery-public-data.world_bank_intl_education.international_education`

                          WHERE indicator_code = 'SE.XPD.TOTL.GD.ZS' and year >= 2010 and year <= 2017

                          GROUP BY country_name

                          ORDER BY avg_ed_spending_pct DESC

                          """



country_spending_results = education_data.query_to_pandas_safe(country_spend_pct_query)



print(country_spending_results.head())

q_1.check()
# q_1.hint()

# q_1.solution()
# Your Code Here

code_count_query = """

                   SELECT indicator_code, indicator_name, count(indicator_code) as num_rows

FROM `bigquery-public-data.world_bank_intl_education.international_education`

                          WHERE year = 2016 

                          GROUP BY indicator_code, indicator_name

                          HAVING COUNT(indicator_code)>=175

                          ORDER BY INDICATOR_CODE

"""





code_count_results = education_data.query_to_pandas_safe(code_count_query)



print(code_count_results.head())

q_2.check()

q_2.hint()

q_2.solution()