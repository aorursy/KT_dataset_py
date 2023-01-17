# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex4 import *



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

education_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                          dataset_name="world_bank_intl_education")
education_data.head('international_education',10)
# Your Code Here



country_spend_pct_query = """

SELECT country_name,AVG(value) as avg_ed_spending_pct

FROM `bigquery-public-data.world_bank_intl_education.international_education`

WHERE indicator_code = 'SE.XPD.TOTL.GD.ZS' and year>2009 and year<2018

GROUP BY country_name

ORDER BY  avg_ed_spending_pct DESC

"""



country_spending_results = education_data.query_to_pandas_safe(country_spend_pct_query)



print(country_spending_results.head())

q_1.check()
q_1.hint()

q_1.solution()
# Your Code Here

code_count_query = """

SELECT indicator_name,indicator_code, COUNT(1) AS num_rows

FROM `bigquery-public-data.world_bank_intl_education.international_education`

WHERE year=2016

GROUP BY indicator_name,indicator_code

HAVING COUNT(1) >= 175

ORDER BY COUNT(1) DESC

"""



code_count_results = education_data.query_to_pandas_safe(code_count_query)



print(code_count_results.head(10))

#q_2.check()

q_2.hint()

q_2.solution()