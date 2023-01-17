import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

wbed = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="world_bank_intl_education")



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
bq_assistant = BigQueryHelper("bigquery-public-data", "world_bank_intl_education")

bq_assistant.list_tables()
bq_assistant.head("series_summary", num_rows=10)
bq_assistant.table_schema("international_education")
query1 = """

SELECT

  country_name,

  AVG(value) AS average

FROM

  `bigquery-public-data.world_bank_intl_education.international_education`

WHERE

  indicator_code = "SE.XPD.TOTL.GB.ZS"

  AND year > 2000

GROUP BY

  country_name

ORDER BY

  average DESC

;

        """

response1 = wbed.query_to_pandas_safe(query1)

response1.head(50)

response1.plot.hist()
le = response1[response1['average']<10]

le



me = response1[response1['average']>20]

me
query2 = """

SELECT

  series_code,

  indicator_name

FROM

  `bigquery-public-data.world_bank_intl_education.series_summary`

WHERE

  indicator_name like "%education%"

 ;

        """

response2 = query_job.result()

print(response2)
bq_assistant.table_schema("country_summary")

query3 = """

SELECT * from 

  `bigquery-public-data.world_bank_intl_education.country_summary`

WHERE

  short_name = "India"

 ;

        """

df = wbed.query_to_pandas_safe(query3)

print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))
df.head(5)
bq_assistant.table_schema("international_education")



query4 = """

 SELECT indicator_name, value,year from 

   `bigquery-public-data.world_bank_intl_education.international_education`

 WHERE

   country_name = "India"

  ;

         """

df2 = wbed.query_to_pandas_safe(query4)
df2