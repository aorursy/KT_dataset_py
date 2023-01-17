#https://www.kaggle.com/jamesdqp/getting-started-with-sql-and-bigquery-jamesdqp

import bq_helper



# create a helper object for our bigquery dataset

education = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "world_bank_intl_education")
#Get total indicators for Unesco source

education.list_tables()

education.query_to_pandas("""select topic, source, count(*) 

from `bigquery-public-data.world_bank_intl_education.series_summary` 

where source = 'UNESCO Institute for Statistics'

group by topic, source

order by topic, source""")

education.list_tables()

education.query_to_pandas("""SELECT d.*,i.topic, i.short_definition, i.source

FROM `bigquery-public-data.world_bank_intl_education.international_education` d

LEFT JOIN `bigquery-public-data.world_bank_intl_education.series_summary` i on i.indicator_name = d.indicator_name

WHERE source = 'UNESCO Institute for Statistics'""")
