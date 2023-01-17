import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
#For plots
import matplotlib.pyplot as plt
tf = BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')
tf_tables = tf.list_tables()
print("There are "+str(len(tf_tables))+" tables in the dataset")
print(tf_tables)
tf.table_schema('accident_2016')
#Please notice the amount of information returned from schema
tf.table_schema('vehicle_2016')
#Please notice the amount of information returned from schema
sql1="""SELECT COUNT(consecutive_number), EXTRACT(hour FROM timestamp_of_crash) AS hour
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY hour
    ORDER BY hour
    """
tf.estimate_query_size(sql1)
hours = tf.query_to_pandas_safe(sql1)
hours.shape
print(hours)
plt.plot(hours.f0_)
plt.xticks(hours.hour)
plt.title("Number of Accidents by hour of Day")
sql2="""SELECT COUNTIF(hit_and_run!='0') as har, registration_state_name
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
    GROUP BY registration_state_name
    ORDER BY har DESC
    """
tf.estimate_query_size(sql2)
har = tf.query_to_pandas_safe(sql2)
har.shape
har.head(5)
har.tail(10)