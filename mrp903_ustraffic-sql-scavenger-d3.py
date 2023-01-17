import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
ustraffic = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', 
                                     dataset_name = 'nhtsa_traffic_fatalities')
ustraffic.list_tables()
ustraffic.table_schema(table_name='accident_2016')
query_1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as Hour, COUNT(EXTRACT(HOUR FROM timestamp_of_crash)) AS Crash
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             GROUP BY Hour
             ORDER BY Crash DESC
          """
ustraffic.estimate_query_size(query_1)
crash_hours = ustraffic.query_to_pandas(query=query_1)
crash_hours
ustraffic.table_schema('vehicle_2016')
query_2 = """SELECT registration_state_name, COUNT(*) as HIT
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             WHERE hit_and_run = 'Yes'
             GROUP  BY registration_state_name
             ORDER BY HIT DESC
          """
ustraffic.estimate_query_size(query_2)


hit_and_run = ustraffic.query_to_pandas_safe(query=query_2, max_gb_scanned=1)
hit_and_run
