import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bq
t_f = bq.BigQueryHelper(active_project= 'bigquery-public-data',
                       dataset_name= 'nhtsa_traffic_fatalities')
# t_f.list_tables()
# t_f.table_schema('accident_2015')
query = """
        select
            extract(hour from timestamp_of_crash) as hour,
            count(consecutive_number) as total_accidents
        from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        group by hour
        order by total_accidents desc
        """
t_f.query_to_pandas(query)
# t_f.table_schema('vehicle_2015')
query = """
        select 
            state_number,
            count(state_number) as count
        from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        where hit_and_run = 'Yes'
        group by state_number
        order by count desc
        """
t_f.query_to_pandas(query)