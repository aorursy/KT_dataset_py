import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper 
traff_fatal = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', 
                                       dataset_name = 'nhtsa_traffic_fatalities')
traff_fatal.list_tables()
traff_fatal.table_schema('accident_2015')
traff_fatal.head('accident_2015')
acc_by_hr_query = """
                  SELECT COUNT(consecutive_number) AS accident_count, 
                      EXTRACT(HOUR FROM timestamp_of_crash) AS hour
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                  GROUP BY hour
                  ORDER BY accident_count DESC
                  """

accidents_by_hr = traff_fatal.query_to_pandas_safe(acc_by_hr_query)
accidents_by_hr
hitrun_query = """
               SELECT registration_state_name, COUNT(vehicle_number) as num_hit_runs
                   FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
               WHERE hit_and_run = "Yes"
               GROUP BY registration_state_name
               ORDER BY num_hit_runs DESC
               """

hitruns_by_state = traff_fatal.query_to_pandas_safe(hitrun_query)
hitruns_by_state
