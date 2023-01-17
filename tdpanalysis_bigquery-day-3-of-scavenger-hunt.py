import bq_helper as bq
us_traffic_fatalities = bq.BigQueryHelper(active_project="bigquery-public-data",
                                         dataset_name="nhtsa_traffic_fatalities")
query = """SELECT COUNT(consecutive_number) AS count,
                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_accident
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
           GROUP BY hour_of_accident
           ORDER BY count DESC
"""
us_traffic_fatalities.estimate_query_size(query)
accident_hours = us_traffic_fatalities.query_to_pandas_safe(query)
accident_hours
accident_hours = accident_hours.sort_values('hour_of_accident')
accident_hours.plot.bar('hour_of_accident','count')
query2 = """SELECT registration_state_name AS state,
                COUNT(hit_and_run) AS count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY state
            ORDER BY count DESC
"""
us_traffic_fatalities.estimate_query_size(query2)
hit_and_runs_by_state = us_traffic_fatalities.query_to_pandas_safe(query2)
hit_and_runs_by_state
hit_and_runs_by_state_sub = hit_and_runs_by_state.iloc[1:]
hit_and_runs_by_state_sub.sort_values('count').plot.barh('state','count',figsize=(8,12))