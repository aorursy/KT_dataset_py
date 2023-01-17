from google.cloud import bigquery



client=bigquery.Client()



dataset_ref=client.dataset('nhtsa_traffic_fatalities',project='bigquery-public-data')



dataset=client.get_dataset(dataset_ref)



table_ref=dataset_ref.table('accident_2015')



table=client.get_table(table_ref)



client.list_rows(table,max_results=5).to_dataframe()
query = """

        SELECT COUNT(consecutive_number) AS num_accidents,

            EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week,

            EXTRACT(QUARTER FROM timestamp_of_crash) AS quarter,

            AVG(minute_of_ems_arrival_at_hospital) AS avg_time,

            AVG(number_of_drunk_drivers)*100 AS drunkurds,

            AVG(number_of_fatalities)*100 AS deaths,

            state_name

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week,quarter,state_name

        ORDER BY quarter,num_accidents DESC,state_name

        """



safe_config=bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job=client.query(query,job_config=safe_config)



accidents_by_day=query_job.to_dataframe()

accidents_by_day