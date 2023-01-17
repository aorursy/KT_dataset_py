import bq_helper
aq_data = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='openaq')
aq_data.list_tables()
aq_data.table_schema('global_air_quality')
aq_data.head('global_air_quality')
aq_data.head('global_air_quality', selected_columns="location, city, country, unit", num_rows=15)
query = """ SELECT DISTINCT country from `bigquery-public-data.openaq.global_air_quality` where unit != "%ppm%" """
aq_data.estimate_query_size(query)
non_ppm_countries = aq_data.query_to_pandas_safe(query)
non_ppm_countries.to_csv('non_ppm_countries.csv')
query = """ SELECT location, country, pollutant 
            from `bigquery-public-data.openaq.global_air_quality` 
            where value = 0.00 """
aq_data.estimate_query_size(query)
zero_value_pollutant = aq_data.query_to_pandas_safe(query)
zero_value_pollutant.to_csv('zero_value_pollutant.csv')