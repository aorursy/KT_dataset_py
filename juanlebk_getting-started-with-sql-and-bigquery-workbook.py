import bq_helper
# create a helper object for our bigquery dataset
historical_air_quality = bq_helper.BigQueryHelper(active_project="bigquery-public-data", \
                                                  dataset_name='epa_historical_air_quality')
# print a list of all the tables in the epa_historical_air_quality dataset
historical_air_quality.list_tables()
# print information on all the columns in the "air_quality_annual_summary" table
# in the epa_historical_air_quality dataset
historical_air_quality.table_schema('air_quality_annual_summary')
# preview the first couple lines of the 'air_quality_annual_summary' table
historical_air_quality.head('air_quality_annual_summary')
# Preview the first ten entries in the state_code and parameter_name columns
historical_air_quality.head(table_name='air_quality_annual_summary', \
                            selected_columns=["state_code", "parameter_name"], num_rows=5)
# this query looks in the air_quality_annual_summary table, then get parameter_name column
query = """SELECT parameter_name 
            FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary`
            """
historical_air_quality.estimate_query_size(query)
query = """SELECT state_code
            FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary`
            WHERE PARAMETER_NAME = 'Sample Min Baro Pressure'"""
historical_air_quality.estimate_query_size(query)
# only run this query if it's less than 40MB
historical_air_quality.query_to_pandas_safe(query=query, max_gb_scanned=0.04)
state_code = historical_air_quality.query_to_pandas(query=query)
state_code.describe()
state_code.head()
state_code.tail()
state_code.columns
state_code.isin(['06']).sum(axis=0)
state_code.query('state_code=="CC"').state_code.count()
state_code.query('state_code=="CC"').count()
state_code.loc[state_code.state_code == '06'].count()
state_code.loc[state_code.state_code == '06', 'state_code'].count()