import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# create a helper object for our bigquery dataset
traffic = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                        dataset_name = "nhtsa_traffic_fatalities")

# declaring table name variable
table_name = "accident_2015"
#print a list of all the tables in the nhtsa_traffic_fatalities
traffic.list_tables()

# print information on all the columns in the "accident_2015" table
# in the nhtsa_traffic_fatalities dataset
traffic.table_schema(table_name)

#Preview the first couple lines of the "accident_2015" table
traffic.head(table_name)

# preview the first ten entries in the by column of the accident_2015 table
traffic.head(table_name, selected_columns="state_number", num_rows=15)

query ="""SELECT state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            """
#traffic.estimate_query_size(query)
result = traffic.query_to_pandas_safe(query)
result.to_csv("accident_2015_state_name.csv")
