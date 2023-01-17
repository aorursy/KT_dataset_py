# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex5 import *



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

chicago_taxi_helper = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                               dataset_name="chicago_taxi_trips")
# Your code here to find the table name

chicago_taxi_helper.list_tables()
# write the table name as a string below

table_name = chicago_taxi_helper.list_tables()[0]



q_1.check()
# q_1.solution()
chicago_taxi_helper.head(table_name)
q_2.solution()
rides_per_year_query = """

        SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, COUNT(1) num_trips

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

        GROUP BY year

        ORDER BY year

"""



chicago_taxi_helper.estimate_query_size(rides_per_year_query)

rides_per_year_result = chicago_taxi_helper.query_to_pandas_safe(rides_per_year_query, max_gb_scanned=25)



print(rides_per_year_result)

q_3.check()
#q_3.hint()

#q_3.solution()
rides_per_month_query = """

        SELECT EXTRACT(MONTH FROM trip_start_timestamp) as month, COUNT(1) num_trips

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`       

        WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017 

        GROUP BY month

        ORDER BY month

"""



rides_per_month_result = chicago_taxi_helper.query_to_pandas_safe(rides_per_month_query)



print(rides_per_month_result)

q_4.check()
#q_4.hint()

#q_4.solution()
speeds_query = """

        WITH RelevantRides AS

        (       

                SELECT EXTRACT(HOUR FROM trip_start_timestamp) as hour_of_day, trip_miles, trip_seconds

                FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                WHERE trip_start_timestamp BETWEEN '2017-01-01' AND '2017-07-01' 

                        AND trip_seconds > 0 AND trip_miles > 0

        )

        SELECT hour_of_day, COUNT(1) AS num_trips, 3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

        FROM RelevantRides

        GROUP BY hour_of_day

        ORDER BY hour_of_day

        """



# Set high max_gb_scanned because this query looks at more data

speeds_result = chicago_taxi_helper.query_to_pandas_safe(speeds_query, max_gb_scanned=20)



print(speeds_result)

q_5.check()
#q_5.hint()

# q_5.solution()
# q_6.solution()