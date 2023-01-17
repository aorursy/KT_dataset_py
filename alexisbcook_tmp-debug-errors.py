from google.cloud import bigquery



# Create client object to access database

client = bigquery.Client()



query = """

        SELECT taxi_id,

            trip_start_timestamp,

            trip_end_timestamp,

            trip_seconds,

            AVG(trip_seconds) 

                OVER (

                      PARTITION BY taxi_id

                      ORDER BY trip_start_timestamp

                     ) AS trip_seconds_avg

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

        WHERE trip_start_timestamp BETWEEN '2017-05-01' AND '2017-05-02'

        """



result = client.query(query).result().to_dataframe()

result.head()