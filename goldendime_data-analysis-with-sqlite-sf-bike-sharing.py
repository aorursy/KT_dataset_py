# Importing all the essential libraries for our data analysis with SQLite

import sqlite3

import pandas as pd
conn = sqlite3.connect('../input/database.sqlite')
#There are four connected tables: 1)Station info, 2)Status with timestamps, 3)Trips & 4)Weather

#Let's see all the columns and first 10 rows of the Station table

pd.read_sql('''

    SELECT *

    FROM station

    LIMIT 10;

''', con=conn)
pd.read_sql('''

    SELECT city, 

    SUM(dock_count) AS total_capacity, 

    COUNT(name) AS station_count, 

    ROUND(SUM(dock_count)/COUNT(name), 2) AS average_capacity_per_station

    FROM station

    GROUP BY city

    ORDER BY station_count DESC;

''', con=conn, index_col='city')
pd.read_sql('''

    SELECT city,

           CASE

           -- m/d/yyyy

           WHEN (length(installation_date) = 8 AND substr(installation_date,2,1) = '/') 

           THEN substr(installation_date,5,4)||'-0'||substr(installation_date,1,1)||'-0'||substr(installation_date,3,1)

           -- m/dd/yyyy

           WHEN (length(installation_date) = 9 AND substr(installation_date,2,1) = '/') 

           THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,1)||'-'||substr(installation_date,3,2)

           -- mm/d/yyyy

           WHEN (length(installation_date) = 9 AND substr(installation_date,3,1) = '/') 

           THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,2)||'-'||substr(installation_date,4,1)

           -- mm/dd/yyyy

           WHEN (length(installation_date) = 10 AND substr(installation_date,3,1) = '/') 

           THEN substr(installation_date,7,4)||'-'||substr(installation_date,1,2)||'-'||substr(installation_date,4,2)

           ELSE installation_date

           END AS installed_date, 

           SUM(dock_count) AS total_dock_ct,

           COUNT(name) AS station_count

    FROM station

    GROUP BY 1,2

    ORDER BY 2,1;

''', con=conn)
'''SELECT city, 

       CAST(SUBSTR(installation_date, LENGTH(installation_date)-3) || '-' ||  

       SUBSTR(installation_date, 0, INSTR(installation_date,'/')) || '-' ||

       REPLACE((SUBSTR(installation_date, INSTR(installation_date,'/') + 1, 2)), '/', '') AS DATE) AS installed_date

       SUM(dock_count) as station_count,

FROM station

GROUP BY 1,2

ORDER BY 2,1;'''
pd.read_sql('''

    WITH t1 AS (SELECT city,

                       CASE

                       -- m/d/yyyy

                       WHEN (length(installation_date) = 8 AND substr(installation_date,2,1) = '/') 

                       THEN substr(installation_date,5,4)||'-0'||substr(installation_date,1,1)||'-0'||substr(installation_date,3,1)

                       -- m/dd/yyyy

                       WHEN (length(installation_date) = 9 AND substr(installation_date,2,1) = '/') 

                       THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,1)||'-'||substr(installation_date,3,2)

                       -- mm/d/yyyy

                       WHEN (length(installation_date) = 9 AND substr(installation_date,3,1) = '/') 

                       THEN substr(installation_date,6,4)||'-0'||substr(installation_date,1,2)||'-'||substr(installation_date,4,1)

                       -- mm/dd/yyyy

                       WHEN (length(installation_date) = 10 AND substr(installation_date,3,1) = '/') 

                       THEN substr(installation_date,7,4)||'-'||substr(installation_date,1,2)||'-'||substr(installation_date,4,2)

                       ELSE installation_date

                       END AS installed_date, 

                       SUM(dock_count) AS total_dock_ct,

                       COUNT(name) AS station_count

                FROM station

                GROUP BY 1,2

                ORDER BY 2,1)

    SELECT CASE 

        WHEN month = '2013-08-01'

        THEN 'AUG13'

        WHEN month > '2013-08-01'

        THEN 'after_AUG13'

        ELSE 'before_AUG13'

        END AS installation_month,

        SUM(total_dock_ct) AS dock_ct,

        SUM(station_count) AS station_ct

    FROM (SELECT DATE(installed_date, 'start of month') as month,

                 total_dock_ct, 

                 station_count

         FROM t1

         ) AS innerquery

    GROUP BY 1;

''', con=conn, index_col='installation_month')
#Let's see all the columns and first 10 rows of the Status table which updated its status every minute

pd.read_sql('''

     SELECT * 

     FROM status

     LIMIT 10;

''', con=conn)
pd.read_sql('''

     SELECT ROUND(AVG(status.bikes_available),2) AS avg_available_bikes, 

            ROUND(AVG(status.docks_available),2) AS free_dock_count,

            station.dock_count AS max_dock_capacity,

            station.name

     FROM status

     INNER JOIN station

     ON status.station_id = station.id

     GROUP BY name

     ORDER BY 1, 2 DESC

     LIMIT 10;

''', con=conn)
pd.read_sql('''

    SELECT ROUND(AVG(status.bikes_available),2) AS avg_available_bikes, 

           ROUND(AVG(status.docks_available),2) AS free_dock_count,

           COUNT(*) AS num_occurrences,

           DATE(SUBSTR(time, 1,4) || '-' || SUBSTR(time, 6,2) || '-' || 

                SUBSTR(time, 9,2) ||  SUBSTR(time, 11,9), 'start of month') AS month 

    FROM status

    INNER JOIN station

    ON status.station_id = station.id

    WHERE station.name = '2nd at Folsom'

    GROUP BY station.name, month

    ORDER BY 2 DESC;

''', con=conn)
pd.read_sql('''

    SELECT *

    FROM trip

    LIMIT 10

''', con=conn)
pd.read_sql('''

    SELECT duration/60/60 AS duration_hr, COUNT(*) AS frequency

    FROM trip

    GROUP BY duration_hr

    ORDER BY duration_hr DESC 

    LIMIT 15;

''', con=conn)
pd.read_sql('''

    SELECT duration/60/60 AS duration_hr, COUNT(*) AS frequency

    FROM trip

    WHERE duration_hr <= 15

    GROUP BY duration_hr

    ORDER BY duration_hr DESC;

''', con=conn)
pd.read_sql('''

    SELECT COUNT(*) AS num_count,

           trip.start_station_name

    FROM trip

    INNER JOIN station

    ON station.id = start_station_id

    WHERE station.city = 'San Francisco'

    GROUP BY 2

    ORDER BY 1 DESC

    LIMIT 5;

''', con=conn)
pd.read_sql('''

    SELECT COUNT(*) AS num_count, 

           trip.end_station_name

    FROM trip

    INNER JOIN station

    ON station.id = start_station_id

    WHERE station.city = 'San Francisco'

    GROUP BY 2

    ORDER BY 1 DESC

    LIMIT 5;

''', con=conn)
pd.read_sql('''

    SELECT COUNT(*) AS num_count,

           ROUND(AVG(duration/60), 2) AS avg_duration_mins,

           trip.start_station_name,

           trip.end_station_name

    FROM trip

    INNER JOIN station

    ON station.id = start_station_id

    WHERE station.city = 'San Francisco' AND duration/60/60 <= 15

    GROUP BY 3, 4

    ORDER BY 1 DESC

    LIMIT 15;

''', con=conn)
pd.read_sql('''

    SELECT AVG(avg_duration_mins) AS avg_duration_for_most_pop

    FROM (SELECT COUNT(*) AS num_count,

                 ROUND(AVG(duration/60), 2) AS avg_duration_mins,

                 trip.start_station_name,

                 trip.end_station_name

          FROM trip

          INNER JOIN station

          ON station.id = start_station_id

          WHERE station.city = 'San Francisco' AND duration/60/60 <= 15

          GROUP BY 3, 4

          HAVING COUNT(*) > 2000

          ORDER BY 1 DESC) AS most_popular_trips

''', con=conn)
pd.read_sql('''

    SELECT COUNT(*) AS num_count,

           ROUND(AVG(duration/60), 2) AS avg_duration_mins,

           trip.start_station_name,

           trip.end_station_name

    FROM trip

    INNER JOIN station

    ON station.id = start_station_id

    WHERE city='San Francisco' AND duration/60/60 <= 15

    GROUP BY 3, 4

    HAVING COUNT(*) > 50

    ORDER BY 2 DESC

    LIMIT 15;

''', con=conn)
pd.read_sql('''

    SELECT subscription_type,

           COUNT(*) AS sub_type_ct

    FROM trip

    GROUP BY 1;

''', con=conn)
pd.read_sql('''

    WITH t1 AS (SELECT DATE(CASE

                -- m/d/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 8 AND substr(start_date,2,1) = '/') 

               THEN substr(start_date,5,4)||'-0'||substr(start_date,1,1)||'-0'||substr(start_date,3,1)

               -- m/dd/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,2,1) = '/') 

               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,1)||'-'||substr(start_date,3,2)

               -- mm/d/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,3,1) = '/') 

               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,2)||'-'||substr(start_date,4,1)

               -- mm/dd/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 10 AND substr(start_date,3,1) = '/') 

               THEN substr(start_date,7,4)||'-'||substr(start_date,1,2)||'-'||substr(start_date,4,2)

               ELSE start_date

               END) AS trip_date,

               subscription_type, 

               (duration / 60) AS duration_min

        FROM trip

        INNER JOIN station

        ON station.id = start_station_id

        WHERE city='San Francisco' AND duration/60/60 <= 15)

    SELECT CASE 

           WHEN (trip_date IN (DATE(trip_date, 'weekday 6'), DATE(trip_date, 'weekday 0')))

           THEN 'weekends'

           ELSE 'weekdays'

           END AS weekday,

           subscription_type,

           ROUND(AVG(duration_min), 2) AS avg_dur_min

           

    FROM t1

    GROUP BY 1,2;

''', con=conn)
pd.read_sql('''

    WITH t1 AS (SELECT DATE(CASE

                -- m/d/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 8 AND substr(start_date,2,1) = '/') 

               THEN substr(start_date,5,4)||'-0'||substr(start_date,1,1)||'-0'||substr(start_date,3,1)

               -- m/dd/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,2,1) = '/') 

               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,1)||'-'||substr(start_date,3,2)

               -- mm/d/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 9 AND substr(start_date,3,1) = '/') 

               THEN substr(start_date,6,4)||'-0'||substr(start_date,1,2)||'-'||substr(start_date,4,1)

               -- mm/dd/yyyy

               WHEN ((INSTR(start_date, ' ')-1) = 10 AND substr(start_date,3,1) = '/') 

               THEN substr(start_date,7,4)||'-'||substr(start_date,1,2)||'-'||substr(start_date,4,2)

               ELSE start_date

               END) AS trip_date,

               subscription_type, 

               (duration / 60) AS duration_min

        FROM trip

        INNER JOIN station

        ON station.id = start_station_id

        WHERE city='San Francisco' AND duration/60/60 <= 15)

    SELECT CASE 

           WHEN trip_date = (DATE(trip_date, 'weekday 1'))

           THEN '1 - Monday'

           WHEN trip_date = (DATE(trip_date, 'weekday 2'))

           THEN '2 - Tuesday'

           WHEN trip_date = (DATE(trip_date, 'weekday 3'))

           THEN '3 - Wednesday'

           WHEN trip_date = (DATE(trip_date, 'weekday 4'))

           THEN '4 - Thursday'

           WHEN trip_date = (DATE(trip_date, 'weekday 5'))

           THEN '5 - Friday'

           WHEN trip_date = (DATE(trip_date, 'weekday 6'))

           THEN '6 - Saturday'

           ELSE '7 - Sunday'

           END AS weekday,

           subscription_type,

           ROUND(AVG(duration_min), 2) AS avg_dur_min

           

    FROM t1

    GROUP BY 1,2

    ORDER BY 2,1;

''', con=conn)
pd.read_sql('''

    SELECT COUNT(*) AS num_count,

           ROUND(AVG(duration/60), 2) AS avg_duration_mins,

           trip.start_station_name,

           trip.end_station_name

    FROM trip

    INNER JOIN station

    ON station.id = start_station_id

    WHERE station.city = 'San Francisco' AND duration/60/60 <= 15 AND subscription_type = 'Customer'

    GROUP BY 3, 4

    ORDER BY 1 DESC

    LIMIT 15;

''', con=conn)