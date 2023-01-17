import numpy as np # linear algebra
import os
import pandas as pd # data processing
import sqlite3 as sq# enable creation and handling of a local db 
# set-up a connection to a newly named flock.db. 
# the connection will be called "conn"
conn = sq.connect('flock.db')
cursor = conn.cursor()
for file in os.listdir('../input'):     # For all files in our directory
    df = pd.read_csv('../input/'+file)  # Read each CSV file
    df.to_sql(file[:-4],conn)           # Create the read file as a table in the database.
# Create the query string to then feed into the Pandas read_sql function. 
q1 = ('SELECT airc.aircraft_type, COUNT(indv.flight_id) AS number_of_flights '
      'FROM individual_flights AS indv '
      'JOIN aircraft AS airc '
      'ON indv.aircraft_id = airc.aircraft_id '
      'GROUP BY 1 '
      'ORDER BY number_of_flights DESC ')

# The function read_sql takes a query string and a database connection, and performs the query. 
r1 = pd.read_sql(q1,conn)

#We only need the first result, so we use iloc[[0]].
print(r1.iloc[[0]])
#Answer form A. 

q2 = ('SELECT airc.aircraft_type, CAST((COUNT(indv.flight_id)*airc.capacity) AS FLOAT)/CAST(airc.cost AS FLOAT) AS passengers_per_cost '
      'FROM individual_flights AS indv '
      'JOIN aircraft AS airc '
      'ON indv.aircraft_id = airc.aircraft_id '
      'GROUP BY 1 '
      'ORDER BY passengers_per_cost DESC')

#Answer form B.
q3 = ('SELECT airc.aircraft_type, CAST(airc.cost AS FLOAT)/CAST((COUNT(indv.flight_id)*airc.capacity) AS FLOAT) AS cost_per_passenger '
      'FROM individual_flights AS indv '
      'JOIN aircraft AS airc '
      'ON indv.aircraft_id = airc.aircraft_id '
      'GROUP BY 1 '
      'ORDER BY cost_per_passenger ASC')

r2 = pd.read_sql(q2,conn)

r3 = pd.read_sql(q3,conn)

print('Form A, passengers/GBP: \n')
print(r2.iloc[:3])

print('\n')
print('Form B, GBP/passenger: \n')
print(r3.iloc[:3])
# Adding an OR in the JOIN ON clause will account for inbound and outbound flights all alike.
q4 = ("""
    SELECT airport_name
      , SUM(n_passenger_per_aircraft) AS n_passengers
      FROM (
          SELECT indvf.aircraft_id
          , airp.airport_name 
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_passenger_per_aircraft
          FROM individual_flights AS indvf
          JOIN airports AS airp
              ON airp.airport_code = indvf.destination_airport_code 
                  OR airp.airport_code = indvf.departure_airport_code
          JOIN aircraft AS aic
              ON aic.aircraft_id = indvf.aircraft_id    
          GROUP BY 1, 2)
      GROUP BY 1
      ORDER BY 2 DESC""")

r4 = pd.read_sql(q4,conn)

print('Number of passengers per airport (inbound and outbound) \n')
print(r4.iloc[:3])
# Only count if outbound form airport X
q5 = ("""SELECT airport_name
      , SUM(n_outbound_passengers) AS outbound_passengers
      FROM (
            SELECT indvf.aircraft_id
          , airp.airport_name 
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_outbound_passengers
          FROM individual_flights AS indvf
          JOIN airports AS airp
          ON airp.airport_code = indvf.departure_airport_code
          JOIN aircraft AS aic
          ON aic.aircraft_id = indvf.aircraft_id
          GROUP BY 1, 2)
       GROUP BY 1
       ORDER BY 2 DESC""")

# Only count if inbound form airport X
q6 = ("""SELECT airport_name
      , SUM(n_inbound_passengers) AS inbound_passengers
      FROM (
          SELECT indvf.aircraft_id
          , airp.airport_name 
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_inbound_passengers
          FROM individual_flights AS indvf
          JOIN airports AS airp
          ON airp.airport_code = indvf.destination_airport_code
          JOIN aircraft AS aic
          ON aic.aircraft_id = indvf.aircraft_id
          GROUP BY 1, 2)
      GROUP BY 1
      ORDER BY 2 DESC""")

r5 = pd.read_sql(q5,conn)
print('Number of passengers per airport (outbound) \n')
print(r5.iloc[:3])
print('\n')

r6 = pd.read_sql(q6,conn)
print('Number of passengers per airport (inbound) \n')
print(r6.iloc[:3])
q7 = ("""SELECT airport_name
      , CAST(SUM(n_passenger_per_aircraft) AS FLOAT)/CAST(airport_size AS FLOAT) AS passagers_per_m2
      FROM (
          SELECT indvf.aircraft_id
          , airp.airport_name 
          , airp.airport_size
          , COUNT(DISTINCT flight_id)*aic.capacity AS n_passenger_per_aircraft
          FROM individual_flights AS indvf
          JOIN airports AS airp
              ON airp.airport_code = indvf.destination_airport_code 
                  OR airp.airport_code = indvf.departure_airport_code
          JOIN aircraft AS aic
              ON aic.aircraft_id = indvf.aircraft_id    
          GROUP BY 1, 2, 3)
      GROUP BY 1
      ORDER BY 2 DESC""")

r7 = pd.read_sql(q7,conn)
print(r7.iloc[[0]])

q8 = ("""SELECT Airline_Name, Year, MAX(RPM_Domestic)  
    FROM (
        SELECT Airline_Code, CAST(substr(Date, -4) AS INT) AS Year
        , SUM(COALESCE(RPM_Domestic,0)) AS RPM_Domestic
        FROM flight_summary_data 
        GROUP BY 1, 2) AS subq
    JOIN
        airlines 
    ON subq.airline_code = airlines.airline_code
    GROUP BY 1""")

r8 = pd.read_sql(q8,conn)

print(r8)

print('\n')

q9 = ("""SELECT Airline_Name, Year, MAX(RPM_International)   
    FROM (
        SELECT Airline_Code, CAST(substr(Date, -4) AS INT) AS Year
        , SUM(COALESCE(RPM_International,0)) AS RPM_International
        FROM flight_summary_data 
        GROUP BY 1, 2) AS subq
    JOIN
        airlines 
    ON subq.airline_code = airlines.airline_code
    GROUP BY 1""")

r9 = pd.read_sql(q9,conn)

print(r9)
q10 = ("""SELECT Airline_Name, Year, MAX(RPM_Total)   
    FROM (
        SELECT Airline_Code, CAST(substr(Date, -4) AS INT) AS Year
        , SUM((RPM_Domestic + COALESCE(RPM_International,0))) AS RPM_Total
        FROM flight_summary_data 
        GROUP BY 1, 2) AS subq
    JOIN
        airlines 
    ON subq.airline_code = airlines.airline_code
    GROUP BY 1""")


r10 = pd.read_sql(q10,conn)

print(r10)
q11 = ("""SELECT Airline_Name, SUM((RPM_Domestic + COALESCE(RPM_International,0))) AS RPM_Total
        FROM flight_summary_data 
        JOIN airlines
        ON flight_summary_data.airline_code = airlines.airline_code
        GROUP BY 1 ORDER BY RPM_Total""")

r11 = pd.read_sql(q11,conn)

print(r11)
q12 = """SELECT SUM(passengers_domestic) AS total_passengers_domestic
, SUM(passengers_international) AS total_passengers_international
, CAST(substr(Date, -4) AS INT) AS Year
, CAST(substr(Date, 4, 2) AS INT) AS Month 
FROM flight_summary_data WHERE Airport_Code = "FKT" 
GROUP BY 3, 4
ORDER BY Year ASC, Month ASC"""

r12 = pd.read_sql(q12,conn)

r12 = r12.set_index(['Year','Month']).diff()

print(r12.plot(figsize=(18,10)))
q15 = """SELECT SUM(passengers_domestic) AS total_passengers_domestic
, CAST(substr(Date, -4) AS INT) AS Year
, CAST(substr(Date, 4, 2) AS INT) AS Month 
FROM flight_summary_data WHERE Airport_Code = "FKT" 
GROUP BY 2, 3
ORDER BY Year ASC, Month ASC"""

q16 = """SELECT SUM(passengers_international) AS total_passengers_international
, CAST(substr(Date, -4) AS INT) AS Year
, CAST(substr(Date, 4, 2) AS INT) AS Month 
FROM flight_summary_data WHERE Airport_Code = "FKT" 
GROUP BY 2, 3
ORDER BY Year ASC, Month ASC"""

r15 = pd.read_sql(q15,conn)
r16 = pd.read_sql(q16,conn)

r15 = r15.set_index(['Year','Month']).diff()
r16 = r16.set_index(['Year','Month']).diff()

mean_q = r15.mean(axis=0)

print(mean_q)
print(r15.plot(figsize=(18,10),title='Total Domestic Passengers by month'))
print(r16.plot(figsize=(18,10),title='Total International Passengers by month'))
q13 = ("""
SELECT SUM(ASM_Domestic) AS sum_asm_domestic
, Airline_Code
, CAST(substr(Date, -4) AS INT) AS Year
FROM flight_summary_data
GROUP BY 2, 3
HAVING CAST(substr(Date, -4) AS INT) <> 2002
""")

airportsquery = ("""
SELECT * FROM airports
""")

airportsdf = pd.read_sql(airportsquery,conn)

r13 = pd.read_sql(q13,conn)
# We get the difference between each year for each airline so we can see how much they "grew" from the past year. We group by Airline_Code first. 
r13['dif'] = r13.groupby(['Airline_Code'])['sum_asm_domestic'].diff()

# Find the indexes of each of the max values for dif. It will have negative values,but since we are talking about growth they will not be included. 
# According to the way we calculate the dif, we need to exclude the year 2002 because it only has a couple of months.  
idx = r13.groupby(['Airline_Code'], sort=False)['dif'].transform(max) == r13['dif']

# We print the years with the Max dif.
print(r13[idx])
q14 = ("""
SELECT SUM(ASM_Domestic) AS sum_asm_domestic
, Airline_Code
, Airport_Code
, CAST(substr(Date, -4) AS INT) AS Year
FROM flight_summary_data
GROUP BY 2, 3, 4
HAVING (Airline_Code='FA'AND CAST(substr(Date, -4) AS INT)=2016) 
OR (Airline_Code='AA'AND CAST(substr(Date, -4) AS INT)=2012)
OR (Airline_Code='GA'AND CAST(substr(Date, -4) AS INT)=2010)
""")
# Cherrypicking with very bad practice in the query the specific years in which I obtained the max growth. This should never be made like this. 

r14 = pd.read_sql(q14,conn)
print(airportsdf)

# Set and index frame to extract the max value'd airport per airline in the desired year. 
indx2 = r14.groupby(['Airline_Code'], sort=True)['sum_asm_domestic'].transform(max) == r14['sum_asm_domestic']

# Before printing, join with the airport table to get the airport names. 
print(pd.merge(r14[indx2], airportsdf, how='inner', left_on='Airport_Code', right_on='Airport_Code')[['Airline_Code','Airport_Name','sum_asm_domestic']])