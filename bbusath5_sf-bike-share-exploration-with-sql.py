import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
db = sqlite3.connect('../input/database.sqlite')

def run_query(query):
    return pd.read_sql_query(query, db)
query='SELECT name FROM sqlite_master;'

run_query(query)
query = 'SELECT * FROM trip LIMIT 3;'

run_query(query)
query = '''
SELECT *
FROM trip
ORDER BY duration DESC
LIMIT 1
'''

run_query(query)
query = '''
SELECT *
FROM trip
ORDER BY duration DESC
LIMIT 10
'''

run_query(query)
query = '''
SELECT count(*)
AS \'Long Trips\'
FROM trip 
WHERE 
duration >= 60*60*24;
'''
#60 seconds in a minute, 60 minutes in an hour, 24 hours in a day

run_query(query)
query = '''
SELECT subscription_type, count(*) AS count
FROM trip
GROUP BY subscription_type
'''

df = pd.read_sql_query(query, db)

labels = ['Casual', 'Subscriber']
sizes = df['count']
colors = ['lightblue', 'lightgreen']
explode = (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, )
plt.title('Subscribed vs Unsubscribed Riders')
plt.axis('equal')
plt.show()

query = '''
SELECT subscription_type, AVG(duration)/60 AS 'Average Duration'
FROM trip
GROUP BY subscription_type
''' 
#since duration is in seconds, we will convert to minutes
run_query(query)
query = '''
SELECT station.name AS Station, count(*) AS Count
FROM station
INNER JOIN trip
ON station.id = trip.start_station_id
GROUP BY station.name
ORDER BY count DESC
LIMIT 5
''' 

run_query(query)

#there are 1047142 total status readings for each station

query = '''
SELECT station.name AS Station, count(*) AS 'Total Empty Readings'
FROM station

INNER JOIN status
ON status.station_id=station.id
WHERE status.bikes_available=0
GROUP BY station.name

ORDER BY count(*) DESC
LIMIT 10

''' 

run_query(query)
query = '''
SELECT bikes_available AS 'Bikes Available'
FROM status

''' 

df = pd.read_sql_query(query, db)
df['Bikes Available'].plot.hist(bins=27, title='Bikes Available (All Stations)', 
                                ec='black', alpha=0.5)
query='''
SELECT start_station_name, end_station_name, COUNT(*) AS Count
FROM trip
GROUP BY start_station_name, end_station_name
ORDER BY Count DESC
LIMIT 10;
  '''
run_query(query)