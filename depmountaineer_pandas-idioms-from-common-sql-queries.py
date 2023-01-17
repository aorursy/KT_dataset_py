"""
Created on Wed Jul 25 22:18:06 2018

These examples are inspired by: https://www.itl.nist.gov/div897/ctg/dm/sql_examples.htm

SQL queries followed by pandas operations that approximate them

This would help one find the proper pandas incantations to mimic common data wrangling operations

I can't take credit for most of below; came from lots of internet searching and some trial-and-error to make it work

@author: tdvance
"""

import pandas as pd
###Create a new table

#CREATE TABLE STATION
#(ID INTEGER PRIMARY KEY,
#CITY CHAR(20),
#STATE CHAR(2),
#LAT_N REAL,
#LONG_W REAL); 

fields = ['ID', 'CITY', 'STATE', 'LAT_N', 'LONG_W']
station = pd.DataFrame(columns=fields)
print(station)
#primary id to be handled when inserting records
###Add data to table

#INSERT INTO STATION VALUES (13, 'Phoenix', 'AZ', 33, 112);
#INSERT INTO STATION VALUES (44, 'Denver', 'CO', 40, 105);
#INSERT INTO STATION VALUES (66, 'Caribou', 'ME', 47, 68); 

records = [
        [13, 'Phoenix', 'AZ', 33, 112],
        [44, 'Denver', 'CO', 40, 105],
        [66, 'Caribou', 'ME', 47, 68]
        ]

primary_keys = [ x[fields.index('ID')] for x in records]

station = station.append(pd.DataFrame(index=primary_keys, columns=fields, data=records), sort=False, ignore_index=False)
print(station)
###Query the table by a single condition

#SELECT * FROM STATION
#WHERE LAT_N > 39.7; 
selected = station[station['LAT_N'] > 39.7][fields]
print(selected)
###Show selected columns of the table

#SELECT ID, CITY, STATE FROM STATION;

selected_fields = ['ID', 'CITY', 'STATE']
selected = station[selected_fields]
print(selected)
###Query the table by a single condition and show selected columns of the table

#SELECT ID, CITY, STATE FROM STATION
#WHERE LAT_N > 39.7; 
selected_fields = ['ID', 'CITY', 'STATE']
selected = station[station['LAT_N'] > 39.7][selected_fields]
print(selected)
###Create another table

#CREATE TABLE STATS
#(ID INTEGER,
#MONTH INTEGER,
#TEMP_F REAL,
#RAIN_I REAL,
#PRIMARY KEY (ID, MONTH)); 

fields = ['ID', 'MONTH', 'TEMP_F', 'RAIN_I'] 

stats = pd.DataFrame(columns=fields)
print(stats)
###Add data to the table

#INSERT INTO STATS VALUES (13, 1, 57.4, 0.31);
#INSERT INTO STATS VALUES (13, 7, 91.7, 5.15);
#INSERT INTO STATS VALUES (44, 1, 27.3, 0.18);
#INSERT INTO STATS VALUES (44, 7, 74.8, 2.11);
#INSERT INTO STATS VALUES (66, 1, 6.7, 2.10);
#INSERT INTO STATS VALUES (66, 7, 65.8, 4.52); 

records = [
        [13, 1, 57.4, 0.31],
        [13, 7, 91.7, 5.15],
        [44, 1, 27.3, 0.18],
        [44, 7, 74.8, 2.11],
        [66, 1, 6.7, 2.10],
        [66, 7, 65.8, 4.52]
        ]

primary_keys = [ (x[fields.index('ID')], x[fields.index('MONTH')]) for x in records]
                                                
stats = stats.append(pd.DataFrame(index=primary_keys, columns=fields, data=records), sort=False, ignore_index=False)
print(stats)
###Query two tables simultaneously; requires constructing a cross product in pandas, an idiom unto itself

#SELECT * FROM STATION, STATS
#WHERE STATION.ID = STATS.ID; 

tmp = '_temp_key_to_delete'
station[tmp] = 1
stats[tmp] = 1
cross_product = pd.merge(station, stats, on=tmp, how='outer', suffixes=('_station', '_stats')).drop(tmp, axis=1)
station.drop(tmp, axis=1, inplace=True)
stats.drop(tmp, axis=1, inplace=True)
selected = cross_product[cross_product['ID_station'] == cross_product['ID_stats']]
print(selected)
###Select certain columns and sort the result

#SELECT MONTH, ID, RAIN_I, TEMP_F
#FROM STATS
#ORDER BY MONTH, RAIN_I DESC;

selected_fields = ['MONTH', 'ID', 'RAIN_I', 'TEMP_F']
selected = stats[selected_fields].sort_values(by=['MONTH', 'RAIN_I'], axis=0, ascending=[True, False])
print(selected)
###Query two tables simultaneously by multiple conditions and sort the result

#SELECT LAT_N, CITY, TEMP_F
#FROM STATS, STATION
#WHERE MONTH = 7
#AND STATS.ID = STATION.ID
#ORDER BY TEMP_F; 
selected_fields = ['LAT_N', 'CITY', 'TEMP_F']
tmp = '_temp_key_to_delete'
stats[tmp] = 1
station[tmp] = 1
cross_product = pd.merge(stats, station, on=tmp, how='outer', suffixes=('_stats', '_station')).drop(tmp, axis=1)
stats.drop(tmp, axis=1, inplace=True)
station.drop(tmp, axis=1, inplace=True)
#parenthes in conditional below are REQUIRED
selected = cross_product[(cross_product['MONTH'] == 7) & (cross_product['ID_stats'] == cross_product['ID_station'])].sort_values(by=['TEMP_F'], axis=0, ascending=[True])[selected_fields]
print(selected)
###group and aggregate the table

#SELECT MAX(TEMP_F), MIN(TEMP_F), AVG(RAIN_I), ID
#FROM STATS
#GROUP BY ID; 
selected = stats.groupby(by='ID').agg({'TEMP_F':['max', 'min'], 'RAIN_I':'mean', 'ID':'first'})
print(selected)
###double query: query one table based on a query to another table

#SELECT * FROM STATION
#WHERE 50 < (SELECT AVG(TEMP_F) FROM STATS
#WHERE STATION.ID = STATS.ID); 
selected_fields = list(station.columns)
selected_fields[selected_fields.index('ID')] = 'ID_station'
grp = stats.groupby(by='ID').agg({'ID':'first', 'TEMP_F':'mean'})
tmp = '_temp_key_to_delete'
station[tmp] = 1
grp[tmp] = 1
cross_product = pd.merge(station, grp, on=tmp, how='outer', suffixes=('_station', '_stats')).drop(tmp, axis=1)
station.drop(tmp, axis=1, inplace=True)
selected = cross_product[(50 < cross_product['TEMP_F']) & (cross_product['ID_stats'] == cross_product['ID_station'])][selected_fields]
print(selected)
###Create a new table based on an existing table; can't quite make it a view

#CREATE VIEW METRIC_STATS (ID, MONTH, TEMP_C, RAIN_C) AS
#SELECT ID,
#MONTH,
#(TEMP_F - 32) * 5 /9,
#RAIN_I * 0.3937
#FROM STATS; 

fields = ['ID', 'MONTH', 'TEMP_C', 'RAIN_C']
metric_stats = pd.DataFrame(columns=fields)
metric_stats['ID'] = stats['ID']
metric_stats['MONTH'] = stats['MONTH']
metric_stats['TEMP_C'] = (stats['TEMP_F'] - 32) * 5.0/9.0
metric_stats['RAIN_C'] = stats['RAIN_I'] * 0.3937
#not really a view but a whole new table

#SELECT * FROM METRIC_STATS;
print(metric_stats)
###Query the new table on two conditions and sort the result

#SELECT * FROM METRIC_STATS
#WHERE TEMP_C < 0 AND MONTH = 1
#ORDER BY RAIN_C; 
selected = metric_stats[(metric_stats['TEMP_C'] < 0) & (metric_stats['MONTH'] == 1)].sort_values(by=['RAIN_C'], axis=0, ascending=[True])
print(selected)
###Update a column of a table

#UPDATE STATS SET RAIN_I = RAIN_I + 0.01; 

stats['RAIN_I'] = stats['RAIN_I'] + 0.01

#SELECT * FROM STATS;
print(stats)
###Update a cell of a table

#UPDATE STATS SET TEMP_F = 74.9
#WHERE ID = 44
#AND MONTH = 7;

row_indexer = stats[(stats['ID'] == 44) & (stats['MONTH'] == 7)].index
stats.loc[row_indexer, 'TEMP_F'] = 74.9

#SELECT * FROM STATS;
print(stats)
###Update cells of a column satisfying certain criteria (two cells will be updated)

#UPDATE STATS SET RAIN_I = 4.50
#WHERE ID = 44; 
row_indexer = stats[stats['ID'] == 44].index
stats.loc[row_indexer, 'RAIN_I'] = 4.50

#SELECT * FROM STATS;
print(stats)
### delete some rows from the table

#DELETE FROM STATS
#WHERE MONTH = 7
#OR ID IN (SELECT ID FROM STATION
#WHERE LONG_W < 90);

selected_ids = list(station[station['LONG_W'] < 90]['ID'])
row_indexer = stats[(stats['MONTH'] == 7) | (stats['ID'].isin(selected_ids))].index
stats = stats.drop(index = row_indexer)

#SELECT * FROM STATS;
print(stats)
###delete some rows from the table

#DELETE FROM STATION WHERE LONG_W < 90; 
row_indexer = station[station['LONG_W'] < 90].index
station = station.drop(index = row_indexer)

#SELECT * FROM STATION;
print(station)