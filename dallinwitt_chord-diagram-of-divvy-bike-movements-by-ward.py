import numpy as np
import pandas as pd
from chord import Chord
#read in all the 2018 csv files
divvy_2018_1 = pd.read_csv('../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q1.csv', usecols = [0, 5, 7])
divvy_2018_2 = pd.read_csv('../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q2.csv', usecols = [0, 5, 7])
divvy_2018_3 = pd.read_csv('../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q3.csv', usecols = [0, 5, 7])
divvy_2018_4 = pd.read_csv('../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q4.csv', usecols = [0, 5, 7])

#read in the list of divvy stations
stations = pd.read_csv('../input/chicagodivvybikestations/Divvy_Bicycle_Stations.csv', usecols = [0, 13])

#get all columns to int dtype
divvy_2018_1[['to_station_id', 'from_station_id']] = divvy_2018_1[['to_station_id', 'from_station_id']].astype(int)
divvy_2018_2[['to_station_id', 'from_station_id']] = divvy_2018_2[['to_station_id', 'from_station_id']].astype(int)
divvy_2018_3[['to_station_id', 'from_station_id']] = divvy_2018_3[['to_station_id', 'from_station_id']].astype(int)
divvy_2018_4[['to_station_id', 'from_station_id']] = divvy_2018_4[['to_station_id', 'from_station_id']].astype(int)

#list of the dataframes
divvy_dfs = [divvy_2018_1, divvy_2018_2, divvy_2018_3, divvy_2018_4]

#concatenate the quarterly ride dataframes into one
divvy_2018 = pd.concat(divvy_dfs, axis = 'rows', ignore_index = True)
#merge with stations df on from_station_id to get the ward number for the from station.
#rename the 'Wards' column to 'from_ward' and drop the ID column
divvy_2018_frommerge = pd.merge(divvy_2018, stations, how = 'left', left_on = 'from_station_id', right_on = 'ID')
divvy_2018_frommerge = divvy_2018_frommerge.rename(columns = {'Wards':'from_ward'})
divvy_2018_frommerge = divvy_2018_frommerge.drop('ID', axis = 1)

#merge with stations df on from_station_id to get the ward number for the from station
#rename the 'Wards' column to 'to_ward' and drop the ID column
divvy_2018_fullmerge = pd.merge(divvy_2018_frommerge, stations, how = 'left', left_on = 'to_station_id', right_on = 'ID')
divvy_2018_fullmerge = divvy_2018_fullmerge.rename(columns = {'Wards':'to_ward'})
divvy_2018_fullmerge = divvy_2018_fullmerge.drop('ID', axis = 1)
#drop both station id columns
divvy_2018_fullmerge = divvy_2018_fullmerge.drop(['from_station_id', 'to_station_id'], axis = 1)

#fill empty ward values with 0
divvy_2018_fullmerge = divvy_2018_fullmerge.fillna(0)

#convert ward number to int type
divvy_2018_fullmerge[['from_ward', 'to_ward']] = divvy_2018_fullmerge[['from_ward', 'to_ward']].astype(int)
#group df by from_ward and to_ward count and rename the trip_id col to num_trips
divvy_2018_bypair = divvy_2018_fullmerge.groupby(by = ['from_ward', 'to_ward']).count()
divvy_2018_bypair = divvy_2018_bypair.rename(columns = {'trip_id':'num_trips'})
#unstack divvy_2018_bypair and convert to a 39*39 matrix
divvy_2018_unstack = divvy_2018_bypair.unstack().fillna(0)
divvy_2018_unstack = divvy_2018_unstack.astype(int)
matrix = divvy_2018_unstack.values
matrix = matrix.tolist()
#create a list of the wards that have divvy stations in them
wards = divvy_2018_fullmerge['to_ward'].unique()
wards_list = wards.tolist()
wards_list.sort(reverse=False)
Chord(matrix, wards_list, wrap_labels=False, label_color="#4c40bf").show()
Chord(matrix, wards_list, wrap_labels=False, label_color="#4c40bf").to_html()