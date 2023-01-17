# Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math

# List of Datafiles
print(os.listdir("../input"))
drivers = pd.read_csv('../input/drivers.csv', encoding='latin1')
laptimes = pd.read_csv('../input/lapTimes.csv')
driverId_and_Ref = drivers[['forename','surname','driverId']]
laptimes = laptimes.merge(driverId_and_Ref, on='driverId')

del driverId_and_Ref
laptimes['driverName'] = laptimes['forename'].str.cat(laptimes['surname'], sep=' ')
laptimes.drop(['forename', 'surname'], axis=1, inplace=True)

laptimes_per_driver = laptimes.drop(['driverId','lap','position','time'],axis=1,inplace=False)

races = pd.read_csv('../input/races.csv')
circuits = pd.read_csv('../input/circuits.csv', encoding='latin1')
raceId_circuitId = races[['raceId','circuitId']]

raceId_circuitId = races[['raceId','circuitId']]
laptimes_per_driver = laptimes_per_driver.merge(raceId_circuitId, on='raceId')
del raceId_circuitId


circuitId_name = circuits[['circuitId', 'name']]
laptimes_per_driver.merge(circuitId_name, on='circuitId')
del circuitId_name
circuitId_name = circuits[['circuitId', 'name']]
laptimes_per_driver = laptimes_per_driver.merge(circuitId_name, on='circuitId')
laptimes_per_driver.drop(['raceId', 'circuitId'], inplace=True, axis=1)
laptimes_per_driver.rename(columns={'name':'circuitName'}, inplace=True)
laptimes_per_driver = laptimes_per_driver.groupby(by=['driverName', 'circuitName']).min().reset_index()
uniqueCircuits = pd.DataFrame(laptimes_per_driver['circuitName'].unique(), columns=['circuit'])
listOfCircuits = uniqueCircuits['circuit'].tolist()
fastest = []
def driverTimesByCircuit(circuit):
    df = laptimes_per_driver[laptimes_per_driver['circuitName'] == circuit]
    df.sort_values('milliseconds', inplace=True)
    return df.iloc[0]

for circuit in listOfCircuits:
    fastest.append(driverTimesByCircuit(circuit))

fastest = pd.DataFrame(fastest)
#fastest['time'] = fastest['milliseconds'].apply(lambda x: str(math.floor(x/60000))+':'+str(math.floor(x%60000/1000)).zfill(2)+':'+str('%.3f' % ((x%1000)/1000)).replace('0.',''))
fastest['time'] = fastest['milliseconds'].apply(lambda x: str(math.floor(x/60000))+':'+str(math.floor(x%60000/1000)).zfill(2)+':'+str(x%1000).zfill(3))
fastest.drop(['milliseconds'], inplace=True, axis=1)
fastest.sort_values(by='time', inplace=True)
fastest.reset_index(drop=True, inplace=True)
fastest
#TODO add circuit length and divide to get average lap speed
most_fast_laps = fastest['driverName'].value_counts().reset_index()
most_fast_laps.rename(columns={'index':'Driver','driverName': '# Fastest Laps Held'}, inplace=True)
most_fast_laps = most_fast_laps.iloc[:10]

fig,ax = plt.subplots(figsize=(10,5))
g = sns.barplot(data=most_fast_laps, y='Driver', x='# Fastest Laps Held', palette='viridis')

