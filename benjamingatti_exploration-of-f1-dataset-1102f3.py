# Imports
#from datetime import datetime
import datetime as dt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math
#import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

#plotly.__version__

# List of Datafiles
print(os.listdir("../input"))
pitstops = pd.read_csv('../input/pitStops.csv')
results = pd.read_csv('../input/results.csv')
races = pd.read_csv('../input/races.csv')
circuits = pd.read_csv('../input/circuits.csv', encoding='latin1')
drivers = pd.read_csv('../input/drivers.csv', encoding='latin1')

#identify yellow flag
laptimes = pd.read_csv('../input/lapTimes.csv')
laptimes.head()

# Time Behind Leader
laptimes.sort_values(by = ['raceId', 'driverId', 'lap'], inplace=True)

laptimes.head()
#calculating the "totalmilli" and creating apropriate column for it in the df
laptimes['totalmilli'] = laptimes.groupby(['raceId', 'driverId'])['milliseconds'].transform(pd.Series.cumsum)

# Creating the copies to mearge:
laptimes_2 = laptimes[['raceId', 'lap', 'position', 'totalmilli']].copy()
laptimes_3 = laptimes[['raceId', 'lap', 'position', 'totalmilli']].copy()

# Adding and subtractin "1" to each position, so than we can merge the "correct" position with the one in front of it:
laptimes_2['position'] = laptimes_2['position'] + 1
laptimes_2.rename(columns={'position': "position_plus_1", 'totalmilli' : 'totalmilli_plus_1'}, inplace=True)

laptimes_3['position'] = laptimes_3['position'] -1
laptimes_3.rename(columns={'position': "position_min_1", 'totalmilli' : 'totalmilli_min_1'}, inplace=True)

# Mearging two dataframes:
merged = pd.merge(laptimes, laptimes_2, how = 'left', left_on=['raceId', 'lap', 'position'],
                  right_on=['raceId', 'lap', 'position_plus_1'])

# Mearging two dataframes:
merged = pd.merge(merged, laptimes_3, how = 'left', left_on=['raceId', 'lap', 'position'],
                  right_on=['raceId', 'lap', 'position_min_1'])

# Calculating how far each car behind/in front:
merged['to_in_front'] = merged['totalmilli'] - merged['totalmilli_plus_1']
merged['to_behind'] = merged['totalmilli_min_1'] - merged['totalmilli']
#Checking Results of Time Between
# 'to_previous' has to be >= 0:
print("positive:", merged[merged['to_in_front']>0].shape)
print("equal zero:", merged[merged['to_in_front']==0].shape)
print('less than zero', merged[merged['to_in_front']<0].shape)
# Now we can delete 'position_plus_1' and 'totalmilli_plus_1' columns if needed.
# merged.drop(['position_plus_1', 'totalmilli_plus_1', 'position_min_1', 'totalmilli_min_1'], axis=1, inplace = True)
# Puting merged df into laptimes
laptimes = merged.copy()
laptimes.head()
leaders = results.loc[results['positionOrder']<10][['raceId','driverId']]
leaders.head(100)

winLaps = laptimes.merge(leaders, on=['raceId','driverId'])

avgLaps = winLaps.groupby(['raceId', 'lap'])['milliseconds'].min().reset_index()
avgLaps.head()

BestLapSpeed = avgLaps.groupby(['raceId'])['milliseconds'].min().reset_index()
BestLapSpeed.head()

#WorstLapSpeed = avgLaps.groupby(['raceId'])['milliseconds'].max().reset_index()
#WorstLapSpeed = WorstLapSpeed.rename(index=str, columns={"milliseconds": "slow"})
#WorstLapSpeed.head()

#BestLapSpeed = BestLapSpeed.merge(WorstLapSpeed, on='raceId')
#BestLapSpeed['yellowThreshold'] = (BestLapSpeed['milliseconds'] + BestLapSpeed['slow'] )/2 
BestLapSpeed['yellowThreshold'] = (BestLapSpeed['milliseconds'] * 1.1) #+ BestLapSpeed['slow'] )/2 
BestLapSpeed.head()

avgLaps = avgLaps.merge(BestLapSpeed[['raceId','yellowThreshold']], on='raceId')
avgLaps['flag'] = (avgLaps['yellowThreshold'] >  avgLaps['milliseconds']).astype(int)

avgLaps = avgLaps.rename(index=str, columns={"milliseconds": "avgLap"})

avgLaps.head()

sns.set(style="whitegrid")
#sns.set(rc={'figure.figsize':(6,7)})
g = sns.FacetGrid(avgLaps.head(1000), col="raceId", aspect=1 ,col_wrap=5, height=5, hue='flag')
g = g.map(sns.scatterplot, "lap", "avgLap", s=100)

#g = g.map(sns.scatterplot, "year", "lap", s=150)
#g = g.map(sns.violinplot, 'year','lap',hue="positionText" )
#ax = sns.violinplot(x="year", y="lap",  data=main, height = 50)

#sns.scatterplot(data=winners.loc[winners['raceId']==21].loc[winners['pit']==1],  x="lap", y="milliseconds_y", s=550,hue = 'pit')
#sns.scatterplot(data=winners.loc[winners['raceId']==960].loc[winners['pit']==1],  x="lap", y="milliseconds_y", s=100,hue = 'pit')
#sns.scatterplot(data=winners.loc[winners['raceId']==962],  x="lap", y="milliseconds_y", s=100,hue = 'pit')




laptimes2 = laptimes.merge(avgLaps, on=['raceId','lap']).reset_index()
laptimes2.head()


#winners['fastmillis'] = pd.to_datetime(winners['fastestLapTime'])
#winners.head()

#avgLaps = avgLaps.merge(winners[['raceId','fastestLapTime']], on=['raceId'])
#avgLaps.head()
winners = results.loc[results['positionOrder']<10]
winners.head()

winners = winners.merge(laptimes2, on=['raceId','driverId'])
winners.head()

winners = winners.merge(pitstops,how='outer', on =['raceId','driverId','lap'])
winners.head()
winners.loc[winners['duration'].notnull()].head()

winners['pit'] = winners['duration'].notnull().astype(int) 
winners['milliseconds'].fillna(0, inplace=True) 

winners['lap_millis'] = winners['milliseconds_y'] -  winners['milliseconds']


winners = winners.loc[winners['raceId']==952] #950
winners.loc[winners['pit']==1].head()
winners.loc[winners['flag']==0].head()




leader = winners.loc[winners['position_y']==1]
leader.head()
leader = leader.rename(index=str, columns={"totalmilli": "lead_milli"})
winners = winners.merge(leader[['lap','lead_milli']], on='lap')
winners[['totalmilli','lead_milli']].head()

winners['behind'] = (winners['totalmilli'] -winners['lead_milli'])  #.astype(dt.timedelta)
winners[['behind','totalmilli','lead_milli']].head()
winners.head()

# Let's build our plot
import matplotlib.pyplot as plt
%matplotlib inline
# needed for jupyter notebooks
    
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax3 = fig.add_subplot(212)

#fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis

car = winners.loc[winners['positionOrder']==1]
color = 'blue'

ret = ax1.bar(data=car.loc[car['pit']==1], x="lap", height="milliseconds", width=1, alpha=0.2, color=color) #plot the Revenue on axis #1
# the next few lines plot the fiscal year data as bar plots and changes the color for each.
#ax2.bar(data=car,  x="lap", height="milliseconds_y",width=2, alpha=0.2, color='blue')
ax2.scatter(data=car, x='lap', y = 'lap_millis',color=color, alpha=0.2)
ax3.scatter(x='lap', y = 'behind', data=car,color=color, alpha=0.2)
ax3.scatter(x='lap', y = 'to_in_front',s=2, data=car, color=color, alpha=0.3)
ax3.scatter(x='lap', y = 'to_behind',s=2, data=car, color=color, alpha=0.3)

car = winners.loc[winners['positionOrder']==2]
color = 'red'

ret = ax1.bar(data=car.loc[car['pit']==1], x="lap", height="milliseconds", width=1, alpha=0.2, color=color) #plot the Revenue on axis #1
# the next few lines plot the fiscal year data as bar plots and changes the color for each.
#ax2.bar(data=car,  x="lap", height="milliseconds_y",width=2, alpha=0.2, color='blue')
ax2.scatter(data=car, x='lap', y = 'lap_millis',color=color, alpha=0.2)
ax3.scatter(x='lap', y = 'behind', data=car,color=color, alpha=0.2)
ax3.scatter(x='lap', y = 'to_in_front',s=2, data=car, color=color, alpha=0.3)
ax3.scatter(x='lap', y = 'to_behind',s=2, data=car, color=color, alpha=0.3)


ret = ax2.bar(data=car.loc[car['flag']==0], x="lap", height='milliseconds_y', width=1, alpha=0.4, color='yellow') #plot the Revenue on axis #1


ax2.grid(b=False) # turn off grid #2
ax1.set_ylim(8000,35000)
ax2.set_ylim(75000,160000)

ax1.set_title('Title')
ax1.set_ylabel('ylabel')
ax2.set_ylabel('y2Label')
 
#ax3.plot(data=car, x='lap', y = 'position_y',color='red', alpha=0.2)
    
# Set the x-axis labels to be more meaningful than just some random dates.
#labels = ['FY 2010', 'FY 2011','FY 2012', 'FY 2013','FY 2014', 'FY 2015']
#ax1.axes.set_xticklabels(labels)

winners['hue'] = winners['position_y'] #/4 
import pylab as plt


plt.scatter(data=winners, x='lap', y = 'milliseconds_y')
#plt.scatter(X,Y2,color='g')
plt.show()

sns.set(style="whitegrid")
#sns.set(rc={'figure.figsize':(6,7)})
#g = sns.FacetGrid(winners.head(1000), col="raceId", aspect=1 ,col_wrap=5, height=5, hue='pit')
#g = g.map(sns.scatterplot, "lap", "milliseconds_y", s=100)

#g = g.map(sns.scatterplot, "year", "lap", s=150)
#g = g.map(sns.violinplot, 'year','lap',hue="positionText" )
#ax = sns.violinplot(x="year", y="lap",  data=main, height = 50)

#sns.scatterplot(data=winners.loc[winners['raceId']==21].loc[winners['pit']==1],  x="lap", y="milliseconds_y", s=550,hue = 'pit')
#sns.scatterplot(data=winners.loc[winners['raceId']==960].loc[winners['pit']==1],  x="lap", y="milliseconds_y", s=100,hue = 'pit')
ret = sns.scatterplot(data=winners.loc[winners['raceId']==964],  x="lap", y="milliseconds_y",style='position_x', s=100,hue = 'hue')

#sns.scatterplot(data=winners.loc[winners['pit']==1],  x="raceId", y="lap", s=100,hue = 'pit')


#palette="Set2"

#sns.pairplot(main, hue="position")

main = results.copy()
race_circuit = races[['raceId','circuitId','year']]
main = main.merge(race_circuit, on='raceId')
main = main.merge(pitstops, on=['raceId','driverId'])

main = main.merge(avgLaps, on=['raceId','lap']).reset_index()
main.head()

#main = main.merge(laptimes2, on=['raceId','driverId'])
#winners.head()

main = main.loc[main['positionOrder'] < 8]
main = main.loc[main['year'] > 2011]
main = main.loc[main['circuitId'] == 4]
main = main.loc[main['milliseconds_y'] < 60000 * 10]
#main.head()
main['x'] = main['year'] + (main['position']/10)
#main['pitflag'] = main['pit'] + (main['flag']*2)

main = main[['positionText','position','circuitId','lap','year','milliseconds_y','x','flag']]
#main = main.loc[main['positionText'] > 0]
main = main[main.positionText.apply(lambda x: x.isnumeric())]
main.head()


#sns.set(style="ticks")
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(14,6)})
#g = sns.FacetGrid(main, col="year", aspect=0.3 ,col_wrap=5, height=5, hue="positionText")
#g = g.map(sns.scatterplot,  "positionText", "lap", s=150, size='milliseconds_y')

#g = g.map(sns.scatterplot, "year", "lap", s=150)
#g = g.map(sns.violinplot, 'year','lap',hue="positionText" )
#ax = sns.violinplot(x="year", y="lap",  data=main, height = 50)

plt = sns.scatterplot(data=main,  x="x", y="lap", s=550,palette="Set2",hue = 'flag', size = 'milliseconds_y')

#sns.pairplot(main, hue="position")
