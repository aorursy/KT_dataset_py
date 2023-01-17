import numpy as np
import pandas as pd

#keep don't use scientific notation for large floats
#and limit decimal places for now.
pd.options.display.float_format = '{:.2f}'.format
#from kaggle presentation setup offline plotly graphs.
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
# filename: data set to load from kernel
filename = "../input/turnstile-usage-data-2018.csv"
# readlimit: Number of rows to load from csv -- to load all, set to None.
readlimit = None
data = pd.read_csv(filename, parse_dates=['Date'], nrows=readlimit)
#Fix columns with trailing spaces
data.columns = data.columns.str.strip()
#Fix Columns with spaces in column name.
data.columns = [col.lower().replace(' ', '_') for col in data.columns]

#narrow columns to only the ones that we're interested in.
# ['station', 'date', 'endtries', 'exits']
data = data.filter(['station', 'date', 'entries', 'exits'], axis=1)

#unique turnstile data by station by date.  
#Problem is there are multiple reports per day for same station
data = data.set_index(['station','date'])
##group by station
##for each station, set index to date
##resample the date by 1D
##sum the data columns by day
##
stationDaySum=data.groupby(by='station').apply(lambda x: x.resample(rule='1D',level='date').sum())
# Function to calculate several metrics on the dataset
# Now that it's grouped by station, date
def calculateTotalsByRow(row):
    d = row.entries - row.exits
    return(row.entries + row.exits, d, abs(d))

## row function returns tuple of values that need to be spread out to each new column
def applyTotals(df):
    (df['flowtotal'], df['flowdelta'],df['flowabs']) = zip(*df.apply(calculateTotalsByRow, axis=1))

applyTotals(stationDaySum)
stationList = stationDaySum.groupby('station')['flowtotal'].mean().sort_values(ascending=False)
data = [go.Bar(x=stationList.index,y=stationList.values)]
layout = go.Layout(title='Total Volume By Station')

fig = go.Figure(data=data, layout=layout)
iplot(fig)

## Report the top n stations
nstations = 6
topnStations = stationList.sort_values(ascending=False).nlargest(nstations).index
print("\n".join(topnStations))
dsrank = stationDaySum[stationDaySum.index.get_level_values('station').isin(topnStations)].groupby(by=['date','station']).mean()
#i'm thinking that indexes suck.
dsrank = dsrank.reset_index()
data = [go.Scatter(name=station, x=dm.date,y=dm.flowtotal)for (station,dm) in dsrank.groupby('station') ]
layout = go.Layout(title='Top {} Station Total Flow By Day'.format(nstations))

fig = go.Figure(data=data, layout=layout)
iplot(fig)



for (station,dm) in dsrank.groupby('station'):
        data = [go.Scatter(name='entries', x=dm.date,y=dm.entries),
                go.Scatter(name='exits', x=dm.date,y=dm.exits)]
        layout = go.Layout(title='Station {} Entries and Exits By Day'.format(station))
        fig = go.Figure(data=data, layout=layout) 
        iplot(fig)

