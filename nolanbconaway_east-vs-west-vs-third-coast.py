import sqlite3

import pandas as pd

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

from plotly.graph_objs import Bar, Scatter, Figure, Layout

init_notebook_mode()



# get the data...

con = sqlite3.connect('../input/database.sqlite')

tags = pd.read_sql_query('SELECT * from tags;', con)

torrents = pd.read_sql_query('SELECT * from torrents;', con)

con.close()
# A list of tags associated with each region. These were handpicked but I think they are fair!

coast_tags = {

    'East Coast': ['new.york', 'east.coast','east.coast.rap'],

    'West Coast': ['bay.area', 'los.angeles', 'west.coast', 'california'],

    'Dirty South': ['dirty.south', 'southern', 'southern.rap','southern.hip.hop', 'new.orleans', 'houston', 'memphis', 'atlanta'],

    }



# Count number of torrents in each tag group

yearly_counts = pd.DataFrame(data = None, columns = ['Year'] + list(coast_tags.keys()))

for year in range(1985, 2017):

    ids = torrents.id.loc[torrents.groupYear==year]

    yeartags = tags.loc[tags.id.isin(ids)]

    

    # create row for dataframe

    row = dict(Year = year)

    for k,v in coast_tags.items():

        releases = yeartags.loc[yeartags.tag.isin(v), 'id']

        row[k] = pd.unique(releases).shape[0]



    # add row

    yearly_counts = yearly_counts.append(row, ignore_index = True)

    
linespecs = {

    'East Coast':  dict(color = 'blue', width = 2),

    'West Coast':  dict(color = 'orange', width = 2),

    'Dirty South': dict(color = 'red', width = 2),

    }



handles = []

for k in coast_tags.keys():

    handles.append(

        Scatter(x = yearly_counts.Year, 

                y = yearly_counts[k], 

                name = k, 

                hoverinfo = 'x+name',

                line = linespecs[k]

               )

        )

    

layout = Layout(

    xaxis = dict(tickmode = 'auto', 

                 nticks = 20, 

                 tickangle = -60, 

                 showgrid = False),

    yaxis = dict(title = 'What.CD Tag Frequency'),

    hovermode = 'closest',

    legend = dict(x = 0.05, y = 0.9),

)



fh = Figure(data=handles, layout=layout)

iplot(fh)