# core libraries

import sqlite3

import pandas as pd

import numpy as np



# plotting libraries

from plotly.offline import init_notebook_mode, iplot

from plotly.graph_objs import Scatter, Figure, Layout

init_notebook_mode()



# get the data...

con = sqlite3.connect('../input/database.sqlite')

torrents = pd.read_sql_query('SELECT * from torrents;', con)

con.close()



# define mixtape and album subset

mixtapes = torrents.loc[torrents.releaseType == 'mixtape']

albums = torrents.loc[torrents.releaseType == 'album']
# define year range

years = np.arange(1991, 2017)



# get average and standard error of log snatches per release 

snatches = pd.DataFrame(0, index = years, columns = ['Mixtapes','Albums'])

stderror = pd.DataFrame(0, index = years, columns = ['Mixtapes','Albums'])



# compute data for each year

for i in years:

    

    # index releases from current year

    year_mixtapes = mixtapes.loc[mixtapes.groupYear == i]

    year_albums = albums.loc[albums.groupYear == i]

    

    # take log transform -- add one to prevent log(0) error

    year_mixtapes = np.log(year_mixtapes.totalSnatched + 1)

    year_albums = np.log(year_albums.totalSnatched + 1)

    

    # get average snatches per release

    snatches.loc[i,'Mixtapes'] = year_mixtapes.mean() 

    snatches.loc[i,'Albums'] = year_albums.mean() 

    

    # get standard error

    stderror.loc[i,'Mixtapes'] = year_mixtapes.std() / np.sqrt(len(year_mixtapes))

    stderror.loc[i,'Albums'] = year_albums.std() / np.sqrt(len(year_albums))    
linespecs = {

    'Mixtapes':  dict(color = 'blue', width = 2),

    'Albums':  dict(color = 'red', width = 2),

    }





handles = []

for k in linespecs.keys():

    handles.append( Scatter(

            x = years, 

            y = snatches[k],

            name = k, 

            hoverinfo = 'x+name',

            line = linespecs[k],

            error_y = dict(type='data', 

                           array = stderror[k], 

                           color = linespecs[k]['color'])

               )

        )



    

layout = Layout(

    xaxis = dict(

                tickmode = 'auto',

                nticks = 20, 

                tickangle = -60, 

                showgrid = False

            ),

    yaxis = dict(title = 'Log Snatches Per Release'),

    hovermode = 'closest',

    legend = dict(x = 0.55, y = 0.15),

)





fh = Figure(data=handles, layout=layout)

iplot(fh)
# aggregate over mixtapes and albums

releases = torrents.loc[torrents.releaseType.isin(['mixtape', 'album'])]

data = pd.DataFrame(index = years, columns = ['Snatches', 'Releases'])



# compute data for each year

for i in years:

    year_releases = releases.loc[releases.groupYear == i]

    data.loc[i,'Snatches'] = np.sum(np.log(year_releases.totalSnatched + 1))

    data.loc[i,'Releases'] = year_releases.shape[0] 



    

# plot as scatter

labels = ["'" + str(i)[2:] for i in years]

sh = Scatter(

    x = data.Releases, y = data.Snatches,

    mode = 'text', text = labels,

    textposition='center',

    hoverinfo = 'none',

    textfont = dict( family='monospace', size=14, color='red'),

    name = None

)



# a quick reference line

slope = 4.1

lh = Scatter(

    x = [min(data.Releases), max(data.Releases)], 

    y = [slope*min(data.Releases), slope*max(data.Releases)],

    mode = 'lines', line = dict(color = 'gray', width = 1),

    name = '2009 Extrapolation'

)



    

layout = Layout(

    yaxis = dict(title = 'Total Log Snatches'),

    xaxis = dict(title = 'Number of Releases'),

    hovermode = 'closest',

    showlegend=False,

    annotations= [dict(

        x = 4000, y = 4000 * slope,

        text = 'log(Snatches) = 4.1 * Releases',

        font = dict(family = 'serif', size = 14),

        showarrow = False, bgcolor = 'white'

        )]

)



fh = Figure(data=[lh, sh], layout=layout)

iplot(fh)