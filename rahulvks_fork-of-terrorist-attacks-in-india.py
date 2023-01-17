

import numpy as np

import pandas as pd



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly

plotly.offline.init_notebook_mode()
Meta_data = pd.read_csv('../input/globalterrorismdb_0616dist.csv',encoding='ISO-8859-1',

                        usecols=[0, 1, 2, 3, 8, 11, 13, 14, 29, 35, 84, 100, 103])
terror_data = Meta_data.rename(

    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',

             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',

             'weaptype1_txt':'weapon', 'attacktype1_txt':'attack',

             'nkill':'fatalities', 'nwound':'injuries'})
terror_IND = terror_data[(terror_data.country == 'India')]
len(terror_IND)
Countries_Terro_Count=Meta_data['country_txt'].value_counts()
Countries_Terro_Count.head(20)
Countries_Terro_Count.tail(20)
INDIA_Terror_Data = terror_data[terror_data['country'].str.contains("India")]
len(INDIA_Terror_Data)
INDIA_Terror_Data.columns
Ind_perstate_Count = pd.DataFrame({'State':INDIA_Terror_Data['state'].value_counts().

                               index, 'Attack Counts':INDIA_Terror_Data['state'].value_counts().values,

                              })
Ind_perstate_Count
Weapons
Weapons=INDIA_Terror_Data['weapon'].value_counts() 
# terrorist attacks by year

terror_peryear = np.asarray(INDIA_Terror_Data.groupby('year').year.count())



terror_years = np.arange(1990, 2016)



terror_years = np.delete(terror_years, [23])



trace = [go.Scatter(

         x = terror_years,

         y = terror_peryear,

         mode = 'lines+markers',

         name = 'Terror Counts',

         line = dict(

             color = 'Viridis',

             width = 3)

         )]



layout = go.Layout(

         title = 'Terrorist Attacks by Year in INDIA (1990-2016)',

         xaxis = dict(

             rangeslider = dict(thickness = 0.10),

             showline = True,

             showgrid = True

            

         ),

         yaxis = dict(

             range = [0.1, 425],

             showline = True,

             showgrid = True)

         )



figure = dict(data = trace, layout = layout)

iplot(figure)