import numpy as np

import pandas as pd

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

unemployment_rate = pd.read_csv('../input/unemployment_rate.csv')

unemployment_rate.columns=unemployment_rate.columns.str.lower()

unemployment_rate['avegare'] = unemployment_rate.drop('year', axis=1).mean(axis=1)

SW_4yearmovingaverage=np.convolve(unemployment_rate['avegare'], np.ones((4,))/4, mode='valid')

from scipy.stats import linregress

linregress(unemployment_rate['year'], unemployment_rate['avegare'])
trace1 = dict(

    x=unemployment_rate['year'],

    y=unemployment_rate['avegare'],

  line= dict(

    color= "rgb(255, 127, 14)", 

    width= 1

  ), 



  mode= "lines+markers", 

  name= "annual average", 

  type= "scatter", 

  uid= "f5d9be"



)



trace2 = dict(

    x=unemployment_rate['year'],

    y=SW_4yearmovingaverage,

  line= dict(

    color= "rgb(255, 51, 204)", 

    width= 3

  ), 

  mode= "lines", 

  name= "4-year moving average", 

  type= "scatter", 

  uid= "f5d9be"

)



trace3 = dict(

    x=unemployment_rate['year'],

    y=0.0293*unemployment_rate['year']-52.73432,

  line= dict(

    color= "rgb(0, 153, 0)", 

    width= 3

  ), 

  mode= "lines", 

  name= "linear trend", 

  type= "scatter", 

  uid= "f5d9be"

)



trace4 = go.Scatter(

    x = [1989,1989],

    y = [0,9],

    mode = 'lines' ,

    name='George H. W. Bush',

    line = dict(color='rgb(204, 51, 0)', width = 4, shape = 'spline', smoothing = 1.3)



)



trace5 = go.Scatter(

    x = [1993,1993],

    y = [0,9],

    mode = 'lines' ,

    name='Bill Clinton',

    line = dict(color='rgb(51, 51, 255)', width = 4, shape = 'spline', smoothing = 1.3)



)



trace6 = go.Scatter(

    x = [2001,2001],

    y = [0,9],

    mode = 'lines' ,

    name='George W. Bush',

    line = dict(color='rgb(204, 51, 0)', width = 4, shape = 'spline', smoothing = 1.3)



)



trace7 = go.Scatter(

    x = [2009,2009],

    y = [0,9],

    mode = 'lines' ,

    name='Barack Obama',

    line = dict(color='rgb(51, 51, 255)', width = 4, shape = 'spline', smoothing = 1.3)



)



trace8 = go.Scatter(

    x = [2017,2017],

    y = [0,9],

    mode = 'lines' ,

    name='Donald Trump',

    line = dict(color='rgb(204, 51, 0)', width = 4, shape = 'spline', smoothing = 1.3)



)





layout = go.Layout(

    title= "Unemployment rate in United States (1989-2017)", 

  xaxis=  {"title": "Year"},  

  yaxis=  {"title": "Unemployment rate"}, 



)

data=[trace1, trace2,trace3, trace4, trace5, trace6, trace7, trace8]

fig=dict(data=data, layout=layout)

iplot(fig)
