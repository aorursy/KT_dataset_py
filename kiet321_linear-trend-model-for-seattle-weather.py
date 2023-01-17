import pandas as pd

import numpy as np

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

SW = pd.read_csv('../input/seattleWeather_1948-2017.csv')



SW.columns=SW.columns.str.lower()

separate=SW.date.str.split('-')

a,b,c=zip(*separate)

SW['year']=a

SW['month']=b

SW_year=SW['year'].unique().astype(int)

SW['avegareinday']=(SW.tmax+SW.tmin)/2

SW_avegareinmonth=(SW.groupby([(SW.year), (SW.month)])['avegareinday'].sum())/SW.groupby([(SW.year), (SW.month)])['avegareinday'].count()

SW_avegareinannualy=SW_avegareinmonth.groupby('year').sum()/12

SW_5yearmovingaverage=np.convolve(SW_avegareinannualy, np.ones((5,))/5, mode='valid')

from scipy.stats import linregress

linregress(SW_year, SW_avegareinannualy.values)
trace1 = dict(

    x=SW_year,

    y=SW_avegareinannualy,

  line= dict(

    color= "rgb(255, 127, 14)", 

    width= 1

  ), 



  mode= "lines+markers", 

  name= "annual average temp", 

  type= "scatter", 

  uid= "f5d9be"

)



trace2 = dict(

    x=SW_year,

    y=SW_5yearmovingaverage,

  line= dict(

    color= "rgb(51, 51, 255)", 

    width= 2

  ), 

  mode= "lines", 

  name= "5-year moving average temp", 

  type= "scatter", 

  uid= "f5d9be"

)



trace3 = dict(

    x=SW_year,

    y=0.04642*SW_year-40.055,

  line= dict(

    color= "rgb(0, 153, 0)", 

    width= 2

  ), 

  mode= "lines", 

  name= "long-term linear trend", 

  type= "scatter", 

  uid= "f5d9be"

)



layout = go.Layout(

    title= "Annualy avegare temperature in Seattle (1948-2017)", 

  xaxis=  {"title": "Years"},  

  yaxis=  {"title": "°C"}, 

annotations=[

        dict(

            x=2006,

            y=49,

            showarrow=False,

            text='y = 0.04642x-40.05572<br>R<sup>2</sup> =0.641239',

            font= {"size": 20}, 

        )

]

)

data=[trace1, trace2,trace3]

fig = dict(data=data, layout=layout)

iplot(fig)
SW_rain=np.asarray(SW.groupby('year')['rain'].sum())

SW_dry=np.asarray(SW.groupby('year')['rain'].count()) - np.asarray(SW.groupby('year')['rain'].sum()) 



labels = ['Dryness', 'Rain']

colors = ['rgb(255, 51, 0)', 'rgb(0, 51, 204)']

x_data = SW_year

y_data = [SW_rain, SW_dry]



traces = []

for i in range(0, 2):

    traces.append(go.Scatter(

        x = x_data,

        y = y_data[i],

        mode = 'splines',

        name = labels[i],

        line = dict(color = colors[i], width = 3)

    ))



layout = { 

  "title": "Rain and dry in Seattle (1948-2017)", 

  "xaxis": {"title": "Years"}, 

  "yaxis": {"title": "day"}

}



figure = dict(data = traces, layout = layout)

iplot(figure)