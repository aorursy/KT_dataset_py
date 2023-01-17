import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.optimize import curve_fit



from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go



import datetime

# the example is taken from plotly example
data = pd.read_csv('../input/covid19-in-usa/us_covid19_daily.csv')





dates = data['date'].values

x = dates - dates[0]

date_str = [str(d)[:4]+'-'+str(d)[4:6]+'-'+str(d)[-2:] for d in dates]

y = data['positive'].values



assert dates.shape[-1] == y.shape[-1]



def exp_func(x, a, b, c):

    return a*np.exp(b*x)+c





popt, pcov = curve_fit(exp_func, x, y, p0=(1, 1e-6, 1))



xx = np.linspace(x[0], x[-1]+5, 200)

yy = exp_func(xx, *popt)



trace1 = go.Scatter(

                  x=x,

                  y=y,

                  mode='markers',

                  marker=go.Marker(color='rgb(255, 127, 14)',

                                  size=7),

                  name='Data'

                  )



trace2 = go.Scatter(

                  x=xx,

                  y=yy,

                  mode='lines',

                  marker=go.Marker(color='rgb(31, 119, 180)'),

                  name='Exponential growth'

                  )



layout = go.Layout(

                title='COVID-19 in USA exponential fitting',

                plot_bgcolor='rgb(229, 229, 229)',

                xaxis= dict(title= f'Days after {date_str[0]}', ticklen= 10, zeroline= False),

                yaxis= dict(title= 'Positive cases',ticklen= 5,zeroline= False),

                )



plot_data = [trace1, trace2]

fig = go.Figure(data=plot_data, layout=layout)







fig.show()