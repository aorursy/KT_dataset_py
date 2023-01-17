import pandas as pd

import plotly

!pip install chart_studio



import chart_studio as cs

from plotly.graph_objects import *

import chart_studio.plotly as py









#import plotly.graph_objects as go

# change kaggle settings to INTERNET CONNECTED 
# https://plot.ly/settings/api#/

# Replace username and api_key values with the ones from your plotly account 

cs.tools.set_credentials_file(username='minhvohandsome', api_key='OZoQdHZtvDZ8r0vtG3Wc')
df = pd.read_csv("../input/data-iphone.csv")

df.head()
df.info()
data = [Bar(x=df.Quarter,

            y=df.Sales)]



py.iplot(data, filename='basic_bar')
data = [Scatter(x=df.Quarter,

            y=df.Sales)]



py.iplot(data, filename='basic_line')
data = [Scatter(x=df.Quarter,

            y=df.YOY_Growth,

            mode = 'lines+markers')]



py.iplot(data, filename='basic_line_point')
trace_sales = Bar(x=df.Quarter,

                y=df.Sales)



trace_growth = Scatter(x=df.Quarter,

                y=df.YOY_Growth,

                mode = 'lines+markers')



data = [trace_sales, trace_growth]

fig = Figure(data=data)



py.iplot(fig, filename='styled_bar')