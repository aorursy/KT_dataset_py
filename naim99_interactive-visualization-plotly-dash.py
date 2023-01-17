

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

# create fake data:

df = pd.DataFrame(np.random.randn(50,3),columns='Col1 Col2 Col3'.split())

df.plot()

plt.show()

import plotly.graph_objs as go

import numpy as np





np.random.seed(42)

random_x = np.random.randint(1,101,100)

random_y = np.random.randint(1,101,100)



fig = go.Figure(data=go.Scatter(x=random_x, y=random_y, mode='markers'))



fig.show()
matplotlib.pyplot.savefig('scatter_plot.png')
import plotly.graph_objects as go



# Create random data with numpy

import numpy as np

np.random.seed(1)



N = 100

random_x = np.linspace(0, 1, N)

random_y0 = np.random.randn(N) + 5

random_y1 = np.random.randn(N)

random_y2 = np.random.randn(N) - 5



fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=random_x, y=random_y0,

                    mode='markers',

                    name='markers'))

fig.add_trace(go.Scatter(x=random_x, y=random_y1,

                    mode='lines+markers',

                    name='lines+markers'))

fig.add_trace(go.Scatter(x=random_x, y=random_y2,

                    mode='lines',

                    name='lines'))



fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

import numpy as np



np.random.seed(42)

random_x = np.random.randint(1,101,100)

random_y = np.random.randint(1,101,100)



data = [go.Scatter(

    x = random_x,

    y = random_y,

    mode = 'markers',

)]

layout = go.Layout(

    title = 'Random Data Scatterplot', # Graph title

    xaxis = dict(title = 'Some random x-values'), # x-axis label

    yaxis = dict(title = 'Some random y-values'), # y-axis label

    hovermode ='closest' # handles multiple points landing on the same vertical

)

fig = go.Figure(data=data, layout=layout)

pyo.plot(fig, filename='scatter2.html')

fig.show() 
import plotly.offline as pyo

import plotly.graph_objs as go

import numpy as np



np.random.seed(42)

random_x = np.random.randint(1,101,100)

random_y = np.random.randint(1,101,100)



data = [go.Scatter(

    x = random_x,

    y = random_y,

    mode = 'markers',

    marker = dict(      # change the marker style

        size = 12,

        color = 'rgb(51,204,153)',

        symbol = 'pentagon',

        line = dict(

            width = 2,

        )

    )

)]

layout = go.Layout(

    title = 'Random Data Scatterplot', # Graph title

    xaxis = dict(title = 'Some random x-values'), # x-axis label

    yaxis = dict(title = 'Some random y-values'), # y-axis label

    hovermode ='closest' # handles multiple points landing on the same vertical

)

fig = go.Figure(data=data, layout=layout)

pyo.plot(fig, filename='scatter3.html')

fig.show() 
import plotly.offline as pyo

import plotly.graph_objs as go

import numpy as np



np.random.seed(56)

x_values = np.linspace(0, 1, 100) # 100 evenly spaced values

y_values = np.random.randn(100)   # 100 random values



# Create traces

trace0 = go.Scatter(

    x = x_values,

    y = y_values+5,

    mode = 'markers',

    name = 'markers'

)

trace1 = go.Scatter(

    x = x_values,

    y = y_values,

    mode = 'lines+markers',

    name = 'lines+markers'

)

trace2 = go.Scatter(

    x = x_values,

    y = y_values-5,

    mode = 'lines',

    name = 'lines'

)

data = [trace0, trace1, trace2]  # assign traces to data

layout = go.Layout(

    title = 'Line chart showing three different modes'

)

fig = go.Figure(data=data,layout=layout)

pyo.plot(fig, filename='line1.html')

fig.show() 
import plotly.offline as pyo

import plotly.graph_objs as go

import numpy as np



np.random.seed(56)

x_values = np.linspace(0, 1, 100) # 100 evenly spaced values

y_values = np.random.randn(100)   # 100 random values



# Create traces

trace0 = go.Scatter(

    x = x_values,

    y = y_values+5,

    mode = 'markers',

    name = 'markers'

)

trace1 = go.Scatter(

    x = x_values,

    y = y_values,

    mode = 'lines+markers',

    name = 'lines+markers'

)

trace2 = go.Scatter(

    x = x_values,

    y = y_values-5,

    mode = 'lines',

    name = 'lines'

)

data = [trace0, trace1, trace2]  # assign traces to data

layout = go.Layout(

    title = 'Line chart showing three different modes'

)

fig = go.Figure(data=data,layout=layout)

fig.show() 

pyo.plot(fig, filename='line1.html')

df = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")
df.columns

df.rename(columns={'Country (or dependency)': 'country'}, inplace=True)
df.head() 