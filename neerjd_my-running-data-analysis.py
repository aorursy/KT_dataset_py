import pandas as pd



run_data = pd.read_csv('../input/runs.csv')

run_data.describe()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



pace_histogram = go.Histogram(

    x=run_data.pace,

    name = "Pace",

    opacity=0.75,

    marker=dict(color='rgba(171, 50, 96, 0.6)')

)



layout = go.Layout(barmode='overlay',

                   title='Pace Histogram',

                   xaxis={'title': 'miles / minute'},

                   yaxis={'title': 'count'}

)

figure = go.Figure(data=[pace_histogram], layout=layout)

iplot(figure)
distance_histogram = go.Histogram(

    x=run_data.distance,

    name = "Distance",

    opacity=0.75,

    marker=dict(color='rgba(171, 50, 96, 0.6)')

)



layout = go.Layout(barmode='overlay',

                   title='Distance Histogram',

                   xaxis={'title': 'miles'},

                   yaxis={'title': 'count'}

)

figure = go.Figure(data=[distance_histogram], layout=layout)

iplot(figure)