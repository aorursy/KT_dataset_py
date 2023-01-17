import pandas as pd 

import urllib.request

from bs4 import BeautifulSoup

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from plotly.subplots import make_subplots

from plotly.subplots import make_subplots

Data = pd.read_csv('../input/the-office-imdb-ratings-per-episode/TheOfficeIMDBPerEpisode.csv')

fig = make_subplots(specs=[[{"secondary_y": True}]])



trace = go.Scatter(

                    x = Data.AirDate,

                    y = Data.Rating,

                    mode = "lines",

                    name = "Rating",

                    line=dict(shape = 'spline',color='rgb(242,219,126)', width=3))

fig.add_trace(trace)



trace = go.Bar(

                    x = Data.AirDate,

                    y = Data.Num_Votes,

                    name = "Votes",

                    marker_color='rgba(0, 0, 100,0.2)'

)

fig.add_trace(trace,secondary_y=True)



       



fig.update_layout(

    title = 'The Office IMDB Ratings Per Episode',

    xaxis_title="Air Date",

    annotations=[

            go.layout.Annotation(

                x='28 Apr. 2011',

                y=9.8,

                xref="x",

                yref="y",

                text="Steve Carell Leaves the Show",

                showarrow=True,

                arrowhead=7,

                ax=-40,

                ay=-40

            )

        ],

    xaxis=dict(

        showline=True,

        showgrid=True,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside'

    ),

    yaxis=dict(

        showgrid=True,

        zeroline=True,

        showline=True,

        showticklabels=True,

        gridcolor='rgba(0, 0, 0,0.09)'

    ),

   

    plot_bgcolor='white'

)

fig.update_xaxes(nticks=10)

fig.update_yaxes(title_text="IMDB Rating", secondary_y=False)

fig.update_yaxes(title_text="# Votes", secondary_y=True)



iplot(fig)