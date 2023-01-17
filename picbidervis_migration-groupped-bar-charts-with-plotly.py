import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
import plotly

plotly.__version__
from __future__ import division

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

import plotly.plotly as py

import plotly.graph_objs as go

import plotly.io as pio
migration = pd.read_excel("../input/migration.xls")

migration.head()
# Filling Nan values with forward fill

migration["cities"].fillna(method="ffill", inplace=True)

migration.head()
# Deleting last 2 characters of cities column.

migration["cities"] = migration["cities"].str[:-2]
migration.tail()
# We still have "-" in some rows.

migration["cities"] = migration["cities"].str.replace("-", "")
migration.head()
migration.shape
# I want to plot grouped bar charts for all cities and save all the charts as an image for a youtube video

# Creating a year list for plot, for x axe

x = migration.years.unique()



# 2008 - 2018 11 years of data for each city. 891\11=81 cities I need to iterate 81 times.

# Set variable for slicing

a = 0

for i in range(81):

    

    # defining y axes

    y1 = migration.loc[a:a+10, "migration"].values

    y2 = migration.loc[a:a+10, "emigration"].values

    y3 = migration.loc[a:a+10, "net_migration"].values

    

    trace1 = go.Bar(

        x=x,

        y=y1,

        text=y1,

        textposition = 'auto',

        marker=dict(

            color='rgb(129, 207, 224)',

            line=dict(

                color='rgb(8,48,107)',

                width=1.5),

            ),

        opacity=0.6, name="verilen göç / migration"

    )



    trace2 = go.Bar(

        x=x,

        y=y2,

        text=y2,

        textposition = 'auto',

        marker=dict(

            color='rgb(77, 5, 232)',

            line=dict(

                color='rgb(8,48,107)',

                width=1.5),

            ),

        opacity=0.6, name="alınan göç / emigration"

        )



    trace3 = go.Bar(

        x=x,

        y=y3,

        text=y3,

        textposition = 'auto',

        marker=dict(

            color='rgb(25, 181, 254)',

            line=dict(

                color='rgb(8,48,107)',

                width=1.5), 

            ),

        opacity=0.9, name="net göç / net migration"

        )



    data = [trace1,trace2, trace3]



    layout = go.Layout(

        title=go.layout.Title(

            text=migration.loc[a,"cities"],

            xref='paper',

            x=0,

            font=dict(

                    family='Courier New, monospace',

                    size=30,

                    color='#7f7f7f')

        ),

    xaxis=go.layout.XAxis(

        title=go.layout.xaxis.Title(

            text='yıl / years',

            font=dict(

                family='Courier New, monospace',

                size=18,

                color="rgb(0,0,0)"

            )

        )

    ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text='kişi sayısı / number of people',

            font=dict(

                family='Courier New, monospace',

                size=14,

                color='#7f7f7f'

                )

            )

        )

    )



    fig = go.Figure(data=data, layout=layout)

    iplot(fig)



    #pio.write_image(fig, './images/{}.png'.format(migration.loc[a,"cities"]), scale=10)

    

    a = a+11