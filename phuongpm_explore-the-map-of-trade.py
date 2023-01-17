# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go

import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/country_partner_hsproduct4digit_year_2016.csv')
df.info()
df.head()
# Get total export and import value of each location
df_total_export = df.groupby([df.location_code, df.location_name_short_en])['export_value'].sum().reset_index(name = 'total_export')
df_total_export.info()
df_total_import = df.groupby([df.location_code, df.location_name_short_en])['import_value'].sum().reset_index(name = 'total_import')
def choroplethMap(df_to_draw, value_to_draw, cb_title, plot_title):
    data = [ dict(
            type = 'choropleth',
            locations = df_to_draw['location_code'],
            z = df_to_draw[value_to_draw],
            colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
                [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
            autocolorscale = False,
            reversescale = True,
            marker = dict(
                line = dict (
                    color = 'rgb(180,180,180)',
                    width = 0.5
                ) ),
            colorbar = dict(
                autotick = False,
                tickprefix = '$',
                title = cb_title),
          ) ]

    layout = dict(
        title = plot_title,
        geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(
                type = 'Mercator'
            )
        )
    )

    fig = dict( data=data, layout=layout )
    py.iplot( fig, validate=False, filename='d3-world-map' )
df_trade_us = df[df['location_code'] == 'USA']
choroplethMap(df_trade_us, 'export_value', 'Export value', 'Export value between US and other countries')
