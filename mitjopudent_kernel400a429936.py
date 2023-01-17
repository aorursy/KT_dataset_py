import csv

import plotly

import pandas



ore = pandas.read_csv('../input/hirundo/bo.csv',sep=',',error_bad_lines=False,low_memory=False).query("year>1995")



import plotly.express as px

fig = px.scatter_geo(ore, lat="decimalLatitude", lon="decimalLongitude",

                     animation_frame="month", color="year", projection = "natural earth")

      



fig.show()