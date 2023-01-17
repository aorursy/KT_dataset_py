#Create a dataframe that contains the GDP data and display using the method head() and take a screen shot.

import pandas as pd

from bokeh.plotting import figure, output_file, show
def make_dashboard(x, gdp_change, unemployment, title, file_name):

    output_file(file_name)

    p = figure(title=title, x_axis_label='year', y_axis_label='%')

    p.line(x.squeeze(), gdp_change.squeeze(), color="firebrick", line_width=4, legend="% GDP change")

    p.line(x.squeeze(), unemployment.squeeze(), line_width=4, legend="% unemployed")

    show(p)
links={'GDP':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_gdp.csv',\

       'unemployment':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_unemployment.csv'}
df_gdp = pd.read_csv(links["GDP"])

df_gdp.head()
df_unemp = pd.read_csv(links['unemployment'])

df_unemp.head()
df_unemp85 = df_unemp[df_unemp['unemployment'] > 8.5]

df_unemp85.head()
x = df_gdp['date']

x.head()
gdp_change = df_gdp['change-current'] # Create your dataframe with column change-current

gdp_change.head()
unemployment = df_unemp['unemployment'] # Create your dataframe with column unemployment

unemployment.head()
title = 'GDP and Unemployment Data'

file_name = "index.html"

make_dashboard(x=x, gdp_change=gdp_change, unemployment= unemployment, title=title, file_name=file_name)