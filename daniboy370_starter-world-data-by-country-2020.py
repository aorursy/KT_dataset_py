import re, os, csv

import numpy as np

import pandas as pd

import plotly.express as px



PATH_root = '/kaggle/input/' # --> '/world-data-by-country-2020'

os.chdir(PATH_root) # os.listdir()
def show_data_on_map( df, arg_i, graphic_i ):

    '''

        Input  : clean dataframe per countries

        --------------------------------------

        Output : map visualization

    '''



    fig = px.choropleth(df, locations='ISO-code', color=arg_i, hover_name="Country",

                        color_continuous_scale=graphic_i, projection='natural earth')

    fig.update_layout(title={'text':arg_i+' per country', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'})

    fig.show()
# Initialize color pallete @ https://plotly.com/python/builtin-colorscales/

graphix = []

graphix.append( px.colors.sequential.Sunset ) 

graphix.append( px.colors.sequential.Bluered ) 

graphix.append( px.colors.sequential.Electric )

graphix.append( px.colors.sequential.Viridis )

graphix.append( px.colors.sequential.Agsunset )

graphix.append( px.colors.sequential.Rainbow )

graphix.append( px.colors.sequential.thermal )
arg_i = 'GDP per capita'

show_data_on_map( pd.read_csv( PATH_root+arg_i+'.csv' ), arg_i, graphix[5] )
arg_i = 'Population growth'

df = pd.read_csv( PATH_root+arg_i+'.csv' )

show_data_on_map( df[df.iloc[:,1].astype(float) < 5], arg_i, graphix[1] )
arg_i = 'Life expectancy'

show_data_on_map( pd.read_csv( PATH_root+arg_i+'.csv' ), arg_i, graphix[6] )
arg_i = 'Median age'

show_data_on_map( pd.read_csv( PATH_root+arg_i+'.csv' ), arg_i, graphix[3] )
arg_i = 'Meat consumption'

show_data_on_map( pd.read_csv( PATH_root+arg_i+'.csv' ), arg_i, graphix[1] )
arg_i = 'Sex-ratio'

df = pd.read_csv( PATH_root+arg_i+'.csv' )

show_data_on_map( df[ (df.iloc[:, 1]).astype(float) < 1.25 ], arg_i, graphix[2] )
arg_i = 'Suicide rate'

show_data_on_map( pd.read_csv( PATH_root+arg_i+'.csv' ), arg_i, graphix[0] )
arg_i = 'Urbanization rate'

show_data_on_map( pd.read_csv( PATH_root+arg_i+'.csv' ), arg_i, graphix[0] )
arg_i = 'Fertility'

show_data_on_map( pd.read_csv( PATH_root+arg_i+'.csv' ), arg_i, graphix[0] )