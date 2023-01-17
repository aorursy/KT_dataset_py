import pandas as pd

import numpy as np

import matplotlib as mpl 

import matplotlib.pyplot as plt

import seaborn as sns 

import geopandas as gpd

import os

from ipywidgets import widgets, interactive



#print(os.listdir("../input/juvenile-crime-df"))

fp = "../input/india-map/Indian_States.shp"

map_df = gpd.read_file(fp)

map_df['st_nm'][23]='Delhi'

map_df['st_nm'][29] ='Andhra Pradesh'



df = pd.read_csv("../input/juvenile-crime-df/transposed_df.csv")

# Create text box and slider 



start_year = widgets.IntSlider(

    value=df.year.min(),

    min=df.year.min(),

    max=df.year.max(),

    step=1,

    description='Year:',

    disabled=False,

    continuous_update=True,

    orientation='horizontal',

    readout=True,

    readout_format='d'

)





# Make a dropdown to select the Area, or "All"

crime = widgets.Dropdown(

    options=list(df.columns[1:-1]),

    value='Murder',

    description='Crime:',

    disabled=False

)





def plotit(crime, start_year):

    """

    Filters and plot the dataframe for reqd year and crime type

    Args:

    -----

        * Crime (str): Crime type



        * start_year: the start



        Note: the dataframe to plot is globally defined here as `df`



    """

    df2 = df.copy()

    df2 = df2[['state_nm','year',crime]]    

    merged = map_df.set_index('st_nm').join(df2.set_index('state_nm'))

    merged['year'] = pd.to_numeric(merged['year'])

    merged = merged[~merged['year'].isna()]

    merged = merged[merged['year']==start_year]

    fig, ax = plt.subplots(1, figsize=(25, 15))

    ax.axis('off')

    ax.set_title(crime, fontdict={'fontsize': '25', 'fontweight' : '3'})

    merged.plot(column=crime, cmap='RdPu', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    
interactive(plotit, crime=crime, start_year=start_year)