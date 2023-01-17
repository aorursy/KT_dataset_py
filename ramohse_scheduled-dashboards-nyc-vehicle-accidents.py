import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import seaborn as sns

# import plotly

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



%matplotlib inline

# Any results you write to the current directory are saved as output.
### Function to plot

def plot_borough_data(df_input, str_title):

    """

    Takes a dataframe plots it on a plotly chart.

    

    Parameters

    ----------

    df_input: pandas.DataFrame

        A pandas dataframe with dates and numerical data

    str_title: str

        Title for the plot



    Returns

    ----------

    plotly.plot

    

    """

    list_data = []

    

    ### loop through and create our list of data

    for c in df_input.columns:

        list_data.append(go.Scatter(x=df_data.index, 

                                    y=df_data[c],

                                    name=c.title()))



    ### specify the layout of our figure

    layout = dict(title = str_title,

                  xaxis= dict(title= 'Date',

                              ticklen= 5,

                              zeroline= False))



    ### create and show our figure

    fig = dict(data = list_data, layout = layout)

    return(iplot(fig))
### path to the file

path_colls = '../input/nypd-motor-vehicle-collisions.csv'



### list of columns to read in

list_cols = ["DATE", 

             "BOROUGH", 

             "NUMBER OF PERSONS INJURED", 

             "NUMBER OF PERSONS KILLED"]



### read in the collision dataset

df_colls = pd.read_csv(path_colls,

                       usecols = list_cols

                      )



### Convert date to a datetime before we start

df_colls["DATE"] = pd.to_datetime(df_colls["DATE"])
### Total injuries by month by borough

df_data = pd.pivot_table(data=df_colls, 

                         values="NUMBER OF PERSONS INJURED",

                         index=df_colls["DATE"].dt.strftime("%Y-%m"),

                         columns="BOROUGH",

                         aggfunc=np.sum

                        )



plot_borough_data(df_data, "Vehicular Injuries by Borough")
### Total deaths by month by borough

df_data = pd.pivot_table(data=df_colls, 

                         values="NUMBER OF PERSONS KILLED",

                         index=df_colls["DATE"].dt.strftime("%Y-%m"),

                         columns="BOROUGH",

                         aggfunc=np.sum

                        )



plot_borough_data(df_data, "Vehicular Deaths by Borough")