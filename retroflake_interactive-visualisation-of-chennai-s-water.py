#Importing all the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
#First we will analyze the reservoir levels in Chennai

df = pd.read_csv("../input/chennai_reservoir_levels.csv")

df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')

df.head()
#To plot the data we'll use a scatter plot, here's a function to represent one using plotly library

def scatter_plot(reservoir, color,res):

    trace = go.Scatter(

        x=reservoir.index[::-1],

        y=reservoir.values[::-1],

        name=res,

        marker=dict(

            color=color,

        ),

    )



    return trace

#Creating trace objects for al 4 rservoirs

reservoir = df["POONDI"]

reservoir.index = df["Date"]

trace1 = scatter_plot(reservoir, 'red','POONDI')



reservoir = df["CHOLAVARAM"]

reservoir.index = df["Date"]

trace2 = scatter_plot(reservoir, 'blue','CHOLAVARAM')



reservoir = df["REDHILLS"]

reservoir.index = df["Date"]

trace3 = scatter_plot(reservoir, 'green','REDHILLS')



reservoir = df["CHEMBARAMBAKKAM"]

reservoir.index = df["Date"]

trace4 = scatter_plot(reservoir, 'purple','CHEMBARAMBAKKAM')

#Plotting the traces on a single plot

fig['layout'].update(height=600, width=1200, title='Water availability in all 4 reservoirs - in mcft')

py.iplot(fig, filename='reservoir_plots')
#From the above plot we see that water levels in the reservoirs is going down significantly since 2018
#For a better picture, let's combine the water levels and then analyze

df["total"] = df["POONDI"] + df["CHOLAVARAM"] + df["REDHILLS"] + df["CHEMBARAMBAKKAM"]

df["total"] = df["POONDI"] + df["CHOLAVARAM"] + df["REDHILLS"] + df["CHEMBARAMBAKKAM"]



reservoir = df["total"]

reservoir.index = df["Date"]

trace5 = scatter_plot(reservoir, 'violet','All')



fig = tools.make_subplots(rows=1, cols=1)

fig.append_trace(trace5, 1, 1)





fig['layout'].update(height=600, width=1200, title='Total water availability from all four reservoirs - in mcft')

py.iplot(fig, filename='reservoir_plots')
#Since the water decline has a steeper slope since 2018, let's isolate that duration 

#Selecting the data since January 2018 till June 2019(present data)

mask = (df['Date'] > '2018-1-1')

df=df.loc[mask]

df.head()
#Creating new traces for the succeeding data

reservoir = df["POONDI"]

reservoir.index = df["Date"]

trace1 = scatter_plot(reservoir, 'red','POONDI')



reservoir = df["CHOLAVARAM"]

reservoir.index = df["Date"]

trace2 = scatter_plot(reservoir, 'blue','CHOLAVARAM')



reservoir = df["REDHILLS"]

reservoir.index = df["Date"]

trace3 = scatter_plot(reservoir, 'green','REDHILLS')



reservoir = df["CHEMBARAMBAKKAM"]

reservoir.index = df["Date"]

trace4 = scatter_plot(reservoir, 'purple','CHEMBARAMBAKKAM')
fig = tools.make_subplots(rows=1, cols=1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 1)

fig.append_trace(trace4, 1, 1)
#Now, plotting the data

fig['layout'].update(height=600, width=1200,title='Declining Water Availability since 2018')

py.iplot(fig, filename='reservoir_plots')
#Finally let's look at combined water shortage 

df["total"] = df["POONDI"] + df["CHOLAVARAM"] + df["REDHILLS"] + df["CHEMBARAMBAKKAM"]

df["total"] = df["POONDI"] + df["CHOLAVARAM"] + df["REDHILLS"] + df["CHEMBARAMBAKKAM"]



reservoir = df["total"]

reservoir.index = df["Date"]

trace5 = scatter_plot(reservoir, 'violet','All')



fig = tools.make_subplots(rows=1, cols=1)

fig.append_trace(trace5, 1, 1)





fig['layout'].update(height=600, width=1200, title='Total water availability from all four reservoirs since January 2018 - in mcft')

py.iplot(fig, filename='reservoir_plots')
#Shocking to see the reservoir levels going down steeply since 2018, it's almmost zero right now

#It's worrisome to see that there is almost no water in reservirs in Chennai 
#Let's analyze the rain data

rain_df = pd.read_csv("../input/chennai_reservoir_rainfall.csv")

rain_df["Date"] = pd.to_datetime(rain_df["Date"], format='%d-%m-%Y')



rain_df["total"] = rain_df["POONDI"] + rain_df["CHOLAVARAM"] + rain_df["REDHILLS"] + rain_df["CHEMBARAMBAKKAM"]



def area_plot(reservoir):

    trace = go.Scatter(

        x=reservoir.index[::-1],

        y=reservoir.values[::-1],

        fill='tonexty'  

    )

    data=[trace]

    return trace



rain_df["YearMonth"] = pd.to_datetime(rain_df["Date"].dt.year.astype(str) + rain_df["Date"].dt.month.astype(str), format='%Y%m')

reservoir = rain_df.groupby("YearMonth")["total"].sum()

trace5 = area_plot(reservoir)



fig = tools.make_subplots(rows=1, cols=1)

fig.append_trace(trace5, 1, 1)





fig['layout'].update(height=600, width=1200, title='Total rainfall in all four reservoir regions - in mm')

py.iplot(fig, filename='rainfall_plots')

#As we can see that the major rainfall seasons are July - November 

#Also we can see that rainfall levels are lower in 2018 compared to others 
