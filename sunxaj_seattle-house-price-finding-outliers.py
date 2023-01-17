# Import libraries and read data
import pandas as pd
import numpy as np
import re
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
# Lead data
data = pd.read_csv('../input/dataset4_with_outliers.csv')
data.head()
# Create a scatter plot
trace = go.Scatter(
    x = data.price,
    y = data.sqft_lot,
    mode = 'markers'
)

data1 = [trace]

# Plot and embed in ipython notebook!
#iplot(data1, filename='basic-scatter')
iplot(data1, filename='basic-scatter')
# Plot the values for selected columns
boxplot = data.iloc[:,1:]
boxplot.plot(kind='box', figsize=(20,20))
# Plot other columns with similar value range
boxplot = data[['waterfront','view','condition','sqft_basement','grade','bedrooms',
                'floors','sqft_living','sqft_above','bathrooms']]
boxplot.plot(kind='box', figsize=(20,20))
# Check if there's any outliers in these columns
print(data.yr_built.unique())
print(data.yr_renovated.unique())
print(data.date.unique())
# Plot price
bp = data.boxplot(column='price', figsize=(10,6), grid=True, return_type='dict')
print(bp.keys())
print("fliers =", bp['fliers'][0].get_ydata()[0])
# Plot price by living size
plt.figure(figsize=(10, 6))
plt.scatter(data.sqft_living, data.price, s=4)
plt.xlabel("Square footage of the living space")
plt.ylabel("Price")
# Create a 3D plot

x = data.lat
y = data.long
z = data.price/data.sqft_living

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=data.bedrooms*2,   # Set size according to number of beadrooms
        color=z,                # set color to values
        colorscale='Portland',   
        opacity=1
    )
)

data1 = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data1, layout=layout)
iplot(fig, filename='3d-scatter-colorscale')
# List the house price that are obnormal
data[data.price > 800*data.sqft_living]
# Remove outliers
data.drop(data[data.price > 800*data.sqft_living].index,axis = 0,inplace = True)
# Create a plot to show the bedroom and bathroom ratio
x = data.bathrooms

trace0 = go.Box(
    y=data.bedrooms,
    x=x,
    name = "Whiskers and Outliers",
    boxpoints = 'outliers',
    marker = dict(
        color = 'Orange'),
    line = dict(
        color = 'rgb(107,174,214)')
)

data0 = [trace0]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000,
    yaxis=dict(
        title='Number of bedrooms',
        zeroline=False
    ),
    boxmode='group'
)
fig = go.Figure(data=data0, layout=layout, )
iplot(fig)
# List the outliers
data[data.bedrooms > 9]
# Remove outliers and save dataframe
data.drop(data[data.bedrooms > 9].index, axis = 0,inplace = True)
data.to_csv('dataset4_solution.csv')
