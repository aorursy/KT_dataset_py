# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy
#this file from https://www.zillow.com/research/data/ but has to be saved as UTF csv file format, in Excel or notepad
#to read in
data =pd.read_csv('../input/Affordability_Income_2018Q3.csv')

data.head(10)
headerNames = data.columns.values.tolist()
print(headerNames, 
      '\n\nNumber of Columns:', len(headerNames))

print("There are ", len(data['RegionID']), " observations in this dataframe.\n")

Top10 = data.head(11)
print(Top10)

data['RegionName'].replace('United States', 'USA', inplace = True)
data['RegionName'].replace('New York, NY', 'NYC', inplace = True)
data['RegionName'].replace('Los Angeles-Long Beach-Anaheim, CA', 'LA-OC', inplace = True)
data['RegionName'].replace('Chicago, IL','Chi-Town', inplace = True)
data['RegionName'].replace('Dallas-Fort Worth, TX','Dallas', inplace = True)
data['RegionName'].replace('Philadelphia, PA','Philly', inplace = True)
data['RegionName'].replace('Houston, TX','Houston', inplace = True)
data['RegionName'].replace('Washington, DC','Washington', inplace = True)
data['RegionName'].replace('Miami-Fort Lauderdale, FL','Miami', inplace = True)
data['RegionName'].replace('Atlanta, GA','Atlanta', inplace = True)
data['RegionName'].replace('Boston, MA','Boston', inplace = True)


Top10 = data.head(11)
print(Top10)
type(Top10)

Name = Top10['RegionName']
print(Name)
type(Name) #Dataframe

Last40Springs = Top10.loc[:,'1979-03':'2018-09':20]

Forty = pd.concat([Name, Last40Springs], axis=1)
print(Forty)
type(Forty)
Last = Last40Springs.T

print(Last)


x = Last.index
print(x)

USA = Last[0]
print(USA)
USA.plot()
plt.title('map')

plt.plot(Last[0],x)
plt.title('Spring 1979 to 2018 in Five Year Increments')

#plot 7 regions and US as a line graph with the dates on the x-axis, and each y-axis the region value by date
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow','black','grey','purple'])
plt.plot(x, Last[0])
plt.plot(x, Last[1])
plt.plot(x, Last[2])
plt.plot(x, Last[3])
plt.plot(x, Last[4])
plt.plot(x, Last[5])
plt.plot(x, Last[6])




plt.legend(['USA', 'NYC', 'LA', 'Chicago','Dallas','Philly','Houston'], loc='upper left')
plt.title('Six Largest US Cities and USA Home Affordability Income March 1979-2018')
plt.show()

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools # for sub figures

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and Affordable Income amount on the y axis
USAplot = [go.Scatter(x=Last.index, y=Last[0])]


# specify the layout of our figure
layout_dates = dict(title = "USA and NYC Affordable Mortgage Income by year 1979-2018",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = USAplot, layout = layout_dates)
iplot(fig)
# Interactive plot with plotly for python 3

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools # for sub figures

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

USA = go.Scatter(
    x=Last.index,
    y=Last[0],
    name='USA'
)
NYC = go.Scatter(
    x=Last.index,
    y=Last[1],
    name='NYC'
)
LA = go.Scatter(
    x=Last.index,
    y=Last[2],
    name='LA'
)
Chicago = go.Scatter(
    x=Last.index,
    y=Last[3],
    name='Chicago'
)
Dallas = go.Scatter(
    x=Last.index,
    y=Last[4],
    name='Dallas'
)
Philly = go.Scatter(
    x=Last.index,
    y=Last[5],
    name='Philly'
)
Houston = go.Scatter(
    x=Last.index,
    y=Last[6],
    name='Houston'
)

data = [USA, NYC, LA, Chicago, Dallas, Philly, Houston]

# specify the layout of our figure
layout_dates = dict(title = "USA and Top Six Largest Regions of Affordable Mortgage Income by year 1979-2018",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False), showlegend=True)

# create and show our figure
fig = dict(data = data, layout = layout_dates)
iplot(fig)