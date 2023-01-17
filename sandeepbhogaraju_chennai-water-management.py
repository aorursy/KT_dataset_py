# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read the data

rain = pd.read_csv(r'/kaggle/input/chennai_reservoir_rainfall.csv')

level = pd.read_csv(r'/kaggle/input/chennai_reservoir_levels.csv')
print(rain.head())

print('=============================================================')

print(level.head())

#Total water level per day

level['TOTAL'] =level['POONDI']+level['CHOLAVARAM']+level['REDHILLS']+level['CHEMBARAMBAKKAM']

#Total rain per day 

rain['TOTAL'] = rain['POONDI']+rain['CHOLAVARAM']+rain['REDHILLS']+rain['CHEMBARAMBAKKAM']
#Total usage per day

level['USAGE']=level['TOTAL'].diff(periods=-1)

rain['RAIN_TREND'] = rain['TOTAL'].diff(periods=-1)



from datetime import datetime

#Convert Date from str to Datetime

level['Date'] = pd.to_datetime(level['Date'],dayfirst=True)

rain['Date'] = pd.to_datetime(rain['Date'],dayfirst=True)



level.head()
rain.head()
#Visualizing the Total water levels and Usage



from bokeh.plotting import figure, output_file, show

from bokeh.layouts import gridplot,row,column

from bokeh.io import output_notebook

from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label



output_notebook() 

source = ColumnDataSource(level)

TOOLS = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,help"





plot1 = figure(tools=TOOLS,plot_width=400,plot_height=400,x_axis_type="datetime",title='POONDI LEVELS')

plot2 = figure(tools=TOOLS,plot_width=plot1.plot_width,plot_height=plot1.plot_height,x_range=plot1.x_range,y_range=plot1.y_range,x_axis_type="datetime",title='CHOLAVARAM LEVELS')

plot3 = figure(tools=TOOLS,plot_width=plot1.plot_width,plot_height=plot1.plot_height,x_range=plot1.x_range,y_range=plot1.y_range,x_axis_type="datetime",title='REDHILLS LEVELS')

plot4 = figure(tools=TOOLS,plot_width=plot1.plot_width,plot_height=plot1.plot_height,x_range=plot1.x_range,y_range=plot1.y_range,x_axis_type="datetime",title='CHEMBARAMBAKKAM LEVELS')

plot5 =figure(tools=TOOLS,plot_width=plot1.plot_width,plot_height=plot1.plot_height,x_range=plot1.x_range,x_axis_type="datetime",title='USAGE LEVELS')

plot6= figure(tools=TOOLS,plot_width=plot1.plot_width,plot_height=plot1.plot_height,x_range=plot1.x_range,x_axis_type="datetime",title='TOTAL LEVELS')



plot1.line(x='Date', y='POONDI',color='blue',source=source)

plot2.line(x='Date', y='CHOLAVARAM',color='red',source=source)

plot3.line(x='Date', y='REDHILLS',color='black',source=source)

plot4.line(x='Date', y='CHEMBARAMBAKKAM',color='green',source=source)

plot5.line(x='Date', y='USAGE',color='green',source=source)

plot6.line(x='Date', y='TOTAL',color='grey',source=source)



plot_levels =  row(column(plot1,plot2),column(plot3,plot4),column(plot5,plot6))



show(plot_levels)

#Plotting Rainfall , Usage and Total Levels to check corellation



from bokeh.plotting import figure, output_file, show

from bokeh.layouts import gridplot,row,column

from bokeh.io import output_notebook

from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label



source2 = ColumnDataSource(rain)



plot7= figure(tools=TOOLS,plot_width=plot1.plot_width,plot_height=plot1.plot_height,x_range=plot1.x_range,x_axis_type="datetime",title='RAIN LEVELS')

plot8= figure(tools=TOOLS,plot_width=plot1.plot_width,plot_height=plot1.plot_height,x_range=plot1.x_range,x_axis_type="datetime",title='RAIN TRENDS')



plot7.line(x='Date', y='TOTAL',color='blue',source=source2)

plot8.line(x='Date', y='RAIN_TREND',color='red',source=source2)



plot_rain = column(row(plot5,plot6),row(plot8,plot7))

show(plot_rain)






