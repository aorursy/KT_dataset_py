# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd





from bokeh.plotting import figure, output_file, show, save

from bokeh.layouts import gridplot

from bokeh.layouts import row

from bokeh.layouts import column

from bokeh.models.tools import HoverTool

from bokeh.models import ColumnDataSource
data_url = '../input/vcscf-rate/vcscf.csv'

df = pd.read_csv(data_url, index_col=0,thousands=',') #index_col=0 means first column will become index, otherwise specify with column name 'example name'

## thousands=',' means treat ',' in number exp 1,800 as figure

print (df.index)

print (df.dtypes)
df['End Time'] = pd.to_datetime(df['End Time'])



df['date'] = [d.date() for d in df['End Time']]

df['time'] = [d.time() for d in df['End Time']]



df['date'] = pd.to_datetime(df['date'])



print (df['date'])

print (df['time'])
df1=df.loc[df['date']=="2020-08-10"]

df2=df.loc[df['date']=="2020-08-11"]

source1 = ColumnDataSource(data={

 'date'  : df1['date'],

 'time'  : df1['time'],

 'count' : df1['Rate of Successful Reregistration(%)'].values

 })



source2 = ColumnDataSource(data={

 'date'  : df2['date'],

 'time'  : df2['time'],

 'count' : df2['Rate of Successful Reregistration(%)'].values

 })
colour=['#ff7f0e','#2ca02c','#d62728','#1f77b4'] ##bokeh D3 Palettes

output_file("test1.html")

p = figure(x_axis_label='Time', y_axis_label='Rate of Successful Reregistration(%)',x_axis_type='datetime',title="Graph 1", plot_width=1100, plot_height=400)

p.line('time','count',source=source1,line_width=2, color=str(colour[0]), legend_label=str(df1.iloc[1,10].date()))

p.line('time','count',source=source2,line_width=2, color=str(colour[1]), legend_label=str(df2.iloc[1,10].date()))





p1 = figure(x_axis_label='Time', y_axis_label='Rate of Successful Reregistration(%)',x_axis_type='datetime',title="Graph 2", plot_width=1100, plot_height=400)

p1.line('time','count',source=source1,line_width=2, color=str(colour[0]), legend_label=str(df1.iloc[1,10].date()))

p.legend.click_policy="hide"

p1.legend.click_policy="hide"



# add a hover tool and show the date in date time format

hover = HoverTool(mode='mouse')

hover.tooltips=[

    ('Date', '@date{%F}'),    ##date format wanted to show

    ('Time', '@time{%H:%M}'), ##time format wanted to show

    ('Count', '@count'),      ##count

    #('(x,y)', '($x, $y)') ##x-coordinates and y-coordinates of the graph

]

hover.formatters = {'@date': 'datetime','@time': 'datetime'}



p.add_tools(hover)

p1.add_tools(hover)
q=column(p,p1)

#show(q)

save(q)