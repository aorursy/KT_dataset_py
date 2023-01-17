# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
proc = pd.read_csv("../input/procurement-notices.csv")
proc.head(10)

print("Types of Procurement:\n",proc["Procurement Type"].unique())
print("Types of Notices",proc["Notice Type"].unique())
display(proc.groupby("Notice Type").size().to_frame(name="Count()"))
print("No of countries",proc["Country Name"].nunique())

print("Total Entries",proc.shape[0])
proc.groupby("Notice Type").size().to_frame(name="No. of Procurements").T.plot.bar(color=['c','g','y','k','m'],rot=0);
country_proc_count = proc.groupby("Country Name").size().sort_values(ascending=False).to_frame(name="Count")
display(country_proc_count)

country_proc_count[country_proc_count["Count"]>10].T.plot.bar(rot=0).legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3);
proc["deadline_converted"] = pd.to_datetime(proc["Deadline Date"]).to_frame()
proc
today = pd.to_datetime('today')
proc[proc["deadline_converted"] > today].shape[0]



proc.groupby("Deadline Date").size().to_frame()
proc[proc.deadline_converted.notnull() & (proc.deadline_converted > today )].groupby(proc.deadline_converted).size().plot.line();
proc.deadline_converted.dt.year ==2018

proc_month = proc.groupby([proc.deadline_converted.dt.month,proc.deadline_converted.dt.year]).size()
proc_month.to_frame()
proc_month = proc_month.to_frame('count')
display(proc_month)
proc_month['date'] = proc_month.index
proc_month = proc_month.reset_index(drop=True)
proc_month['date'] = pd.to_datetime(proc_month['date'], format="(%m.0, %Y.0)")
display(proc_month)
proc_month['count']

proc[proc["Country Name"] == "India"].groupby("Notice Type").size().to_frame(name="Types").T.plot.bar(rot=0).legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1);
date_count_deadline = proc[proc.deadline_converted.notnull() & (proc.deadline_converted > today )].groupby(proc.deadline_converted.rename('date')).size()
date_count_deadline = date_count_deadline.reset_index()
date_count_deadline = date_count_deadline.rename(columns={0:"count"})
proc_month = proc_month.sort_values(by='date')
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=date_count_deadline.date, y=date_count_deadline['count'])]

# specify the layout of our figure
layout = dict(title = "Number of Deadlines",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

country_count_today = proc[proc.deadline_converted.notnull() & (proc.deadline_converted>today)].groupby(['Country Name','Country Code']).size().reset_index()
country_count_today = country_count_today.rename(columns = {0:'size'})
country_count_today
init_notebook_mode()
data = [ dict(
        type = 'choropleth',
        locations = country_count_today['Country Name'],
        z = country_count_today['size'],
        locationmode = "country names",
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
      ) ]
layout = dict(
    title = '',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'robinson'
        ),showcountries=True, countrycolor= "#444444"
    )
)

fig = dict(data = data , layout = layout)
iplot(fig)

