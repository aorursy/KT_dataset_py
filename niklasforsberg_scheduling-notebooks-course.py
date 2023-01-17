# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.offline as offline
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/restaurant-scores-lives-standard.csv")
data['Date'] = pd.to_datetime(data['inspection_date'], format = "%Y-%m-%d")
data["year"] = data.Date.dt.year

#Inspections per month, shamelessly stolen from rachael
inspects_per_month = data['Date'].groupby([data.Date.dt.year, data.Date.dt.month]).agg('count') 
inspects_per_month = inspects_per_month.to_frame()
inspects_per_month['Year_month'] = inspects_per_month.index
inspects_per_month = inspects_per_month.rename(columns={inspects_per_month.columns[0]:"counts"})
inspects_per_month['Year_month'] = pd.to_datetime(inspects_per_month['Year_month'], format="(%Y, %m)")
inspects_per_month = inspects_per_month.reset_index(drop=True)
inspects_per_month['month'] = inspects_per_month.Year_month.dt.month
inspects_per_month['year'] = inspects_per_month.Year_month.dt.year
inspects_per_month["Y_M"] = inspects_per_month.Year_month.astype(str)
from bokeh.plotting import ColumnDataSource, figure
from bokeh.io import output_notebook, show
from bokeh.models import HoverTool

source = ColumnDataSource(inspects_per_month)
p = figure(x_axis_type="datetime", x_axis_label='Date', y_axis_label='Counts',
           title="Number of inspections per month", plot_width=800, plot_height=300)
p.line("Year_month", "counts", source=source)
p.add_tools(HoverTool(tooltips=[('date','@Y_M'), ('counts', '@counts')]))
output_notebook()
show(p)

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data_plotly = [go.Scatter(x=inspects_per_month.Year_month, y=inspects_per_month.counts)]

# specify the layout of our figure
layout = dict(title = "Number of inspections per month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data_plotly, layout = layout)
iplot(fig)
# Yearly development of number of inspections for the 20 most inspected
business_20 = data.business_name.value_counts(sort=True).nlargest(20)
names_top20 = business_20.keys()
names_top20 = list(names_top20)
data_top20 = data.loc[data["business_name"].isin(names_top20)]
##data_top20.business_name.unique().sort() == names_top20.sort()
data20 = data_top20
data20_gr = data20.groupby(["business_name","year"])
inspections_per_business = data20_gr.business_name.count()
plt.figure(figsize=(20,10))
plt.xticks(rotation=90, size = 15)
c_plot = sns.countplot(x="business_name", hue="year", data = data20)
plt.title("Inspections per year for 20 most inspected", fontsize = 20)
plt.legend(loc = "upper center", prop={'size': 20})
plt.show()
inspec = data[(data["inspection_type"] == "Routine - Unscheduled")
              | (data["inspection_type"] == "Complaint") | (data["inspection_type"] == "Reinspection/Followup") ]
# Risk categories in relation to inspection type
#g = sns.catplot(x="risk_category", col="inspection_type",
#...                 data=inspec, kind="count",
#...                 height=4, aspect=.7)
x, y, z = "risk_category", "prop", "inspection_type"
prop_df = (inspec.groupby(z)[x]
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())
plt.figure(figsize=(20,10))
g = sns.barplot(x="risk_category", y="prop", hue="inspection_type", data=prop_df)
## for index, row in prop_df.iterrows():
##    g.text(row.name, row.prop, round(row.prop,2), color='black', ha="center")
plt.title("Proportion of risk categories for a number of inspection types", size = 20)
plt.show()
#Glimpse what food guidelines a certain company has violated
hr = data[(data["risk_category"] == "High Risk")]
hr_gr = hr.groupby("business_name").inspection_id.count().nlargest(30)
##hr_gr
hr = hr[["business_name", "violation_description"]]
##hr[hr["business_name"] == "Mixt Greens/Mixt"].groupby("violation_description").count()