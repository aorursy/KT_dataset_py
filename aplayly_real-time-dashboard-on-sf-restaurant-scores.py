%matplotlib inline
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, Slider, HoverTool
from bokeh.layouts import row, column
import datetime
from dateutil.relativedelta import relativedelta

# Import data as Time Series
data = pd.read_csv('../input/restaurant-scores-lives-standard.csv', index_col='inspection_date', parse_dates=True)

# Data Manipulation
# Convert data type to category
data_to_cat = ['business_id','business_name','business_address','inspection_score','inspection_type',
              'violation_description','risk_category']
data[data_to_cat] = data[data_to_cat].astype('category')

# Replace NaN Value with string 'None' for phone number and postal code
data.business_phone_number.fillna(value='None', inplace=True)
data.business_postal_code.fillna(value='None', inplace=True)
# Create Dashboard that plot inspections and violations per Day
## Plot for Inspections
violation_per_day = pd.DataFrame(data.groupby(data.index)['inspection_id'].count())
violation_per_day.rename(columns={'inspection_id':'total_inspection'}, inplace=True)
source = ColumnDataSource(violation_per_day)

## Add HoverTool object
hover = HoverTool(
    tooltips=[
        ("date", "@inspection_date{%F}"),
        ("Total", "@total_inspection"),
    ],
    formatters={
        'inspection_date' : 'datetime' 
    },
    mode='vline'
)

## Parameter for figure
latest = max(data.index.date)
x_month_ago = latest - relativedelta(months=6)
x_range = [x_month_ago, latest]
p1 = figure(plot_width=800, plot_height=400, x_axis_type='datetime', x_range=x_range, y_range=[0,120],
           x_axis_label = 'Date', y_axis_label = 'Total', title='Number of Inspections per Day')
p1.line(x='inspection_date', y='total_inspection', source=source)
p1.add_tools(hover)
p1.title.text_font_size = '16pt'
p1.toolbar_location = 'below'

## Plot violation based on risk category
risk = data.groupby([data.index,'risk_category'])['violation_id'].count()
risk = pd.DataFrame(risk)
risk.reset_index(inplace=True)
risk = risk.pivot(index='inspection_date', columns='risk_category', values='violation_id').fillna(value=0)
risk.columns=['High Risk','Low Risk','Moderate Risk']
source = ColumnDataSource(risk)

## Add HoverTool object
hover = HoverTool(
    tooltips=[
        ("date", "@inspection_date{%F}"),
        ("Count", "@$name"),
    ],
    formatters={
        'inspection_date' : 'datetime' 
    },
    mode='mouse'
)

## Parameter to figure
latest = max(data.index.date)
x_month_ago = latest - relativedelta(months=6)
x_range = [x_month_ago, latest]
p2 = figure(plot_width=800, plot_height=400, x_axis_type='datetime', x_range=x_range, y_range=[0,80],
           x_axis_label = 'Date', y_axis_label = 'Count', title='Number of Violations per Day')
p2.line(x='inspection_date', y='High Risk', source=source, line_color='red', name='High Risk', legend='High')
p2.line(x='inspection_date', y='Moderate Risk', source=source, line_color='orange', name='Moderate Risk', legend='Moderate')
p2.line(x='inspection_date', y='Low Risk', source=source, line_color='green', name='Low Risk', legend='Low')
p2.add_tools(hover)
p2.title.text_font_size = '16pt'
p2.toolbar_location = 'below'
p2.legend.location = "top_left"
p2.legend.click_policy="hide"

# Plot everything in here
layout1 = column(p1, p2)

# Show plot
output_notebook()
show(layout1)
# Create Dashboard that plot Top 5 Worst Restaurants 3-Monthly and Yearly
## Take restaurant with highest 'High Risk' in this year
today = datetime.date.today()
worst_yr = data[data.index.year == today.year]
worst_yr = worst_yr.groupby(['business_name','risk_category'])['violation_id'].count()
worst_yr = pd.DataFrame(worst_yr)
worst_yr.reset_index(inplace=True)
worst_yr = worst_yr.pivot(index='business_name',columns='risk_category',values='violation_id').fillna(value=0)
worst_yr.columns = ['High Risk', 'Low Risk', 'Moderate Risk']
worst_yr.head()
worst_yr = worst_yr.sort_values(['High Risk', 'Moderate Risk', 'Low Risk'], ascending=False).head()

## Parameter for figure
source = ColumnDataSource(worst_yr)
p3 = figure(plot_width=800, plot_height=400, title='Top 5 Worst Restaurant This Year', y_range=worst_yr.index.tolist()[::-1])
risk_cat = ["High Risk","Moderate Risk", "Low Risk"]
p3.hbar_stack(risk_cat, y='business_name', height=0.5, color=['red','orange','yellow'], source=source,
             legend=["High","Moderate","Low"])
p3.title.text_font_size = '16pt'
p3.toolbar_location = 'below'
p3.legend.location = "top_left"


## Take restaurant with highest 'High Risk' in the past 3 months
today = datetime.date.today()
eq_yr = data.index.year == today.year
eq_mo = data.index.month == today.month
eq_mo_1 = data.index.month == today.month-1
eq_mo_2 = data.index.month == today.month-2
logic = (data.index.year == today.year) & np.logical_or(np.logical_or(eq_mo, eq_mo_1),eq_mo_2)
worst_mo = data[logic]
worst_mo = worst_mo.groupby(['business_name','risk_category'])['violation_id'].count()
worst_mo = pd.DataFrame(worst_mo)
worst_mo.reset_index(inplace=True)
worst_mo = worst_mo.pivot(index='business_name',columns='risk_category',values='violation_id').fillna(value=0)
worst_mo.columns = ['High Risk', 'Low Risk', 'Moderate Risk']
worst_mo.head()
worst_mo = worst_mo.sort_values(['High Risk', 'Moderate Risk', 'Low Risk'], ascending=False).head()

## Parameter for figure
source = ColumnDataSource(worst_mo)
p4 = figure(plot_width=800, plot_height=400, title='Top 5 Worst Restaurant in 3 Month', y_range=worst_mo.index.tolist()[::-1])
risk_cat = ["High Risk","Moderate Risk", "Low Risk"]
p4.hbar_stack(risk_cat, y='business_name', height=0.5, color=['red','orange','yellow'], source=source,
             legend=["High","Moderate","Low"])
p4.title.text_font_size = '16pt'
p4.toolbar_location = 'below'
p4.legend.location = "top_left"

# Plot everything in here
layout2 = column(p3, p4)

# Show plot
output_notebook()
show(layout2)