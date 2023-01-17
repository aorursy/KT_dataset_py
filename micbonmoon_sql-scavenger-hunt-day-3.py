# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
query = """
select extract(hour from timestamp_of_crash),
               count(consecutive_number) 
from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
group by extract(hour from timestamp_of_crash)
order by count(consecutive_number) desc
"""

count_by_hour = accidents.query_to_pandas_safe(query)
count_by_hour.columns = ['hour', 'count']
print(count_by_hour)
from bokeh.io import output_notebook
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
output_notebook()
source = ColumnDataSource(count_by_hour.sort_values('hour'))
hover = HoverTool(tooltips=[
    ('hour', '@hour'),
    ('count', '@count')
])
p = figure(
    plot_height=300, plot_width=400,
    title="Number of accidents by hour 2015",
    tools=[hover, 'box_zoom', 'reset']
)
p.sizing_mode = 'scale_width'
p.line(x='hour', y='count', source=source)
p.xaxis.axis_label = 'Hour'
p.yaxis.axis_label = 'Count'
show(p)
print('Accidents occurred most frequently during hour '
      + str(count_by_hour['hour'][0]) + '.')
query = """
select registration_state_name,
               count(consecutive_number) 
from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
where hit_and_run = 'Yes'
group by registration_state_name
order by count(consecutive_number) desc
"""
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
hit_and_run_by_state.columns = ['state', 'count']
print(hit_and_run_by_state)
source = ColumnDataSource(hit_and_run_by_state[:10])
hover = HoverTool(tooltips=[
    ('state', '@state'),
    ('count', '@count')
])
p = figure(
    y_range=list(hit_and_run_by_state['state'][:10][::-1]),
    plot_height=400, plot_width=600,
    title="Top 10 states with most hit-and-run vehicles 2015",
    tools=[hover, 'box_zoom', 'reset']
)
p.sizing_mode = 'scale_width'
p.hbar(y='state', right='count', height=0.7, source=source)
p.xaxis.axis_label = 'State'
p.yaxis.axis_label = 'Count'
show(p)
print('The state with the most hit-and-run vehicles is '
      + str(hit_and_run_by_state.iloc[0,0])+'.')
hit_and_run_by_known_state = hit_and_run_by_state[hit_and_run_by_state['state']!='Unknown']
source = ColumnDataSource(hit_and_run_by_known_state[:10])
hover = HoverTool(tooltips=[
    ('state', '@state'),
    ('count', '@count')
])
p = figure(
    y_range=list(hit_and_run_by_known_state['state'][:10][::-1]),
    plot_height=400, plot_width=600,
    title="Top 10 states with most hit-and-run vehicles 2015",
    tools=[hover, 'box_zoom', 'reset']
)
p.sizing_mode = 'scale_width'
p.hbar(y='state', right='count', height=0.7, source=source)
p.xaxis.axis_label = 'State'
p.yaxis.axis_label = 'Count'
show(p)