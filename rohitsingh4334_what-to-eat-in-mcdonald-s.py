import pandas as pd 

import numpy as np 

from bokeh.io import output_notebook

from bokeh.models import HoverTool

from bokeh.charts import Bar, show

from bokeh.layouts import column,row

from bokeh.palettes import Blues

%matplotlib inline

output_notebook()
df = pd.read_csv('../input/menu.csv')
df.isnull().any()
df.shape 
df.columns
#Top 10 rows

df.head(4)
df.groupby(['Category'])['Calories'].count()
sum_bar = Bar(df,'Category',values='Calories', title="Food Category by Calories",plot_width=500,agg='sum',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

count_bar = Bar(df,'Category',values='Calories', title="Food Category by Calories",plot_width=500,agg='count',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

mean_bar = Bar(df,'Category',values='Calories', title="Food Category by Calories",plot_width=500,agg='mean',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

hover = sum_bar.select(dict(type=HoverTool))

hover = count_bar.select(dict(type=HoverTool))

hover = mean_bar.select(dict(type=HoverTool))

hover.tooltips=[("Category:", "@x"), ("Calories:", "$y")]

show(column(sum_bar,count_bar,mean_bar))
#This function help us to create bar plot for Food category w.r.t Nutrient

Calories_mean_bar = Bar(df,'Category',values='Calories', title="Food Category by Calories",plot_width=500,agg='mean',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

hover = Calories_mean_bar.select(dict(type=HoverTool))

hover.tooltips=[("Category:", "@x"), ("Calories:", "$y")]

show(Calories_mean_bar)
Fat_mean_bar = Bar(df,'Category',values='Total Fat', title="Food Category by Fat",plot_width=500,agg='mean',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

hover = Fat_mean_bar.select(dict(type=HoverTool))

hover.tooltips=[("Category:", "@x"), ("Calories:", "$y")]

show(Fat_mean_bar)
Cholesterol_mean_bar = Bar(df,'Category',values='Cholesterol', title="Food Category by Cholesterol",plot_width=500,agg='mean',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

hover = Cholesterol_mean_bar.select(dict(type=HoverTool))

hover.tooltips=[("Category:", "@x"), ("Calories:", "$y")]

show(Cholesterol_mean_bar)
Carbohydrates_mean_bar = Bar(df,'Category',values='Carbohydrates', title="Food Category by Carbohydrates",plot_width=500,agg='mean',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

hover = Carbohydrates_mean_bar.select(dict(type=HoverTool))

hover.tooltips=[("Category:", "@x"), ("Calories:", "$y")]

show(Carbohydrates_mean_bar)
Sugars_mean_bar = Bar(df,'Category',values='Sugars', title="Food Category by Sugars",plot_width=500,agg='mean',plot_height=300,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

hover = Sugars_mean_bar.select(dict(type=HoverTool))

hover.tooltips=[("Category:", "@x"), ("Calories:", "$y")]

show(Sugars_mean_bar)
df_BreakFast = df[df['Category']=='Breakfast']

df_BreakFast.head(5)
df_BreakFast_bar = Bar(df_BreakFast,'Item',values='Calories', title="Breakfast by Calories",plot_width=1000,agg='sum',plot_height=600,legend=False,tools='hover,pan,wheel_zoom,box_zoom,reset,resize')

hover = df_BreakFast_bar.select(dict(type=HoverTool))

hover.tooltips=[("Breakfast:", "@x"), ("Calories:", "$y")]

show(df_BreakFast_bar)