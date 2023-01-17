import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))
lemon = pd.read_csv("../input/Lemonade2016-2.csv")
lemon
lemon['Sales'] = lemon['Lemon']+lemon['Orange']
lemon['Revenue'] = lemon['Sales']*lemon['Price']
lemon
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
#output_file("Sukul-DS201X-HW5-plot1.html")
output_notebook()
lemon
# create a new plot with default tools, using figure
p = figure(title="Sales by Temperature. With Revenue as Circle size", x_axis_label = "Temperature", y_axis_label = "Sales")

#plot_width=600, plot_height=600,

# add a circle renderer with a size, color, and alpha
p.circle(lemon['Temperature'], lemon['Sales'], size=lemon['Revenue']/2, line_color="navy", fill_color="orange", fill_alpha=0.5)
#p.title("chart1")
#p.line(auto_clean2['engine-size'], auto_clean2['price'], legend="make.", line_width=2)

show(p) # show the results
#output_file("Sukul-DS201X-HW5-plot2.html")
# reference https://bokeh.pydata.org/en/latest/docs/user_guide/quickstart.html#userguide-quickstart

from bokeh.plotting import *
from bokeh.models import ColumnDataSource


# NEW: create a column data source for the plots to share
D_source = ColumnDataSource(lemon)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

p1 = figure(tools=TOOLS, title="Revenue by Leaflets")
p1.circle('Leaflets', 'Revenue', color="blue", source=D_source)

p2 = figure(tools=TOOLS, title="Revenue vs. Sales")
p2.circle('Sales', 'Revenue', color="green", source=D_source)

p3 = figure(tools=TOOLS, title="Sales by Temperature")
p3.circle( 'Temperature','Sales',  line_color="red", fill_color=None, source=D_source)

p = gridplot([[ p1, p2, p3]], toolbar_location="right")

# show the results
show(p)
lemon.boxplot(column="Revenue", by="Location", figsize=(15,10), fontsize=15,rot=45)
p = figure(title="Lemon and Orange sales by Temperature")

#p.vbar(x=Revenue, bottom=avg-std, top=avg+std, width=0.8, 
#       fill_alpha=0.2, line_color=None, legend="MPG 1 stddev")

p.circle(x=lemon["Temperature"], y=lemon["Lemon"], size=10, alpha=0.5,
         color="green", legend="Lemon")

p.triangle(x=lemon["Temperature"], y=lemon["Orange"], size=10, alpha=0.3,
           color="orange", legend="Orange")

p.legend.location = "top_left"

show(p)