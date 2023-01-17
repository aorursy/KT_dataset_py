import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bokeh.io import output_file,show,output_notebook,save
from bokeh.plotting import figure
df = pd.read_csv("../input/airpassengers-dataframe/AirPassengers.csv")
df["Avg"] = df.mean(axis=1,skipna=True)
df
df["Sum"] = df.iloc[:,:12].sum(axis=1,skipna=True)
df
df["Year"] = [i for i in range(1949,1961)]
df
fig = figure(x_axis_label = "Year",y_axis_label = "Average and Max of Air Passengers per Year")
fig.circle(df["Year"],df["Avg"])
output_notebook()
show(fig)
save(fig,"circle_glyphs.html")
fig.line(df["Year"],df["Avg"])
show(fig)
df["Max"] =  df.iloc[:,:12].max(axis=1)
df
fig.x(df["Year"],df["Max"],line_color="red",size=10)
show(fig)
fig.line(df["Year"],df["Max"],line_color="red")
show(fig)
output_file("line_circle_glyphs.html")
fig_patches = figure(x_axis_label = "Year",y_axis_label = "Air Passengers") 
x = [df.iloc[0:,i].values for i in range(12)]
y = [df.iloc[0:,13].values for i in range(12)]
list(x)
fig_patches.patches(y,x)
show(fig_patches)
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
hv.extension('bokeh')
output_notebook()
cars_df = pd.read_csv("../input/cars93/Cars93.csv")
mapper = linear_cmap(field_name='y', palette=Spectral6 ,low=min(np.array(y[0])) ,high=max(np.array(y[0])))
box = hv.BoxWhisker(cars_df,"Price","Horsepower", label="Box Plot",color=mapper,fill_color=mapper,fill_alpha=1, size=12)
box.options(show_legend=True, width=800)
show(box)
