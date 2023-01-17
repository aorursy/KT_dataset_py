import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,save

from bokeh.plotting import figure
df = pd.read_csv("../input/airpassengers-dataframe/AirPassengers.csv")

df["Avg"] = df.mean(axis=1,skipna=True)

df["Sum"] = df.iloc[:,:12].sum(axis=1,skipna=True)

df["Year"] = [i for i in range(1949,1961)]
fig = figure(x_axis_label = "Year",y_axis_label = "Average and Max of Air Passengers per Year")
fig.circle(df["Year"],df["Avg"])
output_notebook()
show(fig)