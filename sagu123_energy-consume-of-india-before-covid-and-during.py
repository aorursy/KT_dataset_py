!pip install pyforest

from pyforest import *
df= pd.read_csv("../input/state-wise-power-consumption-in-india/long_data_.csv")
df.head()
df.dtypes
df["Dates"]= pd.to_datetime(df.Dates)
df.head()
df.dtypes
df.Dates.dt.month_name()
g=df.groupby(by="States")
g
for state, state_df in g:
  print(state)
  print(state_df)
punjab=g.get_group("Punjab")
punjab.head()
punjab.tail()
ax=sns.relplot(x="Dates",y="Usage", data=punjab, kind="line", markers=True)
for axes in ax.axes.flat:
  axes.set_xticklabels(axes.get_xticklabels(), rotation=65, horizontalalignment='right')
from bokeh.models import HoverTool
import bokeh

from bokeh.io import output_notebook, reset_output, show

from bokeh.plotting import figure

import numpy as np
import pandas as pd

output_notebook()

from bokeh.models import ColumnDataSource
line_plot= figure(plot_width=700, plot_height=300, title="line plot", x_axis_label="Dates", y_axis_label= "Usage", toolbar_location="below")

line_plot.line(df.Dates,df.Usage, legend_label="line")

line_plot.add_tools(HoverTool())

show(line_plot)
punjab=punjab.reset_index()
punjab.head()
ax=sns.barplot(x="States", y="Usage", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=65, horizontalalignment='right')

df.columns
df.dtypes
df2= pd.read_csv("../input/state-wise-power-consumption-in-india/dataset_tk.csv")

df2.head()
df2.columns
df2["Dates"]= df2["Unnamed: 0"]
df2.head()
df2= df2.drop("Unnamed: 0", axis=1)
df2.tail()
df2= df2.set_index("Dates")
df2.head()

df2['NR'] = df2['Punjab']+ df2['Haryana']+ df2['Rajasthan']+ df2['Delhi']+df2['UP']+df2['Uttarakhand']+df2['HP']+df2['J&K']+df2['Chandigarh']
df2['WR'] = df2['Chhattisgarh']+df2['Gujarat']+df2['MP']+df2['Maharashtra']+df2['Goa']+df2['DNH']
df2['SR'] = df2['Andhra Pradesh']+df2['Telangana']+df2['Karnataka']+df2['Kerala']+df2['Tamil Nadu']+df2['Pondy']
df2['ER'] = df2['Bihar']+df2['Jharkhand']+ df2['Odisha']+df2['West Bengal']+df2['Sikkim']
df2['NER'] =df2['Arunachal Pradesh']+df2['Assam']+df2['Manipur']+df2['Meghalaya']+df2['Mizoram']+df2['Nagaland']+df2['Tripura']
df2.head()
df2["NR"].values

import plotly.graph_objects as go
fig = go.Figure( go.Scatter(x=df2.index, y=df2["Punjab"]))
fig.show()

df_new = pd.DataFrame({"Northern Region": df2["NR"].values,
                        "Southern Region": df2["SR"].values,
                        "Eastern Region": df2["ER"].values,
                        "Western Region": df2["WR"].values,
                        "North Eastern Region": df2["NER"].values},index=df2.index)

df_new.head()
fig2 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Northern Region"],fillcolor=None))
fig2.show()

fig3 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Southern Region"],fillcolor=None))
fig3.show()
fig4 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Eastern Region"],fillcolor=None))
fig4.show()
fig5 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Western Region"],fillcolor=None))
fig5.show()
fig6 = go.Figure( go.Scatter(x=df_new.index, y=df_new["North Eastern Region"],fillcolor=None))
fig6.show()
sns.distplot(df_new["Northern Region"], bins=10)
sns.distplot(df_new["Southern Region"], bins=10)
sns.distplot(df_new["Eastern Region"], bins=10)
sns.distplot(df_new["Western Region"], bins=10)
sns.distplot(df_new["North Eastern Region"], bins=10)
sns.pairplot(df)
sns.heatmap(df_new.corr(),annot=True)
!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title='Power Stats', html={'style':{'full_width':False}})
profile.to_widgets()
profile
#x=df.loc[df["Dates"]>"2020-02-01"]