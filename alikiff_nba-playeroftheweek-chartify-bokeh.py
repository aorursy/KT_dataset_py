# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.transform import jitter,factor_cmap
from bokeh.plotting import figure,show, output_notebook
from bokeh.palettes import Category20
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool

import chartify
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files 
#in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/NBA_player_of_the_week.csv")
df.head()
for var in ["Age","Seasons in league"]:
    ch=chartify.Chart(blank_labels=True,y_axis_type="density",layout="slide_100%")
    hist=ch.plot.histogram(df,values_column=var,method="density",bins=len(df[var].unique()))
    
    ch.style.color_palette.reset_palette_order()
    
    kde=ch.plot.kde(df,values_column=var)
    
    ch.figure.plot_width=800

    ch.figure.renderers[-1].glyph.fill_alpha=0.1

    ch.set_subtitle(f"Player of the Week - {var} distribution")
    ch.axes.set_xaxis_label(var)
    ch.axes.set_yaxis_label("Prob. density")

    ch.show()
    
dfkum=df.groupby("Season short")["Age"].mean().to_frame().reset_index()

ch=chartify.Chart(blank_labels=True,x_axis_type="linear",y_axis_type="linear",layout="slide_100%")
ch.plot.line(dfkum,x_column="Season short",y_column="Age")
ch.figure.plot_width=800
ch.set_subtitle("Player of the Week - Mean Age per Season")

ch.axes.set_xaxis_tick_format("0")
ch.axes.set_xaxis_label="Season"

ch.figure.ygrid.grid_line_color="lightgrey"
ch.axes.set_yaxis_range(start=0,end=40)
ch.axes.set_yaxis_label="Mean Age"

ch.show()
dfkum=df.groupby("Player").agg({"Real_value":"sum"})\
        .sort_values("Real_value",ascending=False).head(20).reset_index()
dfkum["label"]=dfkum["Real_value"].astype(int)

ch=chartify.Chart(blank_labels=True,y_axis_type="categorical",layout="slide_100%")
ch.plot.bar(dfkum,categorical_columns="Player",numeric_column="Real_value",
            color_column="Player",categorical_order_ascending=True,)

ch.plot.text(dfkum,categorical_columns="Player",numeric_column="Real_value",
             text_column="label",categorical_order_ascending=True,font_size="13px",
             x_offset=-30,y_offset=1)
ch.figure.renderers[-1].glyph.text_color="white"

ch.figure.plot_width=800; ch.figure.plot_height=800
ch.set_subtitle("Player of the Week - Leaderboard (Top 20)")
ch.figure.xaxis.fixed_location=25
ch.figure.xaxis.visible=False

for i in [10,20,30]:
    ch.callout.line(location=i,orientation="height",line_color="white",
                    line_dash="dashed",line_width=1)
ch.show()
dftemp=df.groupby(["Player","Position"])["Real_value"].sum().reset_index()
source=ColumnDataSource(data=dftemp)

cmapper=factor_cmap("Position",palette=Category20[11],factors=dftemp["Position"].unique())
hover=HoverTool(tooltips=[("Player","@Player"),])

p=figure(y_range=dftemp["Position"].unique(),width=800,height=800,tools=[hover])
p.scatter(x="Real_value",y=jitter('Position', width=0.5, range=p.y_range),source=source,size=10,alpha=0.7,
          fill_color=cmapper,color=cmapper)

p.title.text="Player of the Week - Scatterplot (+Hover-Demo)"
p.title.text_font_size="16px"

p.xaxis.axis_label="Number of times Player of the Week"
p.yaxis.axis_label="Position"
p.ygrid.grid_line_color=None

show(p)