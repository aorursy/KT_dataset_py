!pip install pandas-bokeh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_bokeh
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend','pandas_bokeh')
# create a bokeh table using data frame
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnarDataSource
df = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')
df.head(20)
df.plot_bokeh(kind="line",title='Length and Beats per minute Comparasion',
              figsize=(1000,800),# Figure Size 
              xlabel="Beats.Per.Minute", # X -axis Label
              ylabel="Length.") # Y-axis label
df.plot_bokeh(kind='bar',title='Energy Vs Popularity',
              figsize=(1000,800),
              xlabel="Popularity",
              ylabel="Energy")
df.plot_bokeh(kind="point",title="Dancebility Vs Liveness", figsize=(1000,800),
              xlabel="Liveness",
              ylabel="Dancebility")
df.plot_bokeh(kind="hist",title="Dancebility Vs Liveness", figsize=(1000,800),
              xlabel="Liveness",
              ylabel="Dancebility")
df.plot_bokeh(kind="line",title='Length and Beats per minute Comparasion',
              figsize=(1000,800),
              xlabel="Beats.Per.Minute",
              ylabel="Length.",rangetool=True)
df.plot_bokeh.point(x=df.Energy,xticks=range(0,1), size=5,
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title = "Point Plot - Spotify Songs",fontsize_title=20,
    marker="x",figsize =(1000,800))
df.plot_bokeh.step(
    x=df.Energy,
    xticks=range(-1, 1),
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title="Step Plot - Spotify Songs",
    figsize=(1000,800),
    fontsize_title=20,
    fontsize_label=20,
    fontsize_ticks=20,
    fontsize_legend=8,
    )