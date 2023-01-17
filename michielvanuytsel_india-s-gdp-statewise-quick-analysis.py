import numpy as np

import pandas as pd



import os

data_filepath = "../input/indias-gdp-statewise/Indias GDP (Statewise).csv"

df = pd.read_csv(data_filepath, index_col=0)
df.head()
import pandas_profiling

pandas_profile = pandas_profiling.ProfileReport(df, progress_bar=False)

pandas_profile.to_widgets()
import plotly.express as px
#Unless you need to do multiple adjustments to your figure, there is no need to split it into multiple lines.

#Below I combine your treemap statement into 1 by directly using show()

px.treemap(df, 

         path=['State_UT'],

         values = 'Nominal GDP(Trillion INR)',

         hover_data = ['Nominal GDP(Billion USD)'],

         names='State_UT', 

         color='Nominal GDP(Trillion INR)',

         height=600,

         color_continuous_scale='RdBu',

         title='Indias GDP - Statewise',

        ).show()
#You can also directly plot with Plotly as Pandas backend using the below configuration

#Based on personal experience this is not as flexible as above, but when trying to quickly explore data it might be useful

#Note that this also causes a conflict with the pandas profiling, so make sure to not run this before the pandas profiling

pd.options.plotting.backend = "plotly"



# We need to select which columns we want to plot since "State_UT" as it is not a plottable column for example

plot_features = ['Nominal GDP(Trillion INR)','Nominal GDP(Billion USD)']

df[plot_features].plot()
# You can also specify which format you'd like

df[plot_features].plot.box()