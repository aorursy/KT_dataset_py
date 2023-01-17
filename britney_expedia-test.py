import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from bokeh.plotting import figure, output_notebook, show, vplot, ColumnDataSource
from bokeh.charts import TimeSeries
from bokeh.models import HoverTool, CrosshairTool
from bokeh.palettes import brewer
import gc
import dask.dataframe as dd

output_notebook()
train  =  pd.read_csv('../input/train.csv', usecols = ('date_time', 'hotel_cluster','is_booking'), 
                      parse_dates = ['date_time'])
train['dow'] = train.date_time.dt.weekday
train['year'] = train.date_time.dt.year
train['month'] = train.date_time.dt.month
train['day'] = train.date_time.dt.day
train_agg = train.groupby(['dow','year','month','day', 'hotel_cluster']).agg(['sum', 'count'] )
train_agg.columns = ('bookings', 'total')
train_agg.head()
date_agg_1 = train_agg.groupby(level=0).agg(['sum'] )
date_agg_1.columns = ('bookings', 'total')
date_agg_1.head()