# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_proc_data = pd.read_csv('../input/procurement-notices.csv')
df_proc_data.columns
df_proc_data.info()
df_proc_data['Deadline Date'] = pd.to_datetime(df_proc_data['Deadline Date'])
df_proc_data.info()
from datetime import date
value_to_check = pd.Timestamp(date.today())
print(value_to_check)
# Filter the rows that has deadline date greater than today

filter_criteria = df_proc_data['Deadline Date'] > value_to_check
df_proc_data_filtered = df_proc_data[filter_criteria]

df_proc_data_filtered.count()
proc_data_by_country = df_proc_data_filtered.groupby('Country Name').size()
proc_data_by_country
# Distributon by Country
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected = True)

data = [
    go.Choropleth(
        locationmode = 'country names',
        locations = proc_data_by_country.index,
        z = proc_data_by_country.values
    )
]
layout = dict(
        title = 'Distribution by contry'
    )
fig = dict(data=data, layout=layout)

url = plotly.offline.iplot(fig)
# dist by due dates
proc_data_deadline_date = df_proc_data_filtered.groupby('Deadline Date').size()
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


iplot([{"x": proc_data_deadline_date.index, "y": proc_data_deadline_date.values}])
# dist by Procurement Type

proc_data_procurement_type = df_proc_data_filtered.groupby('Procurement Type').size()
proc_data_procurement_type.index
iplot([{"x": proc_data_procurement_type.index, "y": proc_data_procurement_type.values}])
