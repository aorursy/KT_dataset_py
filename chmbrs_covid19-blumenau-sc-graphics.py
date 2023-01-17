# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-cases-blumenausc/covid_blumenau.csv')
import plotly.express as px
column_name = 'Total Cases'
fig = px.line(df, x='Date', y=column_name, title=f'Covid19 {column_name} in Blumenau SC')
fig.update_layout(yaxis_type="log")
fig.show()
column_name = 'New Cases'
fig = px.bar(df, x='Date', y=column_name, title=f'Covid19 {column_name} in Blumenau SC')
fig.show()
column_name = 'Active Cases'
fig = px.line(df, x='Date', y=column_name, title=f'Covid19 {column_name} in Blumenau SC')
fig.update_layout(yaxis_type="log")
fig.show()
column_name = 'New Deaths'
fig = px.bar(df, x='Date', y=column_name, title=f'Covid19 {column_name} in Blumenau SC')
fig.update_yaxes(dtick = 1)
fig.show()
column_name = 'Total Deaths'
fig = px.line(df, x='Date', y=column_name, title=f'Covid19 {column_name} in Blumenau SC')
fig.update_yaxes(dtick = 1)
fig.show()
fig = px.line(df, x='Total Cases', y='New Cases', title=f'Covid19 New X Total Cases in Blumenau SC')
fig.show()
