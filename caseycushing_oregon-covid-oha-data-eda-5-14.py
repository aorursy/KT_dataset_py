# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/oregon-covid-casehospitalized-oha-dataset-514/Oregons Epi Curve.csv')

data
data = data.rename(columns = {'Hostpitalized':'Hospitalized'})
data.dtypes
data['Hospitalized'] = data['Hospitalized'].fillna(0)

data['Not Hospitalized'] = data['Not Hospitalized'].fillna(0)

#data = data['Hostpitalized'].fillna('0')

#data = data['Not Hospitalized'].fillna('0')

data['Unknown Status'] = data['Unknown Status'].fillna(0)

#data = data.fillna('0')
data.dtypes
data.isna().sum()
import plotly.express as px

px.bar(data,x='Date',y='Case Count')
px.bar(data,x='Date',y='Not Hospitalized')
import statsmodels

px.scatter(data,x='Case Count',y='Not Hospitalized',hover_data=['Date'],trendline='ols')
px.scatter(data,x='Case Count',y='Hospitalized', hover_data = ['Date'],trendline='lowess')
px.bar(data,x='Date',y='Hospitalized')
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(x=data['Date'],y=data['Hospitalized']))

fig.add_trace(go.Bar(x=data['Date'],y=data['Not Hospitalized']))
              
fig.update_layout(barmode='group')
              
fig.show()
data