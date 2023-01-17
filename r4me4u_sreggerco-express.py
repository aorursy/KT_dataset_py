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
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head()
import plotly.express as px
px.box(df, x='Genre', y='Global_Sales')
grouped_year = df.groupby('Year').mean()
grouped_year = grouped_year[['Global_Sales']]
grouped_year.columns = ['Avg_global_sales']
maxed_year = df.groupby('Year')['Global_Sales'].max()
grouped_year.head()
px.bar(df, x='Year', y='Global_Sales', hover_data=['Name'])
import plotly.graph_objects as go
fig = px.line(grouped_year, x=grouped_year.index, y='Avg_global_sales')
fig.add_scatter(x=maxed_year.index ,y=maxed_year, mode='lines')
grouped_year.head(100)
top_thousand = df[df['Rank'] < 1000]
top_ten = df[df['Rank'] < 10]

px.bar(top_thousand, x='Genre', y='Global_Sales', color='Platform', hover_data=['Name'])
px.histogram(top_ten, x='Name', y='Global_Sales', color='Platform', histfunc='sum')
poep = 4

