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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
color = sns.color_palette()
%matplotlib inline
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 100)

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
output_notebook()
df = pd.read_csv("/kaggle/input/nifty-indices-dataset/NIFTY 100.csv",parse_dates=['Date'])
def scatter_plot(count,color):
    trace = go.Scatter(
            x = count.index[::-1],
            y = count.values[::-1],
            showlegend=False,
            marker=dict(color=color))
    return trace
    
layout = go.Layout(
    title=go.layout.Title(
        text="NIFTY 100 Closing values over time",
        x=0.5
    ),
    xaxis_title="Date",
    yaxis_title="Index close value",
    font=dict(size=14),
    width=800,
    height=500,
)

count = df['Close']
index = df['Date']
trace = scatter_plot(count,'red')
data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig,filename='fig')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df_month = df.groupby(['Year','Month']).first().reset_index()
df_month
trace = go.Heatmap(
                    x = (df_month['Month'].values)[::-1],
                    y = (df_month['Year'].values)[::-1],
                    z = (df_month['P/E'].values)[::-1],
                    colorscale="rdylgn_r")

layout = go.Layout(
    title=go.layout.Title(
        text="Historical P/E values of NIFTY 100 (first trading day of each month)",
        x=0.5
    ),
    xaxis_title="Month",
    yaxis_title="Year",
    font=dict(size=14),
    width=800,
    height=800,
)
data = [trace]
fig = go.Figure(data=data,layout=layout)
fig.update_xaxes(nticks=40)
fig.update_yaxes(nticks=40)
py.iplot(fig,filename='fig')
df.columns
df.dtypes
missing_data = df.isnull()
missing_data.head()
missing_data.columns.values.tolist()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print(" ")
import seaborn as sns
sns.heatmap(df.isnull(),cmap='viridis')
average_volume = df['Volume'].astype('float32').mean(axis=0)
print("Average Volume:",average_volume)
avg_tur = df["Turnover"].astype("float").mean(axis=0)
print("Average Turnover:", avg_tur)
df['Volume'].replace(np.nan,average_volume,inplace=True)
df['Turnover'].replace(np.nan,avg_tur,inplace=True)
import seaborn as sns
sns.heatmap(df.isnull(),cmap='viridis')
sns.heatmap(df.corr(),cmap='Purples')
sns.regplot(x='Volume',y='Turnover',data=df,color='blue')
plt.ylim(0,)
df[["Volume", "Div Yield"]].corr()
sns.regplot(x='Volume',y='Div Yield',data=df,color='blue')
plt.ylim(0,)
sns.relplot(x="Month", y="Volume", kind='line',data=df,color='purple')
sns.relplot(x="Month", y="Turnover", kind='line',data=df,color='green')
sns.set_style("darkgrid")
plt.figure(figsize=(15,6))
sns.barplot(x='Year',y='Turnover',data=df)
plt.title("Turnover over years",fontsize=20)
plt.show()
plt.figure(figsize=(15,6))
sns.barplot(x='Year', y='Volume', data=df)
plt.title("Volume over years",fontsize=20)
plt.show()
plt.figure(figsize=(15,6))
sns.barplot(x='Year', y='Div Yield', data=df)
plt.title("Div Yield over years",fontsize=20)
plt.show()
plt.figure(figsize=(15,6))
sns.barplot(x='Year', y='P/E', data=df)
plt.title("P/E over years",fontsize=20)
plt.show()
plt.figure(figsize=(15,6))
sns.barplot(x='Year', y='P/B', data=df)
plt.title("P/B over years",fontsize=20)
plt.show()
df_ohlc = df[['Date','Open', 'High','Low', 'Close']]
df_ohlc.head()

import seaborn as sns; sns.set(style="ticks", color_codes=True)
df = sns.pairplot(df_ohlc, corner=True)
df_ohlc.corr()
sns.jointplot(x='High',y='Low',data=df_ohlc,kind='hex',color='red')
plt.show()
sns.jointplot(x='Open',y='Close',data=df_ohlc,kind='hex',color='red')
plt.show()
