# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import plotly.graph_objs as go 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/videogamesales/vgsales.csv")
df.info()
df.shape
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1, keys=['Total','Percent'])
missing_data.head()
df_num = df.select_dtypes(include=['float64','int64']).columns
df_col = df.select_dtypes(include= ['object']).columns
df_q = df[df_col]
df_q
print("Num of unique val:")
print("Total = {}".format(len(df)))
for col in df_q:
  nb_uniq= len(df[col].unique())
  print("{} = {}  ".format(col,nb_uniq))
df_num = df.select_dtypes(include=['float64','int64']).columns
plt.figure(figsize=(25,10))
sns.barplot("Year","Global_Sales",data=df)
plt.show()
plt.figure(figsize=(25,10))
sns.countplot('Year',data=df)
plt.show()

plt.figure(figsize=(25,10))
sns.countplot('Genre',data=df)
plt.show()
plt.figure(figsize=(25,10))
sns.barplot("Genre","Global_Sales",data=df)
plt.show()
plt.figure(figsize=(25,10))
sns.barplot("Genre","NA_Sales",data=df)
plt.show()
plt.figure(figsize=(25,10))
sns.barplot("Genre","EU_Sales",data=df)
plt.show()
plt.figure(figsize=(25,10))
sns.barplot("Genre","JP_Sales",data=df)
plt.show()
plt.figure(figsize=(20,20))
sns.barplot("Platform","Global_Sales",data=df)
plt.show()
plat1 = pd.crosstab(df.Platform,df.Genre)
plat1
group = df.groupby(df.Year)[["Global_Sales"]].sum()
group = group.sort_values(by= "Global_Sales", ascending=False)
group.head(8)

grouped = df.groupby(df.Year)[["Global_Sales"]].sum()
grouped = grouped.sort_values(by = "Global_Sales" , ascending = False)
grouped = grouped.head(10)


# plottng 

fig = px.pie(data_frame = grouped , 
            names = grouped.index , 
            values = "Global_Sales" , 
            template = "seaborn" , 
            hole = 0.4 , 
            color_discrete_sequence = px.colors.sequential.Inferno , 
            )

fig.update_layout(title = "Top 10 Year of sales", 
                  
                 plot_bgcolor = "rgb(230,230,230)" , 
                 annotations= [dict(text = "Global Sales" , font_size = 20 , showarrow = False , opacity = 0.7)])

fig.update_traces (rotation = 90 , pull = 0.03 , textinfo = "percent+label")
fig.show()

grouped = df.groupby(df.Platform)[["Global_Sales"]].sum()
grouped = grouped.sort_values(by = "Global_Sales" , ascending = False)
grouped = grouped.head(10)

# plottng 

fig = px.pie(data_frame = grouped , 
            names = grouped.index , 
            values = "Global_Sales" , 
            template = "seaborn" , 
            hole = 0.4 , 
            color_discrete_sequence = px.colors.sequential.Inferno , 
            )

fig.update_layout(title = "Top 10 Platforms over the years", 
                  
                 plot_bgcolor = "rgb(230,230,230)" , 
                 annotations= [dict(text = "Global Sales" , font_size = 20 , showarrow = False , opacity = 0.7)])

fig.update_traces (rotation = 90 , pull = 0.03 , textinfo = "percent+label")
fig.show()
grouped.head(10)
df_top=df.head(100)
df_top
plt.figure(figsize=(40,40))
from plotly.offline import init_notebook_mode,iplot
traceNa = go.Scatter(
                    x = df_top.Rank,
                    y = df_top.NA_Sales,
                    mode = "markers",
                    name = "North America",
                    marker = dict(color = 'rgba(30, 19, 239, 0.8)',size=6),
                    text= df.Name)

traceEu = go.Scatter(
                    x = df_top.Rank,
                    y = df_top.EU_Sales,
                    mode = "markers",
                    name = "Europe",
                    marker = dict(color = 'rgba(249, 94, 28, 0.8)',size=6),
                    text= df.Name)
traceJp = go.Scatter(
                    x = df_top.Rank,
                    y = df_top.JP_Sales,
                    mode = "markers",
                    name = "Japan",
                    marker = dict(color = 'rgba(10, 226, 180, 0.8)',size=6),
                    text= df.Name)
traceO = go.Scatter(
                    x = df_top.Rank,
                    y = df_top.Other_Sales,
                    mode = "markers",
                    name = "Other",
                    marker = dict(color = 'pink',size=6),
                    text= df.Name)
                    

data = [traceNa, traceEu,traceJp,traceO]


layout= dict(title='Top 100 Sales across different Regions',
                   xaxis_title='Rank',
                   yaxis_title='Sales_Region')
fig = dict(data = data,layout = layout)
iplot(fig)
df2 = df.loc[:,["Year","Platform","NA_Sales","EU_Sales" ]]
df2
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()
#scatterplot
sns.set()
cols = ['Rank', 'Year','NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
sns.pairplot(df_num[cols], size = 2.5)
plt.show();