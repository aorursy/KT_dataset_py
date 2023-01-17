# Osman Balli





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv")

data.head(10)
data.columns
data.drop(["Unnamed: 0","type"],axis=1,inplace=True)

print(data.columns)
data.shape
data.describe()
# checked null values

print(data.isnull().sum())
#Cleaned of Nan Value

data.drop("Rotten Tomatoes",axis=1,inplace=True)

data.dropna(inplace=True)

print(data.shape)
# Top 10 imdb

IMDB_ = data.sort_values("IMDb",  ascending = False).head(10)[["IMDb","Title","Age"]]

print(IMDB_)
# Grouped by genre

data_sum= data.sum()

print(data_sum["Netflix":"Disney+"])
## plot

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

trace = go.Pie(labels=["Netflix","Hulu","Prime Video","Disney+"], values=data_sum["Netflix":"Disney+"])

layout = go.Layout(title="", height=600, legend=dict(x=0.1, y=1.1))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
#correlation between features

#correlation map 

import seaborn as sns

corr_=data[["Netflix","Hulu","Prime Video","Disney+"]]

plt.figure(figsize=(9,9))

plt.title('Correlation Map')

ax=sns.heatmap(corr_.corr(),

               linewidth=1,

               annot=True,

               center=2)
# Grouped by Age

Age_ = data.groupby('Age').groups

print(Age_)

# Grouped by Age

Age_ = data.groupby('Age').sum()

Age_[["Netflix","Hulu","Prime Video","Disney+"]]
sns.catplot(y = "Age", kind = "count",

            palette = "pastel", edgecolor = ".6",

            data = data)


trace1 = go.Bar(x=Age_.axes[0], y=Age_["Netflix"], name="Netflix",width=[0.5,0.5,0.5,0.5,0.5])

x = [trace1]

layout = go.Layout(title="Count of Netflix content by Age ", legend=dict(x=0.1, y=0.1))

fig = go.Figure(x, layout=layout)

fig.update_traces(marker_color='rgb(102,216,23)', marker_line_color='rgb(87,96,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.show()
trace1 = go.Bar(x=Age_.axes[0], y=Age_["Hulu"], name="Hulu",width=[0.5,0.5,0.5,0.5,0.5])

x = [trace1]

layout = go.Layout(title="Count of Hulu content by Age ", legend=dict(x=0.1, y=0.1))

fig = go.Figure(x, layout=layout)

fig.show()

    
trace1 = go.Bar(x=Age_.axes[0], y=Age_["Prime Video"], name="Prime Video",width=[0.5,0.5,0.5,0.5,0.5])

x = [trace1]

layout = go.Layout(title="Count of Prime Video content by Age ", legend=dict(x=0.1, y=0.1))

fig = go.Figure(x, layout=layout)

fig.update_traces(marker_color='rgb(5,200,200)', marker_line_color='rgb(63,25,200)',

                  marker_line_width=1.5, opacity=0.6)

fig.show()
trace1 = go.Bar(x=Age_.axes[0], y=Age_["Disney+"], name="Disney+",width=[0.5,0.5,0.5,0.5,0.5])

x = [trace1]

layout = go.Layout(title="Count of Disney+ content by Age ", legend=dict(x=0.1, y=0.1))

fig = go.Figure(x, layout=layout)

fig.update_traces(marker_color='rgb(120,225,120)', marker_line_color='rgb(21,100,200)',

                  marker_line_width=1.5, opacity=0.6)

fig.show()
ax = sns.catplot(x='Year',kind='count',data=data,orient="h",height=30,aspect=1)

ax.fig.suptitle('Number of TV series / movies per year')

ax.fig.autofmt_xdate()
# Grouped by Year

Year_ = data.groupby('Year')

Year_sum = data.groupby('Year').sum()

Year_sum[["Netflix","Hulu","Prime Video","Disney+"]]
x=Year_sum.axes[0]

y=Year_sum["Netflix"]

plt.figure(figsize=(45,25))

plt.plot(x,y,linestyle='solid',label="count of TV shows/movies")

plt.xticks(x, x, rotation=75)

plt.ylabel("Count")

plt.xlabel("Years")

plt.legend()

plt.title("Count of Netflix TV shows/movies by Year")

plt.grid(color='r', linestyle='dotted', linewidth=0.5)

plt.show()
x=Year_sum.axes[0]

y=Year_sum["Hulu"]

plt.figure(figsize=(45,25))

plt.plot(x,y,linestyle='solid',label="count of TV shows/movies")

plt.xticks(x, x, rotation=75)

plt.ylabel("Count")

plt.xlabel("Years")

plt.legend()

plt.title("Count of Hulu TV shows/movies by Year")

plt.grid(color='r', linestyle='dotted', linewidth=0.5)

plt.show()
x=Year_sum.axes[0]

y=Year_sum["Prime Video"]

plt.figure(figsize=(45,25))

plt.plot(x,y,linestyle='solid',label="count of TV shows/movies")

plt.xticks(x, x, rotation=75)

plt.ylabel("Count")

plt.xlabel("Years")

plt.legend()

plt.title("Count of Prime Video TV shows/movies by Year")

plt.grid(color='r', linestyle='dotted', linewidth=0.5)

plt.show()
x=Year_sum.axes[0]

y=Year_sum["Disney+"]

plt.figure(figsize=(45,25))

plt.plot(x,y,linestyle='solid',label="count of TV shows/movies")

plt.xticks(x, x, rotation=75)

plt.ylabel("Count")

plt.xlabel("Years")

plt.legend()

plt.title("Count of Disney+ TV shows/movies by Year")

plt.grid(color='r', linestyle='dotted', linewidth=0.5)

plt.show()