#Osman Balli



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

df.head()

df.shape
df.columns
#Count of Null Value

print(df.isnull().sum())
ax=sns.catplot(y = "Platform", kind = "count",

            palette = "pastel", edgecolor = ".10",

            data = df)

ax = sns.catplot(x='Genre',kind='count',data=df,orient="h",height=10,aspect=1)

ax.fig.suptitle('Count of Games per Genre',fontsize=16,color="r")

ax.fig.autofmt_xdate()
Genre_= df.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()

print(Genre_)
import plotly.graph_objs as go

import plotly.offline as pyoff



plot_data = [

    go.Bar(x=Genre_.axes[0], y=Genre_["NA_Sales"], name='NA_Sales',width=0.2,marker_color='rgb(220,20,60)'),

    go.Bar(x=Genre_.axes[0], y=Genre_["EU_Sales"], name='EU_Sales',width=0.2,marker_color='rgb(102,216,23)'),

    go.Bar(x=Genre_.axes[0], y=Genre_["JP_Sales"], name='JP_Sales',width=0.2,marker_color='rgb(5,200,200)'),

    go.Bar(x=Genre_.axes[0], y=Genre_["Other_Sales"], name='Other_Sales',width=0.2,marker_color='rgb(123,104,238)'),

]

plot_layout = go.Layout(

        title='Total Sales by Genre',

        yaxis_title='Sales',

        xaxis_title='Genre'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
import numpy as np

MaxSaleGame=df[df["Global_Sales"]==df["Global_Sales"].max()]["Name"]

list=["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]

list_value=[]

for i in list:

    MaxGlobalSale=df[df["Global_Sales"]==df["Global_Sales"].max()][i]

    list_value.append(MaxGlobalSale[0])

list_value
## plot

title='sales percentages of the game with maximum sales -----> '+MaxSaleGame[0]

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

trace = go.Pie(labels=list, values=list_value, pull=[0.02, 0.02, 0.02, 0.02])

layout = go.Layout(title=title, height=600, legend=dict(x=0.7, y=1))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
# Top 50 sales

Top50 = df.sort_values("Global_Sales").tail(20)[[ 'Name', 'Platform', 'Year', 'Genre', 'Publisher','Global_Sales']]

a=Top50[['Name','Global_Sales']]
import plotly.express as px

fig = px.bar(a, x="Global_Sales", y="Name", 

               text='Name', orientation='h', color_discrete_sequence = ['#a3de83'])

fig.update_layout(title="Top 20 Sales",height=550)

uniq=df["Genre"].unique()

total_Genre = []

for i in uniq:

    total_Genre.append(len(df[df["Genre"]==i]))

import plotly.graph_objects as go

from plotly.subplots import make_subplots



labels = uniq



# Create subplots: use 'domain' type for Pie subplot



fig.add_trace(go.Pie(labels=labels, values=total_Genre, name="Genre"))



fig = go.Figure(data=[go.Pie(labels=labels, values=total_Genre, hole=.3)])

fig.update_layout(

    title_text="Total Game Count in Genre",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Genre', x=0.5, y=0.5, font_size=20, showarrow=False)])

fig.show()
#Remove of Null Value in Publisher

Publisher=df.iloc[df["Publisher"].dropna().index]

Publisher.isnull().sum()
uniq=Publisher["Publisher"].unique()

total_Publisher = []

for i in uniq:

    total_Publisher.append(len(df[df["Publisher"]==i]))

total_Publisher
plot_data = go.Scatter(

    x=uniq,

    y=total_Publisher,

    mode='markers',

)



plot_layout = go.Layout(

        title="Total Game Count in Publisher",

        yaxis_title='Count',

        xaxis_title='Publisher'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
#Remove of Null Value in Year

Year=df.iloc[df["Year"].dropna().index]

Year.isnull().sum()
global_sales_year = Year.groupby("Year")["Global_Sales"].sum()

global_sales_year
import plotly.graph_objects as go



fig = go.Figure(go.Scatter(

    y=global_sales_year,

    x=global_sales_year.axes[0],name='Total Global Sales',mode='lines+markers'))



fig.update_layout(

    title={

        'text': "Total Global Sales in Years",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()