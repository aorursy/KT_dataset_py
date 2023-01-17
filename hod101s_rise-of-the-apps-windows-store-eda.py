import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from collections import defaultdict

import plotly



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"
data = pd.read_csv('../input/windows-store/msft.csv')

data.head()
data.shape
data.info()
def NullUnique(df):

    dic = defaultdict(list)

    for col in df.columns:

        dic['Feature'].append(col)

        dic['NumUnique'].append(len(df[col].unique()))

        dic['NumNull'].append(df[col].isnull().sum())

        dic['%Null'].append(round(df[col].isnull().sum()/df.shape[0] * 100,2))

    return pd.DataFrame(dict(dic)).sort_values(['%Null'],ascending=False).style.background_gradient()
NullUnique(data)
data.iloc[-1]
data.drop(5321, axis=0, inplace = True)
fig = px.bar(x=data.nlargest(n=10, columns="Rating")["Name"],

             y=data.nlargest(n=10, columns="Rating")["Rating"], 

             color=data.nlargest(n=10, columns="Rating")["Name"].values,)

fig.update_xaxes(title="Ratings")

fig.update_yaxes(title="Names")

fig.update_layout(title= "Top rated apps", height = 600, width = 800, showlegend=False)

fig.show()
fig = px.bar(x=data.nsmallest(n=10, columns="Rating")["Name"],

             y=data.nsmallest(n=10, columns="Rating")["Rating"], 

             color=data.nlargest(n=10, columns="Rating")["Name"].values,)

fig.update_xaxes(title="Ratings")

fig.update_yaxes(title="Names")

fig.update_layout(title= "Lowest rated apps", height = 600, width = 800, showlegend=False)

fig.show()
data.Rating.hist()
data.sort_values(['No of people Rated'],ascending=False).iloc[:10][['Name','Rating','No of people Rated']].style.background_gradient()
fig = px.bar(x=data.groupby(['Category']).agg('count').Rating.index,y=data.groupby(['Category']).agg('count').Rating.values,color=data.groupby(['Category']).agg('count').Rating.values)

fig.update_layout(title='Most Popular Category by No of Apps')

fig.show()
fig = px.bar(x=data.groupby(['Category']).agg('mean').Rating.index,y=data.groupby(['Category']).agg('mean').Rating.values,color=data.groupby(['Category']).agg('mean').Rating.values)

fig.update_layout(title='Most Popular Category by Rating')

fig.show()
fig = px.bar(x=data.groupby(['Category']).agg('mean')['No of people Rated'].index,y=data.groupby(['Category']).agg('mean')['No of people Rated'].values,color=data.groupby(['Category']).agg('mean')['No of people Rated'].values)

fig.update_layout(title='Most Popular Category by No. of Rating')

fig.show()
fig = px.box(data,x='Rating',y='No of people Rated')

fig.update_layout(title = 'Distribution of No of Ratings Across Ratings')

fig.show()
data.Date = pd.to_datetime(data.Date)
fig = go.Figure(go.Scatter(

    x = data.groupby(['Date']).agg('count').Rating.index , 

    y =data.groupby(['Date']).agg('count').Rating.values,

    ))

fig.update_layout(title='Temporal Distribution App Uploads')

fig.show()
fig = go.Figure(go.Scatter(x=data.groupby("Date").agg({"Date": "count"}).sort_index()["Date"].cumsum().index,

    y=data.groupby("Date").agg({"Date": "count"}).sort_index()["Date"].cumsum()))

fig.update_layout(title='Rise of The Apps')

fig.show()
data['PriceCat'] = data.Price

data.PriceCat.loc[data.PriceCat != "Free"] = "Paid"

data.PriceCat.unique()
data.PriceCat.hist()
fig = px.scatter(

    x = data.Date , 

    y =data.index,

    color=data.PriceCat

    )

fig.update_layout(title='When did Paid Apps become a thing?')

fig.show()
data[data["Price"] == "Free"] = 0

data["Price"] = data["Price"].str.replace("₹ ", "")

data["Price"] = data["Price"].str.replace(",","")

data["Price"].fillna(0, inplace=True)

data["Price"] = data["Price"].astype(float)
fig = go.Figure([go.Bar(y=data.nlargest(10, columns="Price")["Price"].values, 

                     x=data.nlargest(10, columns="Price")["Name"], 

                     text=data.nlargest(10, columns="Price")["Price"].values,)])

fig.update_layout(title='Most Expensive Apps')

fig.show()
fig = go.Figure([go.Bar(y=data.query("Rating == 5").nsmallest(10, columns="Price")["Price"], 

                     x=data.query("Rating == 5").nsmallest(10, columns="Price")["Name"], 

                     text=data.query("Rating == 5").nsmallest(10, columns="Price")["Price"])])

fig.update_layout(title='Best Rated and Inexpensive Apps')

fig.show()       