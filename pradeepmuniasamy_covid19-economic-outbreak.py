import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

import plotly.graph_objects as go

py.init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gold_df = pd.read_csv("/kaggle/input/economycovid19/gold price.csv")

gold_df.describe()
fig = px.line(gold_df, x='date', y=' value' , title="Price of Gold over the period of 2007 - 2020")

fig.show()
gold_df["date"] = pd.to_datetime(gold_df['date'])  

mask = (gold_df['date'] > '2019-12-01') 

temp_df = gold_df.loc[mask]

train_df= pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
temp = train_df.loc[(train_df["Country_Region"]=="US")].groupby(["Date"])["ConfirmedCases"].sum().reset_index()

fig = px.bar(temp, x='Date', y='ConfirmedCases',

             hover_data=['ConfirmedCases'], color='ConfirmedCases',

             labels={'pop':'Total Number of confirmed Cases'}, height=400)

fig.update_layout(

    title_text="Corona Outbreak in USA")

fig.show()
trace1 = go.Scatter(

    x=temp_df["date"],

    y=temp_df[" value"],

    name = "Price of Gold in USA"

)

trace2 = go.Bar(

    x=temp["Date"],

    y=temp["ConfirmedCases"],

    xaxis="x2",

    yaxis="y2",

    name = "Confirmed Cases in USA"

)



data = [trace1, trace2]



layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    yaxis2=dict(

        anchor="x2"

    )

)



fig = go.Figure(data=data, layout=layout)

fig.show()
sp_historic = pd.read_csv("/kaggle/input/economycovid19/S  P 500 - Sheet1.csv")

sp_lm = pd.read_csv("/kaggle/input/monthly/SP Monthly - Sheet1.csv")

nasdaq_historic = pd.read_csv("/kaggle/input/economycovid19/Nasdaq 100 - Sheet1.csv")

nasdaq_lm = pd.read_csv("/kaggle/input/monthly/Nasdaq Monthly - Sheet1.csv")
fig = px.line(sp_historic[::-1], x='Date', y='Price' , title="SP stock prices over Mar'13 - Mar'20")

fig.show()
sp_lm = sp_lm[3:]

trace1 = go.Scatter(

    x=sp_lm["Date"],

    y=sp_lm["Price"],

    name = "S&P"

)

trace2 = go.Scatter(

    x=nasdaq_lm[::-1]["Date"],

    y=nasdaq_lm[::-1]["Price"],

    xaxis="x2",

    yaxis="y2",

    name = "NASDAQ"

)

data = [trace1, trace2]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    yaxis2=dict(

        anchor="x2"

    )

)

fig = go.Figure(data=data, layout=layout)

fig.show()
crude_oil = pd.read_csv("/kaggle/input/economycovid19/Crude Oil Prices  - Sheet1.csv")



fig = px.line(crude_oil[:20][::-1], x='Date', y='Price' , title="Crude oil Prices")

fig.show()
Country=pd.DataFrame()

#temp = train_df.groupby(["Country/Region"])["ConfirmedCases"].sum().reset_index()

temp = train_df.loc[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country_Region'])["ConfirmedCases"].sum().reset_index()

Country['Name']=temp["Country_Region"]

Country['Values']=temp["ConfirmedCases"]



fig = px.choropleth(Country, locations='Name',

                    locationmode='country names',

                    color="Values")

fig.update_layout(title="Corona spread on 04-04-2020")

fig.show()
global_indices = pd.read_csv("/kaggle/input/economycovid19/Global Indices Performance - Sheet1.csv")

global_indices["Country"] = global_indices["Name"].str.split(' ').str[1]

global_indices



fig = px.choropleth(global_indices, locations='Country',

                    locationmode='country names',

                    color="1 Month")

fig.update_layout(title="Change in the indices for the last month")

fig.show()