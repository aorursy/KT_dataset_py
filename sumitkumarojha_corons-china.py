import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv')
data.head()
data.info()
data.describe()
data.head()
data.rename(columns={'Date': 'date', 
                     'Id': 'id',
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                    }, inplace=True)
data.head()
data["state"]= data["state"].fillna('Unknown')
data["Recovered"]=data["Recovered"].fillna(0)
data["Deaths"]=data["Deaths"].fillna(0)

data["Confirmed"]=data["Confirmed"].fillna(0)


data[["Confirmed","Deaths","Recovered"]] =data[["Confirmed","Deaths","Recovered"]].astype(int) 
Data_per_country1 = data.groupby(["country"])["Deaths"].sum().reset_index().sort_values("Deaths",ascending=False).reset_index(drop=True)
Data_per_country2 = data.groupby(["country"])["Confirmed"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)
Data_per_country3 = data.groupby(["country"])["Recovered"].sum().reset_index().sort_values("Recovered",ascending=False).reset_index(drop=True)
Data_per_country1

Data_per_country2
Data_per_country3.sum()
list=[Data_per_country1,Data_per_country2, Data_per_country3  ]
import matplotlib.pyplot as plt

Data_world = data.groupby(["country"])["Confirmed","Recovered","Deaths"].sum().reset_index()
Data_world
import plotly.express as px
fig = px.choropleth(Data_per_country2, locations=Data_per_country2['country'],
                    color=Data_per_country2['Confirmed'],locationmode='country names', # lifeExp is a column of gapminder
                    hover_name=Data_per_country2['country'], # column to add to hover information
                    color_continuous_scale=px.colors.sequential.deep)
fig.update_layout(
    title='Confirmed Cases In The World',
)
fig.show()
