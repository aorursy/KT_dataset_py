import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

from plotly.offline import plot as plt

import datetime as dt

import warnings

warnings.filterwarnings('ignore')
def diff_data(df,country_list)->bool:

    # initialization

    tmp = []

    for i in range(len(df)):

        tmp.append(0)

    df["new_Deaths_per_day"] = tmp #all zero

    

    for country in country_list:

        flag = 0

        for i in range(len(df)): #index

            if(df.at[i,"Country"] == country):

                if(flag != 0):

                    df.at[i,'new_Deaths_per_day'] = df.at[i,'Deaths'] - yesterday_Deaths

                else:

                    flag = 1

                yesterday_Deaths = df.at[i,"Deaths"]
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#character

data.drop(['SNo','Last Update','Province/State'],axis = 1, inplace = True) 

#Date

data['ObservationDate'] = pd.to_datetime(data["ObservationDate"])

data = data[data['ObservationDate'] >= dt.datetime(2020,3,1) ]

data = data[data['ObservationDate'] <= dt.datetime(2020,4,30)]

data['ObservationDate'] = data['ObservationDate'].dt.strftime('%y-%m-%d')

#rename

data.columns = ['Date','Country',"Confirmed","Deaths","Recovered"]

#sum Country

data = data.groupby(["Date","Country"],as_index = False).sum()

# Top 30 "Counfired" cumsum

country = data.groupby("Country",as_index=False).sum()

country_list = country.sort_values('Confirmed',ascending = False ).head(30)["Country"].values

data = data[data["Country"].isin(country_list)]

data.reset_index(drop=True,inplace=True)

#new_data

diff_data(data,country_list)

data['Deaths_rate(%)']  = data['Deaths'] / data["Confirmed"] * 100

data.tail()
fig = px.bar(data,x = 'Country',y = 'Confirmed',color = "Recovered",animation_frame = 'Date',title="Confirmed in each Countries")

fig.show()
fig = px.bar(data,x = 'Country',y = 'new_Deaths_per_day',color = "Deaths_rate(%)",animation_frame = 'Date',title="new_Deaths in each Countries")

fig.show()
data_fig = data.sort_values("Confirmed",ascending = False)

fig = px.area(data_fig, x="Date", y="Confirmed", color="Country",title="World_Confirmed")

fig.show()
df = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")

df = df[["Country (or dependency)","Population (2020)","Density (P/Km²)","Urban Pop %"]]

#rename

df.columns=["Country","Population","Density (P/Km²)","Urban_rate(%)"]

#replace

df.replace('China','Mainland China',inplace=True)

df.replace('United States', 'US',inplace=True)

df.replace('United Kingdom', 'UK',inplace=True)



df = df[df["Country"].isin(country_list)]

df.reset_index(drop=True,inplace = True)

df.sort_values("Country",ascending = True, inplace = True)

data_Apr_30 = pd.merge(data.tail(30).reset_index(drop=True), df, on = "Country")

data_Apr_30["Confirmed_rate(%)"] = data_Apr_30["Confirmed"] / data_Apr_30["Population"] * 100

data_Apr_30.sort_values("Confirmed_rate(%)",ascending = False,inplace = True)

data_Apr_30.reset_index(drop=True)
fig = px.bar(data_Apr_30, x = "Country", y = "Confirmed_rate(%)",color = "Density (P/Km²)",title = "Confirmed_rate(%) in each country")

fig.show()