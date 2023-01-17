

import numpy as np 

import pandas as pd 

import plotly.graph_objects as go

from workalendar.europe import Turkey





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df= pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
df.head()
df.Last_Update.dtype
df["Last_Update"]= df["Last_Update"].astype("datetime64")
df.Last_Update.dtype
df.isnull().sum()
df.drop("Province/State", axis=1, inplace=True)
df.head()
import plotly.express as px



fig = px.line(df, x='Last_Update', y='Confirmed')

fig.show()
confirmedarray=np.array(df.Confirmed)
liste= [1,]

for i in range(0,161):

    x= i+1

    y=x-1

    artıs=confirmedarray[x]- confirmedarray[y]

    liste.append(artıs)
array =np.array(liste)

array
dfold=df.copy()
df.drop("Confirmed",axis=1,inplace=True)
df["Confirmed"]= array
df.head()
deathsarray=np.array(df.Deaths)
liste= [0,]

for i in range(0,161):

    x= i+1

    y=x-1

    artıs=deathsarray[x]- deathsarray[y]

    liste.append(artıs)
array =np.array(liste)

array
df.drop("Deaths",axis=1,inplace=True)
df["Deaths"]= array
df.head()
recoversarray=np.array(df.Recovered)
liste= [0,]

for i in range(0,161):

    x= i+1

    y=x-1

    artıs=recoversarray[x]- recoversarray[y]

    liste.append(artıs)
array =np.array(liste)

array
df.drop("Recovered",axis=1,inplace=True)
df["Recovered"]= array
df.head()
df = df.reindex(columns=["Country/Region","Confirmed","Deaths","Recovered","Last_Update"])
df.head()
cal = Turkey()

cal.holidays(2020)
fig = px.line(df, x='Last_Update', y=['Confirmed','Recovered'],             

             title="Case and Recovered per Day",

             labels={"variable": "Variable",  "value": "Increase", "Last_Update": "Date"},

             color_discrete_map={"Recovered": "seagreen", "Confirmed": "black"})

fig.update_xaxes(rangeslider_visible=True)

          

fig.show()

fig = px.bar(df, 

             x='Last_Update', 

             y=['Confirmed','Recovered'],

             title="Case and Recovered per Day",

             labels={"variable": "Variable",  "value": "Increase", "Last_Update": "Date"},

             color_discrete_map={"Recovered": "seagreen", "Confirmed": "black"})



fig.update_xaxes(rangeslider_visible=True)

fig.show()

fig = px.line(df, x='Last_Update', y='Deaths')

fig.update_xaxes(rangeslider_visible=True)

          

fig.show()

df["Deaths"][49] = 89
df["Deaths"][49] == 89
df["Deaths"][50] = 93
df["Deaths"][50] == 93
df.to_csv('out.csv', index=False)
fig = px.line(df, x='Last_Update', y='Deaths', 

              title="Deaths per Day",             

              labels={"Last_Update": "Date"},

              color_discrete_map={"Deaths":"black"})

fig.update_xaxes(rangeslider_visible=True)

          

fig.show()

fig = px.bar(df, 

             x='Last_Update', 

             y='Deaths',

             title="Deaths per Day",

             labels={"Last_Update": "Date"},

             color_discrete_map={"Deaths":"black"})

fig.update_xaxes(rangeslider_visible=True)

fig.show()

df["ratecr"]= (df["Deaths"] / df["Confirmed"])*100 
df.head(5)
df= df.fillna(0) 
fig = px.line(df, 

             x='Last_Update', 

             y='ratecr',

             title="% Deaths to Confirmed ",

             labels={"Last_Update": "Date", "ratecr":"%Death to Confirmed"},

             color_discrete_map={"Deaths":"black"})

fig.update_xaxes(rangeslider_visible=True)

fig.show()

df["ratecontorec"]= (df["Recovered"] / df["Confirmed"])*100 
df= df.fillna(0) 
df.head()
fig = px.line(df, 

             x='Last_Update', 

             y='ratecontorec',

             title="% Recovered to Confirmed ",

             labels={"Last_Update": "Date", "ratecontorec":"%Recovered to Confirmed"},

             color_discrete_map={"ratecontorec":"black"})

fig.update_xaxes(rangeslider_visible=True)

fig.show()
