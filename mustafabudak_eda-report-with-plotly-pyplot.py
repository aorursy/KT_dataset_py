# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.graph_objects as go  #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING

import plotly.express as px

import folium

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
data.head()
data = data.rename(columns={"Province/State":"State",

                   "Country/Region":"Country"})

data = data.drop(["SNo","Last Update"],1)

data.ObservationDate = pd.to_datetime(data.ObservationDate)
data_to_date = data[data.ObservationDate == max(data.ObservationDate)].reset_index()

data_to_date["Active"] = data_to_date["Confirmed"] - data_to_date["Recovered"]- data_to_date["Deaths"]

data_to_date.head()
confirmed = confirmed.rename(columns={"Province/State":"State",

                             "Country/Region":"Country"})

confirmed.head()
confirmed = pd.melt(confirmed, id_vars = ['State', 'Country', 'Lat', 'Long'], var_name = 'Date', value_name = 'Confirmed')

confirmed.head()
deaths = deaths.rename(columns={"Province/State":"State",

                             "Country/Region":"Country"})



deaths = pd.melt(deaths, id_vars = ['State', 'Country', 'Lat', 'Long'], var_name = 'Date', value_name = 'Deaths')

deaths.head()
recovered = recovered.rename(columns={"Province/State":"State",

                             "Country/Region":"Country"})



recovered = pd.melt(recovered, id_vars = ['State', 'Country', 'Lat', 'Long'], var_name = 'Date', value_name = 'Recovered')
df = pd.merge(confirmed, deaths, on=["State","Country","Lat","Long","Date"])

df.head()
df = df.replace(np.NaN,"")

data = data.replace(np.NaN,"")

data["ObservationDate"] = pd.to_datetime(data["ObservationDate"])

df["Date"] = pd.to_datetime(df["Date"])

df.head()
grouped = df.groupby('Date')['Confirmed', 'Deaths'].sum().reset_index()

grouped = grouped[grouped["Date"]==max(grouped["Date"])].reset_index(drop=True)

grouped.style.background_gradient(cmap="hot")
data_grouped = data_to_date.groupby("Country")["Confirmed","Deaths","Recovered","Active"].sum().reset_index().sort_values(by='Deaths', ascending=False).reset_index(drop=True)[:50]

data_grouped["DeathRate"] = data_grouped["Deaths"] / data_grouped["Confirmed"]

data_grouped.style.background_gradient(cmap='Reds')
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data.groupby("ObservationDate")["Confirmed"].sum().index,

            y=data.groupby("ObservationDate")["Confirmed"].sum().values,mode='lines+markers'

    ))



fig.add_trace(

    go.Line(name="Deaths",

        x=data.groupby("ObservationDate")["Deaths"].sum().index,

        y=data.groupby("ObservationDate")["Deaths"].sum().values,mode='markers'

    ))



fig.add_trace(

    go.Line(name="Recovered",

        x=data.groupby("ObservationDate")["Recovered"].sum().index,

        y=data.groupby("ObservationDate")["Recovered"].sum().values,mode='markers'

    ))



fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Confirmed, Deaths and Recovered by Date")

fig.show()
df_to_date = df[df.Date == max(df.Date)].reset_index()

df_grouped = df_to_date.groupby("Country")["Confirmed","Deaths"].sum().reset_index().sort_values(by='Deaths', ascending=False).reset_index(drop=True)



m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=2, max_zoom=4, zoom_start=2)



for i in range(0, len(df_to_date)):

    folium.Circle(

        location=[df_to_date.iloc[i]['Lat'], df_to_date.iloc[i]['Long']],

        color='crimson', 

        tooltip =   '<li><bold>Country : '+str(df_to_date.iloc[i]['Country'])+

                    '<li><bold>State : '+str(df_to_date.iloc[i]['State'])+

                    '<li><bold>Confirmed : '+str(df_to_date.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(df_to_date.iloc[i]['Deaths']),

                    

        radius=int(df_to_date.iloc[i]['Confirmed'])).add_to(m)

m
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data_grouped.sort_values(by="DeathRate", ascending=False)["Country"],

        y=data_grouped.sort_values(by="DeathRate", ascending=False)["DeathRate"],mode='lines+markers',

        marker=dict(line=dict(color='white',width=5))

    ))

fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Death rate by country",

                  xaxis =  {'showgrid': False},yaxis = {'showgrid': True})

fig.show()
n_con = list(data_grouped.head(15).sort_values('Confirmed', ascending=False)["Confirmed"])

n_recovered = list(data_grouped.head(15).sort_values('Deaths', ascending=False)["Deaths"])



#plt.style.use('fivethirtyeight')

plt.style.use('dark_background')

plt.figure(figsize=(16,9))



sns.barplot(x="Confirmed", y="Country",palette="autumn",data=data_grouped.head(15).sort_values('Confirmed', ascending=False),label="Confirmed")

sns.barplot(x="Recovered", y="Country",palette="twilight",data=data_grouped.head(15).sort_values('Recovered', ascending=False), label="Recovered")

plt.xticks(rotation= 45)

plt.xlabel('Confirmed/Recovered')

plt.ylabel('Country')

plt.grid(False)

for i, v in enumerate(n_con):

    plt.text(v + 3, i + .25, str(v), color='white', fontweight='bold')

    

for i, v in enumerate(n_recovered):

    plt.text(v + 3, i + .25, str(v), color='cyan', fontweight='bold')

    

plt.legend()    

plt.title('Confirmed and Recovered by Country ')

plt.show()
plt.style.use('dark_background')

plt.figure(figsize=(16,9))



sns.barplot(x="Deaths", y="Country",palette="plasma",data=data_grouped.groupby("Country")["Deaths"].sum().reset_index().sort_values(by="Deaths",ascending=False).head(20))

plt.xticks(rotation= 45)

plt.xlabel('Deaths')

plt.ylabel('Country')

plt.grid(False)

for i, v in enumerate(list(data_grouped.groupby("Country")["Deaths"].sum().reset_index().sort_values(by="Deaths",ascending=False).head(20)["Deaths"])):

    plt.text(v + 3, i + .25, str(v), color='white', fontweight='bold')

plt.title('Deaths by Country ')

plt.show()
plt.style.use('dark_background')

plt.figure(figsize=(16,9))



sns.barplot(x="Active", y="Country",palette="plasma",data=data_grouped.groupby("Country")["Active"].sum().reset_index().sort_values(by="Active",ascending=False).head(20))

plt.xticks(rotation= 45)

plt.xlabel('Active')

plt.ylabel('Country')

plt.grid(False)

for i, v in enumerate(list(data_grouped.groupby("Country")["Active"].sum().reset_index().sort_values(by="Active",ascending=True).head(20)["Active"])):

    plt.text(v + 3, i + .25, str(v), color='white', fontweight='bold')

plt.title('Active by Country ')

plt.show()
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data[data.Country=="US"].groupby("ObservationDate")["Confirmed"].sum().index,

        y=data[data.Country=="US"].groupby("ObservationDate")["Confirmed"].sum().values,mode='lines+markers',

         marker=dict(line=dict(color="green",

                     width=5)

    )))



fig.add_trace(

    go.Line(name="Deaths",

        x=data[data.Country=="US"].groupby("ObservationDate")["Deaths"].sum().index,

        y=data[data.Country=="US"].groupby("ObservationDate")["Deaths"].sum().values,mode='markers',

         marker=dict(line=dict(color="red",

                     width=4)

    )))



fig.add_trace(

    go.Line(name="Recovered",

        x=data[data.Country=="US"].groupby("ObservationDate")["Recovered"].sum().index,

        y=data[data.Country=="US"].groupby("ObservationDate")["Recovered"].sum().values,mode='markers',

         marker=dict(line=dict(color="white",

                     width=3)

    )))

fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Confirmed, Deaths and Recovered by Date of US",

                  xaxis =  {'showgrid': False},yaxis = {'showgrid': True})

fig.show()
usa = data_to_date[data_to_date.Country=="US"].sort_values(by="Confirmed",ascending=False).head(20)



fig = go.Figure()



fig.add_trace(

    go.Bar(name="Confirmed",

        x=usa["State"],

            y=usa["Confirmed"],

            offsetgroup=0,

            marker={'color': usa["Confirmed"],'colorscale': 'Viridis'}

    ))



fig.add_trace(

    go.Scatter(name="Deaths",

        x=usa["State"],

        y=usa["Deaths"]

    ))

fig.add_trace(

    go.Bar(name="Active",

        x=usa["State"],

        y=usa["Active"],

        offsetgroup=0

    ))





fig.update_layout(title_text="Confirmed and Deaths for USA States (TOP 20 for most Confirmed)", width=800, height=500, plot_bgcolor='rgb(10,10,10)')

fig.show()
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data[data.Country=="Italy"].groupby("ObservationDate")["Confirmed"].sum().index,

        y=data[data.Country=="Italy"].groupby("ObservationDate")["Confirmed"].sum().values,mode='lines+markers',

         marker=dict(line=dict(color="green",

                     width=5)

    )))



fig.add_trace(

    go.Line(name="Deaths",

        x=data[data.Country=="Italy"].groupby("ObservationDate")["Deaths"].sum().index,

        y=data[data.Country=="Italy"].groupby("ObservationDate")["Deaths"].sum().values,mode='markers',

         marker=dict(line=dict(color="red",

                     width=4))

    ))



fig.add_trace(

    go.Line(name="Recovered",

        x=data[data.Country=="Italy"].groupby("ObservationDate")["Recovered"].sum().index,

        y=data[data.Country=="Italy"].groupby("ObservationDate")["Recovered"].sum().values,mode='markers',

         marker=dict(line=dict(color="white",

                     width=3))

    ))



fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Confirmed, Deaths and Recovered by Date of Italy",

                 xaxis =  {'showgrid': False},yaxis = {'showgrid': True})

fig.show()
italy = data_to_date[data_to_date.Country=="Italy"].sort_values(by="Confirmed",ascending=False).head(20)



fig = go.Figure()



fig.add_trace(

    go.Bar(name="Confirmed",

        x=italy["State"],

            y=italy["Confirmed"],

            offsetgroup=0,

            marker={'color': italy["Confirmed"],'colorscale': 'Viridis'}

    ))



fig.add_trace(

    go.Scatter(name="Deaths",

        x=italy["State"],

        y=italy["Deaths"]

    ))



fig.add_trace(

    go.Bar(name="Active",

        x=italy["State"],

        y=italy["Active"],

        offsetgroup=0

    ))



fig.update_layout(title_text="Confirmed, Deaths and Active for Italy States (TOP 20 for most Confirmed)", width=800, height=500,  plot_bgcolor='rgb(10,10,10)')

fig.show()
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data[data.Country=="Mainland China"].groupby("ObservationDate")["Confirmed"].sum().index,

        y=data[data.Country=="Mainland China"].groupby("ObservationDate")["Confirmed"].sum().values,mode='lines+markers',

         marker=dict(line=dict(color="green",

                     width=5))

    ))



fig.add_trace(

    go.Line(name="Deaths",

        x=data[data.Country=="Mainland China"].groupby("ObservationDate")["Deaths"].sum().index,

        y=data[data.Country=="Mainland China"].groupby("ObservationDate")["Deaths"].sum().values,mode='markers',

         marker=dict(line=dict(color="red",

                     width=4))

    ))



fig.add_trace(

    go.Line(name="Recovered",

        x=data[data.Country=="Mainland China"].groupby("ObservationDate")["Recovered"].sum().index,

        y=data[data.Country=="Mainland China"].groupby("ObservationDate")["Recovered"].sum().values,mode='markers',

         marker=dict(line=dict(color="white",

                     width=3))

    ))



fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Confirmed, Deaths and Recovered by Date of China")

fig.show()
china = data_to_date[data_to_date.Country=="Mainland China"].sort_values(by="Confirmed",ascending=False).head(20)



fig = go.Figure()



fig.add_trace(

    go.Bar(name="Confirmed",

        x=china["State"],

            y=china["Confirmed"],

            offsetgroup=0,

            marker={'color': china["Confirmed"],'colorscale': 'Viridis'}

    ))



fig.add_trace(

    go.Scatter(name="Deaths",

        x=china["State"],

        y=china["Deaths"]

    ))





fig.update_layout(title_text="Confirmed and Deaths for China States (TOP 20 for most Confirmed)", width=800, height=500, plot_bgcolor='rgb(10,10,10)')

fig.show()
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data[data.Country=="Germany"].groupby("ObservationDate")["Confirmed"].sum().index,

        y=data[data.Country=="Germany"].groupby("ObservationDate")["Confirmed"].sum().values,mode='lines+markers',

         marker=dict(line=dict(color="green",

                     width=5))

    ))



fig.add_trace(

    go.Line(name="Deaths",

        x=data[data.Country=="Germany"].groupby("ObservationDate")["Deaths"].sum().index,

        y=data[data.Country=="Germany"].groupby("ObservationDate")["Deaths"].sum().values,mode='markers',

         marker=dict(line=dict(color="red",

                     width=4))

    ))



fig.add_trace(

    go.Line(name="Recovered",

        x=data[data.Country=="Germany"].groupby("ObservationDate")["Recovered"].sum().index,

        y=data[data.Country=="Germany"].groupby("ObservationDate")["Recovered"].sum().values,mode='markers',

         marker=dict(line=dict(color="white",

                     width=3))

    ))



fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Confirmed, Deaths and Recovered by Date of Germany")

fig.show()
germany = data_to_date[data_to_date.Country=="Germany"].sort_values(by="Confirmed",ascending=False).head(20)



fig = go.Figure()



fig.add_trace(

    go.Bar(name="Confirmed",

        x=germany["State"],

            y=germany["Confirmed"],

            offsetgroup=0,

            marker={'color': china["Confirmed"],'colorscale': 'Viridis'}

    ))



fig.add_trace(

    go.Scatter(name="Deaths",

        x=germany["State"],

        y=germany["Deaths"]

    ))



fig.add_trace(

    go.Bar(name="Active",

        x=germany["State"],

        y=germany["Active"],offsetgroup=0

    ))







fig.update_layout(title_text="Confirmed, Deaths and Active for Germany States (TOP 20 for most Confirmed)", width=800, height=500, plot_bgcolor='rgb(10,10,10)')

fig.show()
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data[data.Country=="Turkey"].groupby("ObservationDate")["Confirmed"].sum().index,

        y=data[data.Country=="Turkey"].groupby("ObservationDate")["Confirmed"].sum().values,mode='lines+markers',

         marker=dict(line=dict(color="green",

                     width=5)

    )))



fig.add_trace(

    go.Line(name="Deaths",

        x=data[data.Country=="Turkey"].groupby("ObservationDate")["Deaths"].sum().index,

        y=data[data.Country=="Turkey"].groupby("ObservationDate")["Deaths"].sum().values,mode='markers',

        marker=dict(line=dict(color='red',width=5))

    ))



fig.add_trace(

    go.Line(name="Recovered",

        x=data[data.Country=="Turkey"].groupby("ObservationDate")["Recovered"].sum().index,

        y=data[data.Country=="Turkey"].groupby("ObservationDate")["Recovered"].sum().values,mode='markers',

        marker=dict(line=dict(color='white',width=5))

    ))





fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Confirmed, Deaths and Recovered by Date of Turkey")

                

fig.show()
brazil = data_to_date[data_to_date.Country=="Brazil"].sort_values(by="Confirmed",ascending=False).head(20)



fig = go.Figure()



fig.add_trace(

    go.Bar(name="Confirmed",

        x=brazil["State"],

            y=brazil["Confirmed"],

            offsetgroup=0,

            marker={'color': china["Confirmed"],'colorscale': 'Viridis'}

    ))



fig.add_trace(

    go.Scatter(name="Deaths",

        x=brazil["State"],

        y=brazil["Deaths"]

    ))



fig.add_trace(

    go.Bar(name="Active",

        x=brazil["State"],

        y=brazil["Active"],offsetgroup=0

    ))







fig.update_layout(title_text="Confirmed, Deaths and Active for Brazil States (TOP 20 for most Confirmed)", width=800, height=500, plot_bgcolor='rgb(10,10,10)')

fig.show()
fig = go.Figure()



fig.add_trace(

    go.Line(name="Confirmed",

        x=data[data.Country=="Brazil"].groupby("ObservationDate")["Confirmed"].sum().index,

        y=data[data.Country=="Brazil"].groupby("ObservationDate")["Confirmed"].sum().values,mode='lines+markers',

         marker=dict(line=dict(color="green",

                     width=5)

    )))



fig.add_trace(

    go.Line(name="Deaths",

        x=data[data.Country=="Brazil"].groupby("ObservationDate")["Deaths"].sum().index,

        y=data[data.Country=="Brazil"].groupby("ObservationDate")["Deaths"].sum().values,mode='markers',

        marker=dict(line=dict(color='red',width=4))

    ))



fig.add_trace(

    go.Line(name="Recovered",

        x=data[data.Country=="Brazil"].groupby("ObservationDate")["Recovered"].sum().index,

        y=data[data.Country=="Brazil"].groupby("ObservationDate")["Recovered"].sum().values,mode='markers',

        marker=dict(line=dict(color='white',width=3))

    ))





fig.update_layout(width=800, height=500, plot_bgcolor='rgb(10,10,10)', title_text="Confirmed, Deaths and Recovered by Dates of Brazil")

                

fig.show()