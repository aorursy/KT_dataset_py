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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import plotly
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
import plotly.offline as pyo
from plotly.offline import init_notebook_mode, plot, iplot
import folium
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize']=10,12
df = pd.read_excel("/kaggle/input/corona-dataset/dataset/Covid cases in India.xlsx")
df
df.columns
df.describe()
df.isnull().sum()
df.drop(['S. No.'], axis=1, inplace=True)
df
df['Total Cases'] = df['Total Confirmed cases (Indian National)'] + df['Total Confirmed cases ( Foreign National )']
df
total_cases_overall = df['Total Cases'].sum()
print("The Total cases in India till 27 March are",total_cases_overall)
df['Active Cases'] = df['Total Cases'] - (df['Cured']+df['Death'])
df
total_active_overall = df['Active Cases'].sum()
print("Total Active Cases till 27 March are",total_active_overall)
df.style.background_gradient(cmap='Reds')
total_active_cases = df.groupby("Name of State / UT")['Active Cases'].sum().sort_values(ascending=False).to_frame()
total_active_cases
total_active_cases.style.background_gradient(cmap='Reds')
plt.rcParams['figure.figsize'] = 10,12
df.plot(kind="bar",x="Name of State / UT",y="Total Cases")
df.plot(kind="scatter",x="Name of State / UT",y="Total Cases")
plt.scatter(df["Name of State / UT"],df["Total Cases"])
fig = plt.figure(figsize=(20,10))
axes = fig.add_axes([0,0,1,1])
axes.bar(df["Name of State / UT"],df["Total Cases"])
axes.set_title("Covid 19 Cases in India")
axes.set_xlabel("Name of State/UT")
axes.set_ylabel("Total Cases")
coord = pd.read_excel("/kaggle/input/corona-dataset/dataset/Indian Coordinates.xlsx")
df_full = pd.merge(coord,df,on="Name of State / UT")
coord.head()
df_full.head()
f,ax = plt.subplots(figsize=(12,8))
data = df_full[['Name of State / UT',"Total Cases","Cured","Death"]]
data.sort_values('Total Cases',ascending=False,inplace=True)
sns.set_color_codes("pastel")
sns.barplot(x="Total Cases", y="Name of State / UT", data=data, label="Total",color="r")

sns.set_color_codes("muted")
sns.barplot(x="Cured", y="Name of State / UT", data=data, label="Cured",color="g")

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0,35), ylabel="", xlabel="Cases")
sns.despine(left=True, bottom=True)
#Global Analysis of Coronavirus
dbd_India = pd.read_excel("/kaggle/input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates=True, sheet_name="India")
dbd_SK = pd.read_excel("/kaggle/input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates=True, sheet_name="Korea")
dbd_Italy = pd.read_excel("/kaggle/input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates=True, sheet_name="Italy")
dbd_Wuhan = pd.read_excel("/kaggle/input/corona-dataset/dataset/per_day_cases.xlsx",parse_dates=True, sheet_name="Wuhan")
dbd_India.head()
dbd_SK.head()
dbd_Italy.head()
dbd_Wuhan.head()
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(dbd_India["Date"],dbd_India["Total Cases"],color='blue')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in India")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.scatter(dbd_India["Date"],dbd_India["Total Cases"],color='blue')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in India")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(dbd_Italy["Date"],dbd_Italy["Total Cases"],color='green')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in Italy")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.scatter(dbd_Italy["Date"],dbd_Italy["Total Cases"],color='green')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in Italy")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(dbd_SK["Date"],dbd_SK["Total Cases"],color='yellow')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in South Korea")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.scatter(dbd_SK["Date"],dbd_SK["Total Cases"],color='yellow')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in South Korea")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(dbd_Wuhan["Date"],dbd_Wuhan["Total Cases"],color='red')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in Wuhan")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.scatter(dbd_Wuhan["Date"],dbd_Wuhan["Total Cases"],color='red')
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in Wuhan")
#Changing Marker
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(dbd_India["Date"],dbd_India["Total Cases"],color='blue',marker="*")
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in India")
fig = plt.figure(figsize=(10,5),dpi=200)
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.scatter(dbd_India["Date"],dbd_India["Total Cases"],color='blue',marker="*")
axes.set_xlabel("Date")
axes.set_ylabel("Total Cases")
axes.set_title("COVID Cases in India")
fig = go.Figure()
fig.add_trace(go.Scatter(x=dbd_India['Date'],y=dbd_India['Total Cases'],mode='lines+markers'))
fig.update_layout(title_text="Trend of Coronvirus Cases in India(Cumulative Cases)",plot_bgcolor='rgb(230,230,230)')
#Daily Reported COVID Cases in India
fig = px.bar(dbd_India, x="Date", y="New Cases", barmode="group", height=400)
fig.update_layout(title_text="Coronavirus Cases in India on Daily Basis",plot_bgcolor="rgb(230,230,230)")
px.scatter(df, x="Name of State / UT",y="Total Cases")
fig = go.Figure()
fig.add_trace(go.Bar(x=df["Name of State / UT"], y=df["Active Cases"]))
fig.update_layout(title="COVID Cases in India",xaxis=dict(title='Name of State / UT'),yaxis=dict(title='Total Cases'))
fig = px.bar(dbd_India, x="Date", y="Total Cases", color="Total Cases", title="COVID in India")
fig.show()
fig = px.scatter(dbd_India, x="Date", y="Total Cases", color="Total Cases", title="COVID in India")
fig.show()
fig = px.bar(dbd_Italy, x="Date", y="Total Cases", color="Total Cases", title="COVID in Italy")
fig.show()
fig = px.scatter(dbd_Italy, x="Date", y="Total Cases", color="Total Cases", title="COVID in Italy")
fig.show()
fig = px.bar(dbd_SK, x="Date", y="Total Cases", color="Total Cases", title="COVID in South Korea")
fig.show()
fig = px.scatter(dbd_SK, x="Date", y="Total Cases", color="Total Cases", title="COVID in South Korea")
fig.show()
fig = px.bar(dbd_Wuhan, x="Date", y="Total Cases", color="Total Cases", title="COVID in Wuhan")
fig.show()
fig = px.scatter(dbd_Wuhan, x="Date", y="Total Cases", color="Total Cases", title="COVID in Wuhan")
fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(
    rows=2,cols=2,
    specs=[[{"secondary_y":True},{"secondary_y":True}],[{"secondary_y":True},{"secondary_y":True}]],
    subplot_titles=("S. Korea","Italy","India","Wuhan")
)

fig.add_trace(go.Bar(x=dbd_SK['Date'], y=dbd_SK['Total Cases'], marker=dict(color=dbd_SK['Total Cases'], coloraxis='coloraxis')),1,1)
fig.add_trace(go.Bar(x=dbd_Italy['Date'], y=dbd_Italy['Total Cases'], marker=dict(color=dbd_Italy['Total Cases'], coloraxis='coloraxis')),1,2)
fig.add_trace(go.Bar(x=dbd_India['Date'], y=dbd_India['Total Cases'], marker=dict(color=dbd_India['Total Cases'], coloraxis='coloraxis')),2,1)
fig.add_trace(go.Bar(x=dbd_Wuhan['Date'], y=dbd_Wuhan['Total Cases'], marker=dict(color=dbd_Wuhan['Total Cases'], coloraxis='coloraxis')),2,2)

fig.update_layout(coloraxis = dict(colorscale="Bluered_r"),showlegend=False,title_text="Total COVID Cases in 4 Countries")
fig.update_layout(plot_bgcolor = 'rgb(230,230,230)')
fig.show()
fig = make_subplots(
    rows=2,cols=2,
    specs=[[{"secondary_y":True},{"secondary_y":True}],[{"secondary_y":True},{"secondary_y":True}]],
    subplot_titles=("S. Korea","Italy","India","Wuhan")
)

fig.add_trace(go.Scatter(x=dbd_SK['Date'], y=dbd_SK['Total Cases'], marker=dict(color=dbd_SK['Total Cases'], coloraxis='coloraxis')),1,1)
fig.add_trace(go.Scatter(x=dbd_Italy['Date'], y=dbd_Italy['Total Cases'], marker=dict(color=dbd_Italy['Total Cases'], coloraxis='coloraxis')),1,2)
fig.add_trace(go.Scatter(x=dbd_India['Date'], y=dbd_India['Total Cases'], marker=dict(color=dbd_India['Total Cases'], coloraxis='coloraxis')),2,1)
fig.add_trace(go.Scatter(x=dbd_Wuhan['Date'], y=dbd_Wuhan['Total Cases'], marker=dict(color=dbd_Wuhan['Total Cases'], coloraxis='coloraxis')),2,2)

fig.update_layout(coloraxis = dict(colorscale="Bluered_r"),showlegend=False,title_text="Total COVID Cases in 4 Countries")
fig.update_layout(plot_bgcolor = 'rgb(230,230,230)')
df_confirmed = pd.read_csv("/kaggle/input/corona-dataset/dataset/time_series_covid_19_confirmed.csv")
df_confirmed
df_confirmed.rename(columns={'Country/Region':'Country'},inplace=True)
df_confirmed
df = pd.read_csv("/kaggle/input/corona-dataset/dataset/covid_19_data.csv")
df = pd.read_csv("/kaggle/input/corona-dataset/dataset/covid_19_data.csv",parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date','Country/Region':'Country'}, inplace=True)
df.groupby('Date').sum()
confirmed = df.groupby('Date').sum()['Confirmed'].to_frame().reset_index()
deaths = df.groupby('Date').sum()['Deaths'].to_frame().reset_index()
recover = df.groupby('Date').sum()['Recovered'].to_frame().reset_index()
confirmed.head()
deaths.head()
recover.head()
fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['Date'],y=confirmed['Confirmed'],mode='lines+markers',name='Confirmed',line=dict(color='blue')))
fig.add_trace(go.Scatter(x=deaths['Date'],y=deaths['Deaths'],mode='lines+markers',name='Deaths',line=dict(color='red')))
fig.add_trace(go.Scatter(x=recover['Date'],y=recover['Recovered'],mode='lines+markers',name='Recovered',line=dict(color='green')))
map = folium.Map(location=[20,70],zoom_start=5, tiles="CartoDB Dark_Matter")
for lat,long,value,name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Total Cases'], df_full['Name of State / UT']):
  folium.CircleMarker([lat,long], radius=value*0.3, popup=('<strong>State</strong>: '+str(name).capitalize()+'<br>'+'<strong>Total Cases</strong>'+str(value)+'<br>'),
  color = 'red', fill_color='red',fill_opacity=0.3).add_to(map)
map
#Trend after crossing 100 Cases
labels = ['S. Korea','Italy','India','Wuhan']
colors = ["rgb(122,128,0)","rgb(255,0,0)","rgb(49,130,189)","rgb(0,255,0)"]
mode_size = [10,10,12,12]
line_size = [1,1,8,8]

fig = go.Figure()

fig.add_trace(go.Scatter(x=dbd_SK['Days after surpassing 100 cases'], 
                         y=dbd_SK['Total Cases'],
                         mode='lines',
                         name=labels[0],
                         line=dict(color=colors[0], width=line_size[0]),
                         connectgaps=True))

fig.add_trace(go.Scatter(x=dbd_Italy['Days after surpassing 100 cases'], 
                         y=dbd_Italy['Total Cases'],
                         mode='lines',
                         name=labels[1],
                         line=dict(color=colors[1], width=line_size[1]),
                         connectgaps=True))

fig.add_trace(go.Scatter(x=dbd_India['Days after surpassing 100 cases'], 
                         y=dbd_India['Total Cases'],
                         mode='lines',
                         name=labels[2],
                         line=dict(color=colors[2], width=line_size[2]),
                         connectgaps=True))

annotations = []

annotations.append(dict(xref='paper',yref='paper',x=0.5,y=0.1,
                        xanchor="center",yanchor="top",
                        text="Days after crossing 100 cases",
                        font = dict(family="Arial",
                                    size=12,
                                    color="rgb(150,150,150)"),
                        showarrow=False
                        ))

fig.update_layout(annotations=annotations, plot_bgcolor="white",yaxis_title="Total Cases")
fig.show()
from fbprophet import Prophet
confirmed.columns
confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.head()
m = Prophet(interval_width=0.96)
m.fit(confirmed)
future = m.make_future_dataframe(periods=7)
future.head()
forecast = m.predict(future)
forecast
forecast_plot = m.plot(forecast)
components_plot = m.plot_components(forecast)
deaths.columns
deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])
deaths.head()
m = Prophet(interval_width=0.96)
m.fit(deaths)
future = m.make_future_dataframe(periods=7)
future.head()
forecast = m.predict(future)
forecast
forecast_plot = m.plot(forecast)
components_plot = m.plot_components(forecast)
recover.columns
recover.columns = ['ds','y']
recover['ds'] = pd.to_datetime(recover['ds'])
recover.head()
m = Prophet(interval_width=0.96)
m.fit(recover)
future = m.make_future_dataframe(periods=7)
future.head()
forecast = m.predict(future)
forecast
forecast_plot = m.plot(forecast)
components_plot = m.plot_components(forecast)