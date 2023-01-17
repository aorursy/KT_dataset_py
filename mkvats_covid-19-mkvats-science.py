#@title
# importing the required libraries
import pandas as pd
import numpy as np
import datetime 
today = datetime.date.today()
yesterday = today - datetime.timedelta(days = 1)
world_file = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{:%m-%d-%Y}.csv'.format(yesterday)

# Visualisation libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly
#plotly.io.renderers.default = 'colab'
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import folium 
from folium import plugins

# Manipulating the default plot size
plt.rcParams['figure.figsize'] = 10, 12

# Disable warnings 
import warnings
warnings.filterwarnings('ignore')

# Google Drive
#from google.colab import auth
#auth.authenticate_user()
#import gspread
#from oauth2client.client import GoogleCredentials
#gc = gspread.authorize(GoogleCredentials.get_application_default())

# File reads
# df_india = pd.read_excel('data/Covid cases in India.xlsx')
# df_json = pd.read_json (r'https://raw.githubusercontent.com/covid19india/api/master/raw_data.json')
df_india = pd.read_excel('https://github.com/MkVats/COVID-19/blob/master/Covid%20-%20India.xlsx?raw=true', sheet_name='Statewise Total')
df_india_map = pd.read_excel('https://github.com/MkVats/COVID-19/blob/master/Covid%20-%20India.xlsx?raw=true', sheet_name='Cordinates')
df_all = pd.read_csv('https://raw.githubusercontent.com/MkVats/COVID-19/master/covid_19_data.csv')

df_world = pd.read_csv(world_file)
df_world_confirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
df_world_recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
df_world_deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")

def Country_Timeline(country):
  if country=='All':
    return df_all
  else:
    #df_world[df_world.Country_Region == "India"]
    return df_all.loc[df_all['Country/Region'] == country]

df_china_timeline = Country_Timeline("Mainland China")
df_italy_timeline = Country_Timeline("Italy")
df_spain_timeline = Country_Timeline("Spain")
df_iran_timeline = Country_Timeline("Iran")
df_skorea_timeline = Country_Timeline("South Korea")
df_pakistan_timeline = Country_Timeline("Pakistan")
df_us_timeline = Country_Timeline("US")
df_india_timeline = Country_Timeline("India")
df_russia_timeline = Country_Timeline("Russia")
#@title 
print('Latest Cases in India till date')
df_world.query('Country_Region=="India"').groupby("Last_Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df_india
#@title
df_india['Total cases'] = df_india['Total Confirmed cases (Indian National)'] + df_india['Total Confirmed cases ( Foreign National )']
total_cases = df_india['Total cases'].sum()
last_date = df_india.at[0,'Last_Updated_Time']
df_india['Total Active'] = df_india['Total cases'] - (df_india['Death'] + df_india['Cured/\nDischarged/Migrated'])
total_active = df_india['Total Active'].sum()
print('Total number of active COVID-19 cases in India till ', last_date, 'is', total_active)
print('Total number of confirmed COVID-19 cases in India till ', last_date, 'is', total_cases)
df_india.groupby('Name of State / UT')['Total Active'].sum().sort_values(ascending=False).to_frame()
#df_active = pd.merge(df_india,total_active,on='Name of State / UT')
#df_active.groupby('Name of State / UT')['Total Active'].sum().sort_values(ascending=False).to_frame()
df_india.style.background_gradient(cmap='Reds')
#@title 
df_map = pd.merge(df_india_map,df_india,on='Name of State / UT')
map = folium.Map(location=[20, 70], zoom_start=5,tiles='Stamenterrain')
for lat, lon, value, name in zip(df_map['Latitude'], df_map['Longitude'], df_map['Total cases'], df_map['Name of State / UT']):
    folium.CircleMarker([lat, lon], radius=value*0.01, popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>'),color='red',fill_color='red',fill_opacity=0.3 ).add_to(map)
map
#@title 
import IPython

IPython.display.HTML("<div class='tableauPlaceholder' id='viz1585145553118' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book1_31496&#47;Dashboard3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Book1_31496&#47;Dashboard3' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book1_31496&#47;Dashboard3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1585145553118');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")
#@title 
df_india.rename(columns={'Name of State / UT':'State','Total Confirmed cases (Indian National)':'ConfInd','Total Confirmed cases ( Foreign National )':'ConfFor'}, inplace=True)

def India_BarPlot_SNS():
  f, ax = plt.subplots(figsize=(20, 8))
  data = df_map[['Name of State / UT','Total cases','Cured/\nDischarged/Migrated','Death']]
  data.sort_values('Total cases',ascending=False,inplace=True)
  sns.set_color_codes()
  sns.barplot(x="Total cases", y="Name of State / UT", data=data,label="Total", color="b")

  sns.set_color_codes("muted")
  sns.barplot(x="Cured/\nDischarged/Migrated", y="Name of State / UT", data=data, label="Cured", color="g")

  sns.set_color_codes("pastel")
  sns.barplot(x="Death", y="Name of State / UT", data=data, label="Death", color="r")

  ax.legend(ncol=2, loc="lower right", frameon=True)
  ax.set(xlim=(0, 10000), ylabel="",xlabel="Cases")
  sns.despine(left=True, bottom=True)


def India_BarPlot_IndFor():
  plt.figure(figsize=(10,6))
  plt.title("Total Confirmed cases by State")
  sns.set_style(style="white")
  sns.barplot(df_india.State, df_india.ConfInd, color='red', label="Indian National")
  sns.barplot(df_india.State, df_india.ConfFor, color='blue', label="Foreign National",bottom=df_india.ConfInd)
  plt.legend()
  plt.xticks(rotation=90)
  plt.show()

def India_Donut_States():
  df_india["ConfTot"]=df_india.ConfInd+df_india.ConfFor
  x=df_india.ConfTot
  labels=df_india.State
  explode=np.zeros(df_india.shape[0],)
  explode=explode+0.1
  plt.figure(figsize=(12,12))
  plt.title("Total Confirmed cases  by State", fontsize=16)
  plt.pie(x, labels=labels, explode=explode,wedgeprops=dict(width=0.5),autopct='%1.1f%%', startangle=0, )
  plt.show()

def India_Pie_IndFor():
  x=[df_india.ConfInd.sum(),df_india.ConfFor.sum()]
  labels=["Indians","Foreign Nationals"]
  explode=[0.1,0.1]

  plt.figure(figsize=(8,8))
  plt.title("Total confirmed cases in India (Indians vs Foreigners)", fontsize=16)
  plt.pie(x, labels=labels, explode=explode,
          autopct='%1.1f%%')
  plt.legend()
  plt.show()

def India_Timeline_All():
  confirmed = df_india_timeline.groupby('ObservationDate').sum()['Confirmed'].reset_index()
  deaths = df_india_timeline.groupby('ObservationDate').sum()['Deaths'].reset_index()
  recovered = df_india_timeline.groupby('ObservationDate').sum()['Recovered'].reset_index()

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
  fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
  fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=(confirmed['Confirmed']-recovered['Recovered']-deaths['Deaths']), mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))
  fig.add_trace(go.Scatter(x=recovered['ObservationDate'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
  fig.update_layout(title='Timline of Cases in India<br>इन आंकड़ों के विश्लेषण से मेरे अनुसार भारत में कुल पुष्टि मामलों की संख्या:<br>30 मार्च तक 1250, 8-10 अप्रैल तक 6000-6500, 20 अप्रैल तक 15000-17000 एवं 30 अप्रैल तक 30000-32000 हो सकते है|', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))

  fig.show()

def India_Reports():
  India_BarPlot_SNS()
  India_BarPlot_IndFor()
  India_Donut_States()
  India_Pie_IndFor()



India_BarPlot_IndFor()
India_Pie_IndFor()
India_Donut_States()
India_BarPlot_SNS()
India_Timeline_All()
df_india_timeline.style.background_gradient(cmap='Reds')
df_world.head()
df_world_confirmed.head()
df_world_recovered.head()
df_world_recovered.head()
df_world.style.background_gradient(cmap='Reds')
#@title 
def World_Map(df_all, df_world_confirmed):
  df_world_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
  df_all.rename(columns={'Country/Region':'Country'}, inplace=True)
  df_world_confirmed = df_world_confirmed[["Province/State","Lat","Long","Country"]]
  df_temp = df_all.copy()
  df_temp['Country'].replace({'Mainland China': 'China'}, inplace=True)
  df_latlong = pd.merge(df_temp, df_world_confirmed, on=["Country", "Province/State"])

  fig = px.density_mapbox(df_latlong, lat="Lat", lon="Long", hover_name="Country", hover_data=["Confirmed","Deaths","Recovered"], animation_frame="ObservationDate", color_continuous_scale="Portland", radius=7, zoom=0,height=700)
  fig.update_layout(title='Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered')

  fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

  fig.show()

def World_Timeline_All():
  confirmed = df_all.groupby('ObservationDate').sum()['Confirmed'].reset_index()
  deaths = df_all.groupby('ObservationDate').sum()['Deaths'].reset_index()
  recovered = df_all.groupby('ObservationDate').sum()['Recovered'].reset_index()

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
  fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
  fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=(confirmed['Confirmed']-recovered['Recovered']-deaths['Deaths']), mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))
  fig.add_trace(go.Scatter(x=recovered['ObservationDate'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
  fig.update_layout(title='Timline of Cases in World', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))

  fig.show()

World_Map(df_all, df_world_confirmed)
World_Timeline_All()
#@title 
def Country_Bar_Time(df, Country):
  fig = px.bar(df.groupby('ObservationDate', as_index=False)['Confirmed'].sum(), x="ObservationDate", y="Confirmed", color='Confirmed', orientation='v', height=500, title='Confirmed Cases in '+Country, color_discrete_sequence = px.colors.cyclical.HSV)
  fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')
  fig.show()

Country_Bar_Time(df_china_timeline, 'China')
Country_Bar_Time(df_italy_timeline, 'Italy')
Country_Bar_Time(df_spain_timeline, 'Spain')
Country_Bar_Time(df_us_timeline, 'USA')
Country_Bar_Time(df_iran_timeline, 'Iran')
Country_Bar_Time(df_skorea_timeline, 'South Korea')
Country_Bar_Time(df_pakistan_timeline, 'Pakistan')
Country_Bar_Time(df_india_timeline, 'India')
Country_Bar_Time(df_russia_timeline, 'Russia')
#@title 
from plotly.subplots import make_subplots

def Country_Active_Cases(df):
  return df['Confirmed'] - df['Deaths'] - df['Recovered']


fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], subplot_titles=("South Korea","China","India"))

fig.add_trace(go.Scatter(x=df_skorea_timeline['ObservationDate'], y=Country_Active_Cases(df_skorea_timeline), marker=dict(color='Orange', coloraxis="coloraxis")), 1, 1)
#fig.add_trace(go.Scatter(x=df_skorea_timeline['ObservationDate'], y=df_skorea_timeline['Deaths'], marker=dict(color='Red', coloraxis="coloraxis")), 1, 1)
#fig.add_trace(go.Scatter(x=df_skorea_timeline['ObservationDate'], y=df_skorea_timeline['Confirmed'], marker=dict(color='Blue', coloraxis="coloraxis")), 1, 1)
#fig.add_trace(go.Scatter(x=df_skorea_timeline['ObservationDate'], y=df_skorea_timeline['Recovered'], marker=dict(color='Green', coloraxis="coloraxis")), 1, 1)

fig.add_trace(go.Scatter(x=df_china_timeline['ObservationDate'], y=Country_Active_Cases(df_china_timeline), marker=dict(color=df_china_timeline['Confirmed'], coloraxis="coloraxis")), 1, 2)

fig.add_trace(go.Scatter(x=df_india_timeline['ObservationDate'], y=Country_Active_Cases(df_india_timeline), marker=dict(color=df_india_timeline['Confirmed'], coloraxis="coloraxis")), 2, 1)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Active Cases")
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.show()
#@title 
confirmed = df_china_timeline.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = df_china_timeline.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = df_china_timeline.groupby('ObservationDate').sum()['Recovered'].reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=(confirmed['Confirmed']-recovered['Recovered']-deaths['Deaths']), mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))
fig.add_trace(go.Scatter(x=recovered['ObservationDate'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
fig.update_layout(title='Timline of Cases in China', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))

fig.show()

confirmed = df_skorea_timeline.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = df_skorea_timeline.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = df_skorea_timeline.groupby('ObservationDate').sum()['Recovered'].reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=(confirmed['Confirmed']-recovered['Recovered']-deaths['Deaths']), mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))
fig.add_trace(go.Scatter(x=recovered['ObservationDate'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
fig.update_layout(title='Timline of Cases in South Korea', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))

fig.show()
#@title 
confirmed = df_italy_timeline.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = df_italy_timeline.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = df_italy_timeline.groupby('ObservationDate').sum()['Recovered'].reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=(confirmed['Confirmed']-recovered['Recovered']-deaths['Deaths']), mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))
fig.add_trace(go.Scatter(x=recovered['ObservationDate'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
fig.update_layout(title='Timline of Cases in Italy', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))

fig.show()

confirmed = df_spain_timeline.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = df_spain_timeline.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = df_spain_timeline.groupby('ObservationDate').sum()['Recovered'].reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=(confirmed['Confirmed']-recovered['Recovered']-deaths['Deaths']), mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))
fig.add_trace(go.Scatter(x=recovered['ObservationDate'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
fig.update_layout(title='Timline of Cases in Spain', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))

fig.show()
#@title 
confirmed = df_us_timeline.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = df_us_timeline.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = df_us_timeline.groupby('ObservationDate').sum()['Recovered'].reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['ObservationDate'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'], y=(confirmed['Confirmed']-recovered['Recovered']-deaths['Deaths']), mode='lines+markers', name='Active', line=dict(color='Orange', width=2)))
fig.add_trace(go.Scatter(x=recovered['ObservationDate'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
fig.update_layout(title='Timline of Cases in USA', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))

fig.show()
#@title 
from fbprophet import Prophet
confirmed = df_all.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = df_all.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = df_all.groupby('ObservationDate').sum()['Recovered'].reset_index()

confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()
#@title 
mk_confirm = Prophet(interval_width=1)
mk_confirm.fit(confirmed)
future = mk_confirm.make_future_dataframe(periods=7)
future.tail()
forecast = mk_confirm.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = mk_confirm.plot(forecast)
confirmed_forecast_plot =mk_confirm.plot_components(forecast)
#@title 
deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])
#mk_death = Prophet(interval_width=0.95)
mk_death = Prophet(interval_width=1)
mk_death.fit(deaths)
future = mk_death.make_future_dataframe(periods=7)
future.tail()
forecast = mk_death.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = mk_death.plot(forecast)
deaths_forecast_plot = mk_death.plot_components(forecast)
#@title 
recovered.columns = ['ds','y']
recovered['ds'] = pd.to_datetime(recovered['ds'])
#mk_recover = Prophet(interval_width=0.95)
mk_recover = Prophet(interval_width=1)
mk_recover.fit(recovered)
future = mk_recover.make_future_dataframe(periods=7)
future.tail()
forecast = mk_recover.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
recovered_forecast_plot = mk_recover.plot(forecast)
recovered_forecast_plot = mk_recover.plot_components(forecast)
#@title 
confirmed = df_india_timeline.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = df_india_timeline.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = df_india_timeline.groupby('ObservationDate').sum()['Recovered'].reset_index()

confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()
#@title 
#mk_india_confirm = Prophet(interval_width=0.95)
mk_india_confirm = Prophet(interval_width=0.95)
mk_india_confirm.fit(confirmed)
future = mk_india_confirm.make_future_dataframe(periods=5)
future.tail()
forecast = mk_india_confirm.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
IPython.display.HTML('<iframe src="https://mkvats.science/COVID-19/Stats" frameborder="0" width="640" height="370"></iframe>')