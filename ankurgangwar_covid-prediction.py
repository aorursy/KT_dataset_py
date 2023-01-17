# importing the required libraries

import pandas as pd



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium 

from folium import plugins



# Manipulating the default plot size

plt.rcParams['figure.figsize'] = 10, 12



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
#Learn how to read a .xls file by creating a dataframe using pandas

# Reading the datasets

df= pd.read_excel('/kaggle/input/Covid cases in India.xlsx')

df_india = df

df
# Coordinates of India States and Union Territories

India_coord = pd.read_excel('/kaggle/input/Indian Coordinates.xlsx')



#Day by day data of India, Korea, Italy and Wuhan

dbd_India = pd.read_excel('/kaggle/input/per_day_cases.xlsx',parse_dates=True, sheet_name='India')

dbd_Italy = pd.read_excel('/kaggle/input/per_day_cases.xlsx',parse_dates=True, sheet_name="Italy")

dbd_Korea = pd.read_excel('/kaggle/input/per_day_cases.xlsx',parse_dates=True, sheet_name="Korea")

dbd_Wuhan = pd.read_excel('/kaggle/input/per_day_cases.xlsx',parse_dates=True, sheet_name="Wuhan")
#Learn how to play around with the dataframe and create a new attribute of 'Total Case'

#Total case is the total number of confirmed cases (Indian National + Foreign National)



df.drop(['S. No.'],axis=1,inplace=True)

df['Total cases'] = df['Total Confirmed cases (Indian National)'] + df['Total Confirmed cases ( Foreign National )']

total_cases = df['Total cases'].sum()

print('Total number of confirmed COVID 2019 cases across India till date (22nd March, 2020):', total_cases)

#Learn how to highlight your dataframe

df.style.background_gradient(cmap='Reds')
#Total Active  is the Total cases - (Number of death + Cured)

df['Total Active'] = df['Total cases'] - (df['Death'] + df['Cured'])

total_active = df['Total Active'].sum()

print('Total number of active COVID 2019 cases across India:', total_active)

Tot_Cases = df.groupby('Name of State / UT')['Total Active'].sum().sort_values(ascending=False).to_frame()

Tot_Cases.style.background_gradient(cmap='Reds')
# Learn how to use folium to create a zoomable map

df_full = pd.merge(India_coord,df,on='Name of State / UT')

map = folium.Map(location=[20, 70], zoom_start=4,tiles='Stamenterrain')



for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Total cases'], df_full['Name of State / UT']):

    folium.CircleMarker([lat, lon], radius=value*0.8, popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>'),color='red',fill_color='red',fill_opacity=0.3 ).add_to(map)

map
#Learn how to use Seaborn for visualization

f, ax = plt.subplots(figsize=(12, 8))

data = df_full[['Name of State / UT','Total cases','Cured','Death']]

data.sort_values('Total cases',ascending=False,inplace=True)

sns.set_color_codes("pastel")

sns.barplot(x="Total cases", y="Name of State / UT", data=data,label="Total", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Cured", y="Name of State / UT", data=data, label="Cured", color="g")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 35), ylabel="",xlabel="Cases")

sns.despine(left=True, bottom=True)
#This cell's code is required when you are working with plotly on colab

#import plotly

#plotly.io.renderers.default = 'colab'
#Learn how to create interactive graphs using plotly

# import plotly.graph_objects as go

# Rise of COVID-19 cases in India

fig = go.Figure()

fig.add_trace(go.Scatter(x=dbd_India['Date'], y = dbd_India['Total Cases'], mode='lines+markers',name='Total Cases'))

fig.update_layout(title_text='Trend of Coronavirus Cases in India (Cumulative cases)',plot_bgcolor='rgb(230, 230, 230)')

fig.show()



# New COVID-19 cases reported daily in India



import plotly.express as px

fig = px.bar(dbd_India, x="Date", y="New Cases", barmode='group', height=400)

fig.update_layout(title_text='Coronavirus Cases in India on daily basis',plot_bgcolor='rgb(230, 230, 230)')



fig.show()
# import plotly.express as px

fig = px.bar(dbd_India, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in India', color_discrete_sequence = px.colors.cyclical.IceFire)



'''Colour Scale for plotly

https://plot.ly/python/builtin-colorscales/

'''



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()



fig = px.bar(dbd_Italy, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in Italy', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()



fig = px.bar(dbd_Korea, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in South Korea', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()

fig = px.bar(dbd_Wuhan, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in Wuhan', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()
#Learn how to create subplots using plotly

# import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(

    rows=2, cols=2,

    specs=[[{}, {}],

           [{"colspan": 2}, None]],

    subplot_titles=("S.Korea","Italy", "India","Wuhan"))



fig.add_trace(go.Bar(x=dbd_Korea['Date'], y=dbd_Korea['Total Cases'],

                    marker=dict(color=dbd_Korea['Total Cases'], coloraxis="coloraxis")),1, 1)



fig.add_trace(go.Bar(x=dbd_Italy['Date'], y=dbd_Italy['Total Cases'],

                    marker=dict(color=dbd_Italy['Total Cases'], coloraxis="coloraxis")),1, 2)



fig.add_trace(go.Bar(x=dbd_India['Date'], y=dbd_India['Total Cases'],

                    marker=dict(color=dbd_India['Total Cases'], coloraxis="coloraxis")),2, 1)



# fig.add_trace(go.Bar(x=dbd_Wuhan['Date'], y=dbd_Wuhan['Total Cases'],

#                     marker=dict(color=dbd_Wuhan['Total Cases'], coloraxis="coloraxis")),2, 2)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Total Confirmed cases(Cumulative)")



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()
# import plotly.graph_objects as go



title = 'Main Source for News'

labels = ['S.Korea', 'Italy', 'India']

colors = ['rgb(122,128,0)', 'rgb(255,0,0)', 'rgb(49,130,189)']



mode_size = [10, 10, 12]

line_size = [1, 1, 8]



fig = go.Figure()





fig.add_trace(go.Scatter(x=dbd_Korea['Days after surpassing 100 cases'], 

                 y=dbd_Korea['Total Cases'],mode='lines',

                 name=labels[0],

                 line=dict(color=colors[0], width=line_size[0]),            

                 connectgaps=True))

fig.add_trace(go.Scatter(x=dbd_Italy['Days after surpassing 100 cases'], 

                 y=dbd_Italy['Total Cases'],mode='lines',

                 name=labels[1],

                 line=dict(color=colors[1], width=line_size[1]),            

                 connectgaps=True))



fig.add_trace(go.Scatter(x=dbd_India['Days after surpassing 100 cases'], 

                 y=dbd_India['Total Cases'],mode='lines',

                 name=labels[2],

                 line=dict(color=colors[2], width=line_size[2]),            

                 connectgaps=True))

    

    

    

annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,

                              xanchor='center', yanchor='top',

                              text='Days after crossing 100 cases ',

                              font=dict(family='Arial',

                                        size=12,

                                        color='rgb(150,150,150)'),

                              showarrow=False))



fig.update_layout(annotations=annotations,plot_bgcolor='white',yaxis_title='Cumulative cases')



fig.show()
df = pd.read_csv('/kaggle/input/covid_19_clean_complete.csv',parse_dates=['Date'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



df_confirmed = pd.read_csv("/kaggle/input/time_series_covid19_confirmed_global.csv")

df_recovered = pd.read_csv("/kaggle/input/time_series_covid19_recovered_global.csv")

df_deaths = pd.read_csv("/kaggle/input/time_series_covid19_deaths_global.csv")



df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.head()
df.head()
df2 = df.groupby(["Date", "Country", "Province/State"])[['Date', 'Province/State', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

df2.head()
# Check for India's data

df.query('Country=="India"').groupby("Date")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
#Overall worldwide Confirmed/ Deaths/ Recovered cases 

df.groupby('Date').sum().head()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
fig = go.Figure()

#Plotting datewise confirmed cases

fig.add_trace(go.Scatter(x=confirmed['Date'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=deaths['Date'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))

fig.add_trace(go.Scatter(x=recovered['Date'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))

fig.update_layout(title='Worldwide NCOVID-19 Cases', xaxis_tickfont_size=14,yaxis=dict(title='Number of Cases'))



fig.show()
from fbprophet import Prophet
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=7)

future.tail()
#predicting the future with date, and upper and lower limit of y value

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
confirmed_forecast_plot =m.plot_components(forecast)
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)
deaths_forecast_plot = m.plot_components(forecast)
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet(interval_width=0.95)

m.fit(recovered)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
recovered_forecast_plot = m.plot(forecast)
recovered_forecast_plot = m.plot_components(forecast)
