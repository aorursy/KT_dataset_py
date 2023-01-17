! pip install wget



# Data Processing

import numpy as np

import pandas as pd

import wget

from datetime import datetime, timedelta



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

import folium

%matplotlib inline



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# html embedding

from IPython.display import Javascript

from IPython.core.display import display, HTML



# set formatting

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



print("Setup Complete")
yesterday = datetime.today() - timedelta(days=1)

yesterday = yesterday.strftime('%m-%d-%Y')
yesterday
# url of the raw csv dataset

urls = [

    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',

    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',

    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',

    f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{yesterday}.csv'

]

[wget.download(url) for url in urls]
confirmed_df = pd.read_csv(r'time_series_covid19_confirmed_global.csv')

death_df = pd.read_csv(r'time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv(r'time_series_covid19_recovered_global.csv')

df = pd.read_csv(f'{yesterday}.csv')
df.head()
confirmed_df.head()
dates = confirmed_df.columns[4:]
confirmed_df_long = confirmed_df.melt(

    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

    value_vars=dates, 

    var_name='Date', 

    value_name='Confirmed'

)



death_df_long = death_df.melt(

    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

    value_vars=dates, 

    var_name='Date', 

    value_name='Deaths'

)



recovered_df_long = recovered_df.melt(

    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

    value_vars=dates, 

    var_name='Date', 

    value_name='Recovered'

)
recovered_df_long
confirmed_df_long
recovered_df_long = recovered_df_long[recovered_df_long['Country/Region']!='Canada']
# Merging confirmed_df_long and death_df_long

full_table = confirmed_df_long.merge(

  right=death_df_long, 

  how='left',

  on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']

)
# Merging full_table and recovered_df_long

full_table = full_table.merge(

  right=recovered_df_long, 

  how='left',

  on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']

)
full_table.head()
full_table['Date'] = pd.to_datetime(full_table['Date'])
full_table.isna().sum()
full_table['Recovered'] = full_table['Recovered'].fillna(0).astype(int)
ship_rows = full_table['Province/State'].str.contains('Grand Princess') | full_table['Province/State'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('MS Zaandam')
ship_df = full_table[ship_rows]

ship_df.head()
full_table = full_table[~(ship_rows)]
full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']
full_table.head()
full_grouped = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
# new cases 

temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



# renaming columns

temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']



# merging new values

full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])



# filling na with 0

full_grouped = full_grouped.fillna(0)



# fixing data types

cols = ['New cases', 'New deaths', 'New recovered']

full_grouped[cols] = full_grouped[cols].astype('int')



# 

full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)
full_grouped.head()
temp = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

world_cases = temp.to_numpy()

temp.style.background_gradient(cmap='Pastel1')
temp = full_grouped.groupby(['Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'].max()

temp.sort_values('Confirmed', ascending=False).style.bar(subset=['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'],

                                                         align = 'left', color='#d65f5f')
temp = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Confirmed', 'Deaths', 'Recovered', 'Active'],

                 var_name='Case', value_name='Count')



fig = px.line(temp, x="Date", y="Count", color='Case',

             title='Cases over time')

fig
fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Cases over time')

fig
fig = px.bar(temp, x="Date", y="Count", color='Case',

             title='Cases over time')

fig
temp = temp[~(temp.Case == 'Confirmed')]

fig = px.pie(temp, values='Count', names='Case', title= f'Confirmed Cases: {world_cases.item(1)}')

fig.update_traces(textinfo='percent+value+label')

fig.show()
temp = full_grouped

# adding two more columns

temp['Mortality Rate'] = round(temp['Deaths']/temp['Confirmed'], 3)

temp['Recovery Rate'] = round(temp['Recovered']/temp['Confirmed'], 3)



temp.groupby(['Country/Region'])['Mortality Rate', 'Recovery Rate' ].max().sort_values('Recovery Rate', ascending=False).style.background_gradient(cmap='Reds')
temp = full_grouped.groupby('Date').sum().reset_index()



temp['Mortality Rate'] = round(temp['Deaths']/temp['Confirmed'], 3)

temp['Recovery Rate'] = round(temp['Recovered']/temp['Confirmed'], 3)



temp = temp.melt(id_vars='Date', value_vars=['Mortality Rate', 'Recovery Rate'], 

                 var_name='Ratio', value_name='Value')



fig = px.line(temp, x="Date", y="Value", color='Ratio', log_y=True, 

              title='Recovery and Mortality Rate Over The Time')

fig
fig = px.bar(temp, x="Date", y="Value", color='Ratio', log_y=True, 

              title='Recovery and Mortality Rate Over The Time')

fig
temp = full_grouped.groupby('Date')['New cases', 'New deaths', 'New recovered'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['New cases', 'New deaths', 'New recovered'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.line(temp, x="Date", y="Count", color='Case',

             title='Daily Cases')

fig
fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Daily Cases')

fig
fig = px.bar(temp, x="Date", y="Count", color='Case',

             title='Daily Cases')

fig
fig = px.pie(temp, values='Count', names='Case', title='Confirmed Cases')

fig.update_traces(textinfo='percent+value+label')

fig.show()
temp = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Confirmed', 'Deaths', 'Recovered', 'Active'],

                 var_name='Case', value_name='Count')

temp['Log Count'] = np.log(temp['Count'])



fig = px.line(temp, x="Date", y="Log Count", color='Case',

             title='Log of Cases over time')

fig
fig = px.area(temp, x="Date", y="Log Count", color='Case',

             title='Log of Cases over time')

fig
fig = px.bar(temp, x="Date", y="Log Count", color='Case',

             title='Log of Cases over time')

fig
fig = px.pie(temp, values='Log Count', names='Case', title='Confirmed Cases')

fig.update_traces(textinfo='percent+label')

fig.show()
country_grouped = df.groupby('Country_Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
country_grouped['Active'] = country_grouped['Active'].astype(int)
country_grouped = country_grouped.sort_values('Confirmed', ascending=False)
others_series = pd.Series(np.sum(country_grouped[10:]))

country_grouped_others = country_grouped[:10]

country_grouped_others = country_grouped_others.append(others_series, ignore_index=True)

country_grouped_others.iloc[10,0] = 'Rest of the World'
# Confirmed Cases

fig = px.choropleth(country_grouped, locations="Country_Region", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country_Region", range_color=[1,700000], 

                    color_continuous_scale="aggrnyl", 

                    title='Countries with Confirmed Cases')

fig
fig = px.pie(country_grouped_others, values='Confirmed', names='Country_Region', title='Confirmed Cases')

fig.update_traces(textinfo='percent+label')

fig.show()
fig = px.bar(country_grouped.head(20).sort_values('Confirmed', ascending=True), 

             x="Confirmed", y="Country_Region",title='Confirmed Cases Top 20 Countries', 

             text='Confirmed', orientation='h', 

             width=700, height=700)

fig.update_traces(opacity=0.6)

fig
# Deaths

fig = px.choropleth(country_grouped[country_grouped['Deaths']>0], 

                    locations="Country_Region", locationmode='country names',

                    color="Deaths", hover_name="Country_Region", 

                    range_color=[1,50000], color_continuous_scale="agsunset",

                    title='Countries with Deaths Reported')

fig
fig = px.pie(country_grouped_others.sort_values('Deaths', ascending=False), values='Deaths', names='Country_Region', title='Total Deaths')

fig.update_traces(textinfo='percent+label')

fig.show()
fig = px.bar(country_grouped.sort_values('Deaths', ascending=False).head(20).sort_values('Deaths', ascending=True), 

             x="Deaths", y="Country_Region", title='Total Deaths Top 20 Countries', text='Deaths', orientation='h', 

             width=700, height=700)

fig.update_traces(opacity=0.6)

fig
# Recoveris

fig = px.choropleth(country_grouped[country_grouped['Recovered']>0], 

                    locations="Country_Region", locationmode='country names',

                    color="Recovered", hover_name="Country_Region", 

                    range_color=[1,50000], color_continuous_scale="agsunset",

                    title='Countries Recovered Cases')

fig
fig = px.pie(country_grouped_others.sort_values('Recovered', ascending=False), values='Recovered', names='Country_Region', title='Total Recovered')

fig.update_traces(textinfo='percent+label')

fig.show()
fig = px.bar(country_grouped.sort_values('Recovered', ascending=False).head(20).sort_values('Recovered', ascending=True), 

             x="Recovered", y="Country_Region", title='Total Recovered Top 20 Countries', text='Recovered', orientation='h', 

             width=700, height=700)

fig.update_traces(opacity=0.6)

fig
# Active

fig = px.choropleth(country_grouped[country_grouped['Active']>0], 

                    locations="Country_Region", locationmode='country names',

                    color="Active", hover_name="Country_Region", 

                    range_color=[1,50000], color_continuous_scale="agsunset",

                    title='Countries Active Cases')

fig
fig = px.pie(country_grouped_others.sort_values('Active', ascending=False), values='Active', names='Country_Region', title='Total Active Cases')

fig.update_traces(textinfo='percent+label')

fig.show()
fig = px.bar(country_grouped.sort_values('Active', ascending=False).head(20).sort_values('Active', ascending=True), 

             x="Active", y="Country_Region", title='Top 20 Countries Active Cases', text='Active', orientation='h', 

             width=700, height=700)

fig.update_traces(opacity=0.6)

fig
import altair as alt



top_countries = country_grouped['Country_Region'].head(10)

top_countries_data = full_grouped[full_grouped['Country/Region'].isin(top_countries)]



interval = alt.selection_interval()





circle = alt.Chart(top_countries_data).mark_circle().encode(

    x = 'monthdate(Date):O',

    y = 'Country/Region',

    color = alt.condition(interval, 'Country/Region', alt.value('lightgray')),

    size = alt.Size('New cases:Q',

            scale = alt.Scale(range = [0, 1000]),

            legend = alt.Legend(title = 'Daily new cases')

    )

).properties(

    width = 500,

    height = 300,

    selection = interval

)



bars = alt.Chart(top_countries_data).mark_bar().encode(

    y = 'Country/Region',

    color = 'Country/Region',

    x = 'sum(New cases):Q'

).properties(

    width = 500

).transform_filter(

    interval

)



circle & bars
# https://app.flourish.studio/visualisation/1571387/edit

HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
def cdr_graph(df, region):

    """

    Input:

        df of type :

            * Date: datetime64[ns]

            * Country/Region: object

            * Confirmed: int64

            * Deaths: int64

            * Recovered: int64

            * Active: int64

            * New cases: int64

            * New deaths: int64

            * New recovered: int64

            * Mortality Rate: float64

            * Recovery Rate: float64<br>

            dtype: object

        region of df: String

    return:

        line, area, bar Graphs of ['Confirmed', 'Deaths', 'Recovered', 'Active'] cases 



    """

    temp = df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

    temp = temp.melt(id_vars="Date", value_vars=['Confirmed', 'Deaths', 'Recovered', 'Active'],

                 var_name='Case', value_name='Count')



    line = px.line(temp, x="Date", y="Count", color='Case',

             title=f'{region} Cases over time')

    area = px.area(temp, x="Date", y="Count", color='Case',

             title=f'{region} Cases over time')

    bar = px.bar(temp, x="Date", y="Count", color='Case',

             title=f'{region} Cases over time')

    temp = temp[~(temp.Case == 'Confirmed')]

    pie = px.pie(temp, values='Count', names='Case', title='Confirmed Cases')

    pie.update_traces(textinfo='percent+label')

    

    display(line)

    display(area)

    display(bar)

    display(pie)

def mr_graph(df, region):

    """

    Input:

        df of type :

            * Date: datetime64[ns]

            * Country/Region: object

            * Confirmed: int64

            * Deaths: int64

            * Recovered: int64

            * Active: int64

            * New cases: int64

            * New deaths: int64

            * New recovered: int64

            * Mortality Rate: float64

            * Recovery Rate: float64<br>

            dtype: object

        region of df: String

    return:

        line, bar Graphs of ['Mortality Rate', 'Recovery Rate'] cases 



    """

    temp = df.groupby('Date').sum().reset_index()



    temp['Mortality Rate'] = round(temp['Deaths']/temp['Confirmed'], 3)

    temp['Recovery Rate'] = round(temp['Recovered']/temp['Confirmed'], 3)



    temp = temp.melt(id_vars='Date', value_vars=['Mortality Rate', 'Recovery Rate'], 

                     var_name='Ratio', value_name='Value')



    line = px.line(temp, x="Date", y="Value", color='Ratio', log_y=True, 

                  title=f'Recovery and Mortality Rate of {region} Over The Time')

    bar = px.bar(temp, x="Date", y="Value", color='Ratio', 

                 title=f'Recovery and Mortality of {region} Rate Over The Time')

    display(line)

    display(bar)

def daily_graph(df, region):

    """

    Input:

        df of type :

            * Date: datetime64[ns]

            * Country/Region: object

            * Confirmed: int64

            * Deaths: int64

            * Recovered: int64

            * Active: int64

            * New cases: int64

            * New deaths: int64

            * New recovered: int64

            * Mortality Rate: float64

            * Recovery Rate: float64<br>

            dtype: object

        region of df: String

    return:

        line, area, bar Graphs of ['New cases', 'New deaths', 'New recovered'] cases 



    """

    temp = df.groupby('Date')['New cases', 'New deaths', 'New recovered'].sum().reset_index()

    temp = temp.melt(id_vars="Date", value_vars=['New cases', 'New deaths', 'New recovered'],

                     var_name='Case', value_name='Count')



    line = px.line(temp, x="Date", y="Count", color='Case', title=f'{region} Daily Cases')

    area = px.area(temp, x="Date", y="Count", color='Case', title=f'{region} Daily Cases')

    bar = px.bar(temp, x="Date", y="Count", color='Case', title=f'{region} Daily Cases')

    pie = px.pie(temp, values='Count', names='Case', title='Confirmed Cases')

    pie.update_traces(textinfo='percent+label')

    

    display(line)

    display(area)

    display(bar)

    display(pie)
india_data = full_grouped[full_grouped['Country/Region'] == 'India']
HTML('<img src = "https://upload.wikimedia.org/wikipedia/commons/9/95/COVID-19_India_Total_Cases_Animated_Map.gif" height = "700", width = "500">')
cdr_graph(india_data, 'India')
mr_graph(india_data, 'India')
daily_graph(india_data, 'India')
# import Packages for Prediction

!pip install pmdarima

from pmdarima.arima import auto_arima

from sklearn.metrics import mean_squared_error

from datetime import timedelta

from fbprophet import Prophet
# change Date column name

full_grouped = full_grouped.rename(columns = {'Date': 'ds'})



# Group data

df_group = full_grouped.groupby(by = 'ds')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()



# change index to datetime

df_group.index = pd.to_datetime(df_group.index)



# Set frequncy of time series

df_group = df_group.asfreq(freq = '1D')



# Sort the values

df_group = df_group.sort_index(ascending = True)



# Fill NA values with zero

df_group = df_group.fillna(value = 0)



df_group = df_group.rename(columns = {'Date': 'ds'})



# Show the end of th data

display(df_group.tail())

display(df_group.head())
model_scores=[]
model_train = df_group.iloc[:int(df_group.shape[0]*0.95)]

valid = df_group.iloc[int(df_group.shape[0]*0.95):]

y_pred = valid.copy()

model_scores=[]
model_ar = auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=5,max_q=0,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ar.fit(model_train["Confirmed"])
prediction_ar=model_ar.predict(len(valid))

y_pred["AR Model Prediction"]=prediction_ar
model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))

print("Root Mean Square Error for AR Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["AR Model Prediction"],

                    mode='lines+markers',name="Prediction of Confirmed Cases",))

fig.update_layout(title="Confirmed Cases AR Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases")

fig.show()
AR_model_new_prediction=[]

new_date=[]

# predicting next 20 days

for i in range(0,21):

    new_date.append(df_group.index[-1]+timedelta(days=i))

    AR_model_new_prediction.append(model_ar.predict(len(valid)+i)[-1])
pd.options.display.float_format = '{:.3f}'.format

model_predictions=pd.DataFrame(zip(new_date,AR_model_new_prediction),

                               columns=['Dates', 'AR_model_new_prediction'])

model_predictions.head()
model_train = df_group.iloc[:int(df_group.shape[0]*0.95)]

valid = df_group.iloc[int(df_group.shape[0]*0.95):]

y_pred = valid.copy()
model_ma= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=5,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ma.fit(model_train["Confirmed"])
prediction_ma=model_ma.predict(len(valid))

y_pred["MA Model Prediction"]=prediction_ma
model_scores.append(np.sqrt(mean_squared_error(valid["Confirmed"],prediction_ma)))

print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(valid["Confirmed"],prediction_ma)))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["MA Model Prediction"],

                    mode='lines+markers',name="Prediction for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases MA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases")

fig.show()
MA_model_new_prediction=[]

for i in range(0, 21):

    MA_model_new_prediction.append(model_ma.predict(len(valid)+i)[-1])

model_predictions["MA Model Prediction"]=MA_model_new_prediction

model_predictions.head()
model_train = df_group.iloc[:int(df_group.shape[0]*0.95)]

valid = df_group.iloc[int(df_group.shape[0]*0.95):]

y_pred = valid.copy()
model_arima= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=6,max_q=6,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_arima.fit(model_train["Confirmed"])
prediction_arima=model_arima.predict(len(valid))

y_pred["ARIMA Model Prediction"]=prediction_arima
model_scores.append(np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima)))

print("Root Mean Square Error for ARIMA Model: ",np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima)))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],

                    mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],

                    mode='lines+markers',name="Validation Data for Confirmed Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Model Prediction"],

                    mode='lines+markers',name="Prediction for Confirmed Cases",))

fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases")

fig.show()
ARIMA_model_new_prediction=[]

for i in range(0, 21):

    ARIMA_model_new_prediction.append(model_arima.predict(len(valid)+i)[-1])

model_predictions["ARIMA Model Prediction"]=ARIMA_model_new_prediction

model_predictions.head()
df_prophet = df_group[['Confirmed']]
df_prophet = df_prophet.reset_index()
df_prophet = df_prophet.rename(columns = {'ds': 'ds', 'Confirmed': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
m = Prophet()

m.fit(df_prophet)
future = m.make_future_dataframe(periods = 20)

forecast = m.predict(future)
model_scores.append(np.sqrt(mean_squared_error(df_group["Confirmed"],forecast['yhat'].head(df_group.shape[0]))))

print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(df_group["Confirmed"],forecast['yhat'].head(df_group.shape[0]))))
figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Confirmed Cases')
figure2 = m.plot_components(forecast)
model_predictions["Prophet's Prediction"]=list(forecast["yhat"].tail(21))
model_predictions.head()
model_names=['Auto Regressive Model (AR)', 'Moving Average Model (MA)', 'ARIMA Model', 'Facebook\'s Prophet Model']
model_summary=pd.DataFrame(zip(model_names,model_scores),columns=["Model Name","Root Mean Squared Error"]).sort_values(["Root Mean Squared Error"])
model_summary
model_train = df_group.iloc[:int(df_group.shape[0]*0.95)]

valid = df_group.iloc[int(df_group.shape[0]*0.95):]

y_pred = valid.copy()
model_arima_deaths= auto_arima(model_train["Recovered"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=6,max_q=6,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_arima_deaths.fit(model_train["Recovered"])
predictions_deaths=model_arima_deaths.predict(len(valid))

y_pred["ARIMA Recovered Prediction"]=predictions_deaths
print("Root Mean Square Error for ARIMA Model: ",np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima)))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Recovered"],

                    mode='lines+markers',name="Train Data for Recovered Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Recovered"],

                    mode='lines+markers',name="Validation Data for Recovered Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Recovered Prediction"],

                    mode='lines+markers',name="Prediction for Recovered Cases",))

fig.update_layout(title="Recovered Cases ARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Recovered Cases")

fig.show()
model_train = df_group.iloc[:int(df_group.shape[0]*0.95)]

valid = df_group.iloc[int(df_group.shape[0]*0.95):]

y_pred = valid.copy()
model_arima_deaths= auto_arima(model_train["Deaths"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=6,max_q=6,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_arima_deaths.fit(model_train["Deaths"])
predictions_deaths=model_arima_deaths.predict(len(valid))

y_pred["ARIMA Death Prediction"]=predictions_deaths
print("Root Mean Square Error for ARIMA Model: ",np.sqrt(mean_squared_error(valid["Confirmed"],prediction_arima)))
fig=go.Figure()

fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Deaths"],

                    mode='lines+markers',name="Train Data for Death Cases"))

fig.add_trace(go.Scatter(x=valid.index, y=valid["Deaths"],

                    mode='lines+markers',name="Validation Data for Death Cases",))

fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Death Prediction"],

                    mode='lines+markers',name="Prediction for Death Cases",))

fig.update_layout(title="Death Cases ARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Death Cases")

fig.show()