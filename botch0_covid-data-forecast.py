# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from fbprophet import Prophet
import matplotlib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
complete_df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
complete_df.head()
# Used as the ultimate truth
df_by_country = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
df_confirmed.head()
def spiltRows(df):
    train = df.iloc[:190]
    test = df.iloc[190:]
    
    return train, test

def getColumn(col):
    return np.array(col).reshape(-1,1)
# Melt Data from Column 4 onward

dates = df_confirmed.columns[4:]

confirmed = pd.melt(
    df_confirmed,
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
    value_vars=dates, 
    var_name='Date', 
    value_name='Confirmed'
)

deaths = pd.melt(
    df_deaths,
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
    value_vars=dates, 
    var_name='Date', 
    value_name='Deaths'
)

recovered = pd.melt(
    df_recovered,
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 
    value_vars=dates, 
    var_name='Date', 
    value_name='Recovered'
)
merge_data = confirmed.merge(
  right=deaths, 
  how='left',
  on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
)

merge_data = merge_data.merge(
  right=recovered, 
  how='left',
  on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
)
df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv", parse_dates=["Date"])
df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv", parse_dates=["Date"])
df_train.head()
# Remove NaN values
merge_data.isna().sum()
merge_data[['Province/State']] = merge_data[['Province/State']].fillna('')
merge_data[['Confirmed', 'Deaths', 'Recovered']] = merge_data[['Confirmed', 'Deaths', 'Recovered']].fillna(0)
merge_data.isna().sum()
# Convert Date to to pd datatime
merge_data['Date'] = pd.to_datetime(merge_data['Date'])
complete_df.isna().sum()
complete_df[['Province/State']] = complete_df[['Province/State']].fillna('')
complete_df.isna().sum()
df_train.isna().sum()
df_test.isna().sum()
df_train.fillna("")
df_test.fillna("")
df_train.isna().sum()
df_test.isna().sum()
df_train_canada = complete_df[complete_df['Country/Region']=='Canada'].groupby(['Date'])[['Confirmed','Deaths', 'Recovered']].sum().reset_index()
df_train_canada.head()
df_train_world = complete_df.groupby(["Date"])[['Confirmed','Deaths', 'Recovered']].sum().reset_index()
x = df_train_world.index
y = df_train_world.Confirmed
y1= df_train_world.Deaths
y2 = df_train_world.Recovered


fig, ax = plt.subplots(figsize=(15,15))

plt.scatter(x,y,color='blue' , label='Confirmed Cases')
plt.scatter(x,y1,color='red' ,label="Deaths Cases")
plt.scatter(x,y2,color='grey',label="Recovered Cases")
plt.title("World infections")
plt.legend()
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()
df_train_world.head()
x = df_train_canada.index
y = df_train_canada.Confirmed
y1= df_train_canada.Deaths
y2 = df_train_canada.Recovered

fig, ax = plt.subplots(figsize=(15,15))

plt.scatter(x,y,color='blue' , label='Confirmed Cases')
plt.scatter(x,y1,color='red' ,label="Deaths Cases")
plt.scatter(x,y2,color='grey',label="Recovered Cases")
plt.title("Canada Cases")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend()
plt.show()
#Global
df_global = df_by_country.copy().drop(['Lat', 'Long_', 'Last_Update', 'Incident_Rate', 'People_Tested', 'People_Hospitalized', 'Mortality_Rate', 'UID', 'ISO3'], axis=1)
global_sum = pd.DataFrame(df_global.sum()).transpose()
global_sum[['Confirmed', 'Deaths', 'Recovered', 'Active']].style.format("{:,.0f}")
#Canada
canada_summ = df_global[df_global['Country_Region']=='Canada'].reset_index()
canada_summ[['Confirmed', 'Deaths', 'Recovered', 'Active']].style.format("{:,.0f}")
global_confirmed = complete_df[complete_df['Date'] == max(complete_df['Date'])].reset_index()
canada_confirmed_grouped = global_confirmed[global_confirmed['Country/Region']=='Canada'].groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
global_confirmed_grouped = global_confirmed.groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
fig = go.Figure(data=go.Choropleth(
                    locations=global_confirmed_grouped["Country/Region"],
                    z=global_confirmed_grouped['Confirmed'],
                    locationmode='country names', 
                    colorscale = 'Viridis',
                   ))
fig.update_geos(fitbounds="locations", visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title='Confirmed Cases Worldwide')
fig.show()
fig = go.Figure(data=go.Choropleth(
                    locations=global_confirmed_grouped["Country/Region"],
                    z=global_confirmed_grouped['Recovered'],
                    locationmode='country names', 
                    colorscale = 'Viridis',
                   ))
fig.update_geos(fitbounds="locations", visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title='Recovered Cases Worldwide')
fig.show()
fig = go.Figure(data=go.Choropleth(
                    locations=global_confirmed_grouped["Country/Region"],
                    z=global_confirmed_grouped['Active'],
                    locationmode='country names', 
                    colorscale = 'Viridis',
                   ))
fig.update_geos(fitbounds="locations", visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title='Active Cases Worldwide')
fig.show()
fig = go.Figure(data=go.Choropleth(
                    locations=canada_confirmed_grouped["Country/Region"],
                    z=canada_confirmed_grouped['Confirmed'],
                    locationmode='country names', 
                    colorscale = 'Viridis',
                    showscale=False
                   ))
fig.update_geos(fitbounds="locations", visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title='Confirmed Cases Canada')
fig.show()
fig = go.Figure(data=go.Choropleth(
                    locations=canada_confirmed_grouped["Country/Region"],
                    z=canada_confirmed_grouped['Active'],
                    locationmode='country names', 
                    colorscale = 'Viridis',
                    showscale=False
                   ))
fig.update_geos(fitbounds="locations", visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title='Active Cases Canada')
fig.show()
Canada = df_train[df_train['Country_Region']=='Canada']
Canada.set_index('Date', inplace=True)
Canada.groupby('Target')['TargetValue'].plot(legend=True)
global_total_cases = merge_data.groupby(["Date"])[['Confirmed','Deaths', 'Recovered']].sum().reset_index()
global_total_cases.tail(1)
df_train_world.tail(1)
X, Y = spiltRows(global_total_cases)
y_pred = X["Confirmed"]

poly = PolynomialFeatures(degree = 5) # 2 -5 
train = poly.fit_transform(getColumn((X.index)))
global_train = poly.fit_transform(getColumn(global_total_cases.index))

regressor = LinearRegression(normalize=True)
regressor.fit(train, y_pred)

predictions_poly = regressor.predict(global_train)
x = global_total_cases.Date
y = global_total_cases["Confirmed"]
y1= predictions_poly

fig, ax = plt.subplots(figsize=(15,15))

plt.scatter(x, y, label='Confirmed Cases Worldwide', color='blue')
plt.scatter(x, y1, label="Polynomial Regression", color='red')
plt.title("Using Polynomial Regression for Prediction of Cases worldwide")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend()
plt.show()
canada_total_cases = df_train_canada.copy()
X, Y = spiltRows(canada_total_cases)
y_pred = X["Confirmed"]

poly = PolynomialFeatures(degree = 5)
train = poly.fit_transform(getColumn((X.index)))
global_train = poly.fit_transform(getColumn(canada_total_cases.index))

regressor = LinearRegression(normalize=True)
regressor.fit(train, y_pred)

predictions_poly = regressor.predict(global_train)
x = canada_total_cases.Date
y = canada_total_cases["Confirmed"]
y1= predictions_poly

fig, ax = plt.subplots(figsize=(15,15))

plt.scatter(x, y, label='Confirmed Cases Canda', color='blue')
plt.scatter(x, y1, label="Polynomial Regression", color='red')
plt.title("Using Polynomial Regression for Prediction of Cases Canada")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend()
plt.show()
X = pd.DataFrame(df_train_world['Date'])
Y = pd.DataFrame(df_train_world['Confirmed'])
X_train, X_test, y_train, y_test = train_test_split(X, Y)
model_SVC = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(), 
    StandardScaler(),
    SVC()
)
model_SVC.fit(X_train, y_train)
import matplotlib

x=df_train_world.index

y=df_train_world.Confirmed

fig, ax = plt.subplots(figsize=(15,15))

plt.scatter(X,Y,color='red')
plt.plot(X_test,model_SVC.predict(X_test),color='blue')
plt.title("Support Vector Model")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()
df_predict_world_confirmed = df_train_world[['Date', 'Confirmed']]
df_predict_world_confirmed.columns = ['ds', 'y']

df_predict_world_recovered = df_train_world[['Date', 'Recovered']]
df_predict_world_recovered.columns = ['ds', 'y']

df_predict_world_deaths = df_train_world[['Date', 'Deaths']]
df_predict_world_deaths.columns = ['ds', 'y']
pred = Prophet(yearly_seasonality=False, daily_seasonality=False)
pred.fit(df_predict_world_confirmed)
future = pred.make_future_dataframe(periods=6, freq='MS')
forecast = pred.predict(future)

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
fig = pred.plot(forecast,ax=ax, xlabel='Date', ylabel='Cases')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title("Worldwide Confirmed Forecast", size=34)
plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
pred = Prophet(yearly_seasonality=False, daily_seasonality=False)
pred.fit(df_predict_world_recovered)
future = pred.make_future_dataframe(periods=6, freq='MS')
forecast = pred.predict(future)

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
fig = pred.plot(forecast,ax=ax, xlabel='Date', ylabel='Cases')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title("Worldwide Recovered Forecast", size=34)
plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
pred = Prophet(yearly_seasonality=False, daily_seasonality=False)
pred.fit(df_predict_world_deaths)
future = pred.make_future_dataframe(periods=6, freq='MS')
forecast = pred.predict(future)

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
fig = pred.plot(forecast,ax=ax, xlabel='Date', ylabel='Cases')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title("Worldwide Deaths Forecast", size=34)
plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
df_predict_canada_confirmed = df_train_canada[['Date', 'Confirmed']]
df_predict_canada_confirmed.columns = ['ds', 'y']

df_predict_canada_deaths = df_train_canada[['Date', 'Deaths']]
df_predict_canada_deaths.columns = ['ds', 'y']
pred = Prophet(yearly_seasonality=False, daily_seasonality=False)
pred.fit(df_predict_canada_confirmed)
future = pred.make_future_dataframe(periods=6, freq='MS')
forecast = pred.predict(future)

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
fig = pred.plot(forecast,ax=ax, xlabel='Date', ylabel='Cases')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title("Canada Confirmed Forecast", size=34)
plt.show()
pred = Prophet(yearly_seasonality=False, daily_seasonality=False)
pred.fit(df_predict_canada_deaths)
future = pred.make_future_dataframe(periods=6, freq='MS')
forecast = pred.predict(future)

f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(15)
fig = pred.plot(forecast,ax=ax, xlabel='Date', ylabel='Cases')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title("Canada Deaths Forecast", size=34)
plt.show()
pred.plot_components(forecast)
# confirmed future DF
pred = Prophet(yearly_seasonality=False, daily_seasonality=False)
pred.fit(df_predict_canada_confirmed)
future_dates = pred.make_future_dataframe(periods=6, freq='MS')
canada_forecast_df_confirmed = pred.predict(future_dates)

# deaths future DF
pred = Prophet(yearly_seasonality=False, daily_seasonality=False)
pred.fit(df_predict_canada_deaths)
future_dates = pred.make_future_dataframe(periods=6, freq='MS')
canada_forecast_df_deaths = pred.predict(future_dates)
canada_forecast_df_confirmed[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
canada_forecast_df_deaths[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
canada_predict_heat = canada_forecast_df_confirmed[['ds', 'yhat']].tail(1)
canada_predict_heat.columns = ['Date', 'Confirmed']
canada_predict_heat['Deaths'] = canada_forecast_df_deaths[['yhat']].tail(1)
canada_predict_heat['Country/Region'] = 'Canada'
canada_predict_heat
canada_predict_heat['text'] = 'Confirmed ' + canada_predict_heat['Confirmed'].astype(str) + '<br>' \
                              'Deaths ' + canada_predict_heat['Deaths'].astype(str)
# Forecasted cases and deaths
fig = go.Figure(data=go.Choropleth(
                    locations=canada_predict_heat["Country/Region"],
                    text=canada_predict_heat["text"],
                    z=canada_predict_heat['Confirmed'].astype(float),
                    locationmode='country names', 
                    colorscale = 'electric',
                    showscale=False
                   ))
fig.update_geos(fitbounds="locations", visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title='Predicted Confirmed/Deaths Canada')
fig.show()
# Current Case count
fig = go.Figure(data=go.Choropleth(
                    locations=canada_confirmed_grouped["Country/Region"],
                    z=canada_confirmed_grouped['Confirmed'],
                    locationmode='country names', 
                    colorscale = 'Viridis',
                    showscale=False
                   ))
fig.update_geos(fitbounds="locations", visible=False)
fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title='Confirmed Cases Canada')
fig.show()
