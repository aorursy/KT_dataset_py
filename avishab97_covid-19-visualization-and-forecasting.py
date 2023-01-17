import pandas as pd

import numpy as np

import geopandas as gpd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium

import requests

!pip install googletrans

import googletrans

import re

from pyproj import CRS

from pandas.plotting import lag_plot

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

!pip install pmdarima

from pmdarima.arima import auto_arima

from fbprophet import Prophet

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from statsmodels.tsa.seasonal import seasonal_decompose

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

import time

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
confirmed.head()
confirmed['Province/State'] = confirmed['Province/State'].fillna('Unknown')

deaths['Province/State'] = deaths['Province/State'].fillna('Unknown')

recovered['Province/State'] = recovered['Province/State'].fillna('Unknown')
column = confirmed.columns[len(confirmed.columns)-1]

confirmed_per_country = confirmed.groupby('Country/Region',as_index=False)[column].sum()
fig = px.choropleth(confirmed_per_country, locations=confirmed_per_country['Country/Region'],color=confirmed_per_country[column],

                   locationmode='country names',hover_name=confirmed_per_country['Country/Region'],

                    color_continuous_scale=px.colors.sequential.Tealgrn)

fig.update_layout(

    title='Total Confirmed Cases Per Country',

)

fig.show()
column = deaths.columns[len(confirmed.columns)-1]

deaths_per_country = deaths.groupby('Country/Region',as_index=False)[column].sum()
fig = px.choropleth(deaths_per_country, locations=deaths_per_country['Country/Region'],color=deaths_per_country[column],

                   locationmode='country names',hover_name=deaths_per_country['Country/Region'],

                    color_continuous_scale=px.colors.sequential.Redor)

fig.update_layout(

    title='Total Deaths Per Country',

)

fig.show()
column = recovered.columns[len(recovered.columns)-1]

recovered_per_country = recovered.groupby('Country/Region',as_index=False)[column].sum()
fig = px.choropleth(recovered_per_country, locations=recovered_per_country['Country/Region'],color=recovered_per_country[column],

                   locationmode='country names',hover_name=recovered_per_country['Country/Region'],

                    color_continuous_scale=px.colors.sequential.Blues)

fig.update_layout(

    title='Total Recoveries Per Country',

)

fig.show()
top_10_countries_confirmed = confirmed_per_country.sort_values(column,ascending=True).tail(10)

fig = plt.figure(figsize=(10,7))

fig.suptitle('Highest Confirmed Cases as of 9-Sept-2020', fontsize=20)

plt.xlabel('xlabel', fontsize=18)

plt.ylabel('ylabel', fontsize=16)

sns.set_style('whitegrid')

sns.barplot(x=column,y='Country/Region',data=top_10_countries_confirmed,palette='Greens')

plt.show()
top_10_countries_deaths = deaths_per_country.sort_values(column,ascending=True).tail(10)

fig = plt.figure(figsize=(10,7))

fig.suptitle('Highest Deaths as of 9-Sept-2020', fontsize=20)

plt.xlabel('xlabel', fontsize=18)

plt.ylabel('ylabel', fontsize=16)

sns.set_style('whitegrid')

sns.barplot(x=column,y='Country/Region',data=top_10_countries_deaths,palette='OrRd')

plt.show()
top_10_countries_recovered = recovered_per_country.sort_values(column,ascending=True).tail(10)

fig = plt.figure(figsize=(10,7))

fig.suptitle('Most Recoveries as of 9-Sept-2020', fontsize=20)

plt.xlabel('xlabel', fontsize=18)

plt.ylabel('ylabel', fontsize=16)

sns.set_style('whitegrid')

sns.barplot(x=column,y='Country/Region',data=top_10_countries_recovered,palette='Blues')

plt.show()
columns = confirmed.columns[4:]

data = confirmed.groupby('Country/Region',as_index=False)[columns].sum()

temp = data.melt(['Country/Region'],var_name='Date', value_name='Cases')

temp.head()
fig = px.choropleth(temp, locations=temp['Country/Region'],

                    color=temp['Cases'],locationmode='country names', 

                    hover_name=temp['Country/Region'], 

                    color_continuous_scale=px.colors.sequential.deep,

                    animation_frame="Date")

fig.update_layout(



    title='Evolution of confirmed cases In Each Country',

)

fig.show()
data_over_time = pd.DataFrame()
dates = []

confirm = []

death = []

recoveries = []

for col in confirmed.columns[4:]:

    dates.append(col)

    confirm.append(confirmed[col].sum())

    death.append(deaths[col].sum())

    recoveries.append(recovered[col].sum())

data_over_time['observationDate'] = dates

data_over_time['confirmCases'] = confirm

data_over_time['deaths'] = death

data_over_time['recoveries'] = recoveries

data_over_time['observationDate'] = pd.to_datetime(data_over_time['observationDate'])
data_over_time.head()
temp = data_over_time.melt(id_vars='observationDate',value_vars=['confirmCases','deaths','recoveries'],var_name='Case',value_name='Count')

temp.head()
fig = px.area(temp, x="observationDate", y="Count", color="Case",

    height=600, width=700,

             title='Cases over time', color_discrete_sequence = ['rgb(27,158,119)','#FF7F0E','#1F77B4'])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig = px.line(data_over_time, x="observationDate", y='confirmCases', width=700, color_discrete_sequence=['rgb(27,158,119)'])

fig.show()
fig = px.line(data_over_time, x="observationDate", y='deaths', width=700, color_discrete_sequence=['#FF7F0E'])

fig.show()
fig = px.line(data_over_time, x="observationDate", y='recoveries', width=700, color_discrete_sequence=['#1F77B4'])

fig.show()
confirmed_china = confirmed[confirmed['Country/Region'] == 'China'].reset_index(drop=True)
gdf_china = gpd.GeoDataFrame(

    confirmed_china, geometry=gpd.points_from_xy(confirmed_china.Long, confirmed_china.Lat))
gdf_china.head()
data = pd.DataFrame()

data['geoid'] = gdf_china.index.astype(str)

data['confirmed_cases_by_9/10/20'] = gdf_china['9/10/20']

data['province/state'] = gdf_china['Province/State'].str.lower()
url = 'https://storage.googleapis.com/kagglesdsdata/datasets/496669/922532/china.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201001%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201001T184046Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=37959d504d9db53cb4acf85367f26d4f63ccda696901eee5132351c8a1f3b19bf765a9b2dd025526a4f1cbc7f56abc0296a31d07099f4680b21225361d175e1efbf43d05ec05ef5806648d4988f4e8a61bd73753a9e507c956f7d0959c4da996628521eabef7af98a90463943ab02cfdab8bb85525a8fa4e9e1576759de17045ccddc7d4202ddf8913a3e0002be19f0aaf3aa04feee7cb6af9f849965d43f245de55ad6beef6028d4be3c60084ed2be8d97b17f16c04ae9b7f2b621d9002a1f9171951ca995a8f379f311135e1a455d76fe009e2b9f49a90904394472cda22558a2b9e2b4848751f15f9fd4bcc20ac3ae89735dd41096c1bde571a92d4eac2a3'

china_geo = requests.get(url).json()

df = gpd.GeoDataFrame.from_features(china_geo, crs='EPSG:4326')
translator = googletrans.Translator()

result = []

for name in df['name']:

    result.append(translator.translate(name).text)

df['english_name'] = result
province = []

for name in df['english_name']:

    #print(name)

    a = re.match('\w+ \w+ autonomous region',name.lower()) 

    b = re.match('\w+ autonomous region',name.lower())

    c = re.match('\w+ province',name.lower())

    d = re.match('\w+ \w+ sar',name.lower())

    e = re.match('\w+ special administrative region',name.lower())

    f = re.match('\w+ \w+ special administrative region',name.lower())

    g = re.match('\w+ city',name.lower())

    if a:

        words = a[0].split(' ')

        province.append(words[0]+' '+words[1])

    elif b:

        words = b[0].split(' ')

        province.append(words[0])

    elif c:

        words = c[0].split(' ')

        province.append(words[0])

    elif d:

        words = d[0].split(' ')

        province.append(words[0]+' '+words[1])

    elif e:

        words = e[0].split(' ')

        province.append(words[0])

    elif f:

        words = f[0].split(' ')

        province.append(words[0]+' '+words[1])

    elif g:

        words = g[0].split(' ')

        province.append(words[0])

    else:

        province.append(name.lower())

df['english_province'] = province

        
df['english_name'].unique()
def correct_provinces(row):

    province = row

    if  row == 'macau':

        province =  'macao'

    elif row == 'xinjiang':

        province = 'xinjiang uygur'

    elif row == 'guangxi':

        province = 'guangxi zhuang'

    elif row == 'ningxia':

        province = 'ningxia hui'

    return province

data['province/state'] = data['province/state'].apply(lambda row: correct_provinces(row),1)
geometry = []

for province in data['province/state']:

    for idx,row in df.iterrows():

        if province == row['english_province']:

            geo = df['geometry'][idx]

            geometry.append(geo)

            break
data['geometry'] = geometry
data_gdf = gpd.GeoDataFrame(

    data, geometry='geometry')

type(data_gdf)
data_gdf.crs = CRS.from_epsg(4326)
m = folium.Map(location=[30.5928, 114.3055], tiles = 'cartodbpositron', control_scale=True,zoom_start=4,

    min_zoom=3,

    max_zoom=7)



choropleth = folium.Choropleth(

    geo_data=data_gdf,

    name='Confirmed Cases in China as of 9/10/20',

    data=data_gdf,

    columns=['geoid', 'confirmed_cases_by_9/10/20'],

    key_on='feature.id',

    fill_color='YlOrRd',

    fill_opacity=0.7,

    line_opacity=0.2,

    line_color='white',

    line_weight=0,

    highlight=True,

    smooth_factor=1.0,

    zoom_on_click=True,

    #threshold_scale=[100, 250, 500, 1000, 2000],

    legend_name= 'Confirmed Cases in China as of 9/10/20').add_to(m)



choropleth.geojson.add_child(folium.features.GeoJsonTooltip(

        fields=['confirmed_cases_by_9/10/20'],

        aliases=['Cases'],

        style=('background-color: grey; color: white;'),

        localize=True

        )

)

#Show map

m
columns = gdf_china.columns[4:-2]

data1 = gdf_china.groupby('Province/State',as_index=False)[columns].sum()

temp = data1.melt(['Province/State'],var_name='Date', value_name='Cases')

temp.head()
fig = px.line(temp, x='Date', y='Cases', color='Province/State', title='China: State-wise cases')

fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),

                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))

fig.show()
daily_confirmed = data_over_time[['observationDate','confirmCases']]
daily_confirmed = daily_confirmed.set_index('observationDate')
train_size = int(len(daily_confirmed) * 0.95)

train_confirmed, test_confirmed = daily_confirmed[0:train_size], daily_confirmed[train_size:len(daily_confirmed)]
train_confirmed.head()
lag_plot(train_confirmed)
model_comparison = []
model_ar= auto_arima(train_confirmed['confirmCases'],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=7,max_q=0,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ar.fit(train_confirmed['confirmCases'])
prediction_ar=model_ar.predict(len(test_confirmed['confirmCases']))
rmse = np.sqrt(mean_squared_error(test_confirmed['confirmCases'],prediction_ar))

print("Root Mean Square Error for AR Model: ",rmse)

model_comparison.append(rmse)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prediction_ar,

              mode='lines+markers',name="AR predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases AR Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
AR_model_new_prediction=[]

for i in range(1,21):

    AR_model_new_prediction.append(model_ar.predict(len(test_confirmed['confirmCases'])+i)[-1])

print(AR_model_new_prediction)
model_ma= auto_arima(train_confirmed['confirmCases'],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=6,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_ma.fit(train_confirmed['confirmCases'])
prediction_ma=model_ma.predict(len(test_confirmed['confirmCases']))
np.sqrt(mean_squared_error(test_confirmed['confirmCases'],prediction_ma))

print("Root Mean Square Error for MA Model: ",rmse)

model_comparison.append(rmse)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prediction_ma,

              mode='lines+markers',name="MA Predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases MA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
MA_model_new_prediction=[]

for i in range(1,21):

    MA_model_new_prediction.append(model_ma.predict(len(test_confirmed['confirmCases'])+i)[-1])

print(MA_model_new_prediction)
model_arima= auto_arima(train_confirmed['confirmCases'],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=7,max_q=7,

                   suppress_warnings=True,stepwise=False,seasonal=False)

model_arima.fit(train_confirmed['confirmCases'])
prediction_arima=model_arima.predict(len(test_confirmed['confirmCases']))
rmse = np.sqrt(mean_squared_error(test_confirmed['confirmCases'],prediction_arima))

print("Root Mean Square Error for MA Model: ",rmse)

model_comparison.append(rmse)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prediction_arima,

              mode='lines+markers',name="ARIMA Predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
ARIMA_model_new_prediction=[]

for i in range(1,21):

    ARIMA_model_new_prediction.append(model_arima.predict(len(test_confirmed['confirmCases'])+i)[-1])

print(ARIMA_model_new_prediction)
model_sarima= auto_arima(train_confirmed['confirmCases'],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=7,max_q=7,

                   m=7,suppress_warnings=True,stepwise=True,seasonal=True)

model_sarima.fit(train_confirmed['confirmCases'])
prediction_sarima=model_sarima.predict(len(test_confirmed['confirmCases']))
rmse = np.sqrt(mean_squared_error(test_confirmed['confirmCases'],prediction_sarima))

print("Root Mean Square Error for SARIMA Model: ",rmse)

model_comparison.append(rmse)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prediction_sarima,

              mode='lines+markers',name="SARIMA Predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases SARIMA Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
prophet_daily_confirmed = data_over_time[['observationDate','confirmCases']]

prophet_daily_confirmed.rename(columns = {"observationDate": "ds", 

                                  "confirmCases":"y"},inplace=True) 
model_prophet = Prophet()

model_prophet.fit(prophet_daily_confirmed)
future = model_prophet.make_future_dataframe(periods=20)

forecast = model_prophet.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
rmse = np.sqrt(mean_squared_error(daily_confirmed['confirmCases'],

                    forecast['yhat'].head(daily_confirmed['confirmCases'].shape[0])))

print("Root Mean Squared Error for Prophet Model: ",rmse)

model_comparison.append(rmse)
fig1 = model_prophet.plot(forecast)
fig2 = model_prophet.plot_components(forecast)
model_holt = Holt(np.asarray(train_confirmed['confirmCases'])).fit(smoothing_level=0.38, 

                                                                   smoothing_slope=0.38,optimized=False)
prdeictions_holt = model_holt.forecast(len(test_confirmed))

rmse = np.sqrt(mean_squared_error(

    test_confirmed["confirmCases"],prdeictions_holt))

print("Root Mean Square Error Holt's Linear Model: ",rmse)

model_comparison.append(rmse)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prdeictions_holt,

              mode='lines+markers',name="Holt Linear Predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases Holt Linear Model Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
Holt_model_new_prediction=[]

for i in range(1,21):

    Holt_model_new_prediction.append(model_holt.predict(len(test_confirmed['confirmCases'])+i)[-1])

print(Holt_model_new_prediction)
decomposition = seasonal_decompose(daily_confirmed)

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid
plt.figure(figsize=(10,12))

plt.subplot(411)

plt.plot(daily_confirmed, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()
resample = daily_confirmed.resample('2W')

weekly_mean_confirmed = resample.mean()

weekly_mean_confirmed.head(5).plot()

plt.show()
model_es=ExponentialSmoothing(np.asarray(train_confirmed['confirmCases']),seasonal_periods=14,trend='mul', seasonal='add').fit()
prdeictions_es = model_es.forecast(len(test_confirmed))

rmse = np.sqrt(mean_squared_error(

    test_confirmed["confirmCases"],prdeictions_es))

print("Root Mean Square Error Exponential Smoothing Model: ",rmse)

model_comparison.append(rmse)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prdeictions_es,

              mode='lines+markers',name="Exponential Smoothing Predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases Exponential Smoothing Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
ES_model_new_prediction=[]

for i in range(1,21):

    ES_model_new_prediction.append(model_es.predict(len(test_confirmed['confirmCases'])+i)[-1])

print(ES_model_new_prediction)
datewise_confirmed = data_over_time[['observationDate','confirmCases']].copy()
datewise_confirmed['month'] = datewise_confirmed['observationDate'].dt.month

datewise_confirmed['day'] = datewise_confirmed['observationDate'].dt.day

datewise_confirmed['week'] = datewise_confirmed['observationDate'].dt.week

datewise_confirmed['quarter'] = datewise_confirmed['observationDate'].dt.quarter

datewise_confirmed['daysSince'] = (datewise_confirmed['observationDate'] - datewise_confirmed['observationDate'].min()).dt.days
unixtime = []

    

for date in datewise_confirmed['observationDate']:

    unixtime.append(time.mktime(date.timetuple()))

datewise_confirmed['DateTime'] = unixtime

datewise_confirmed = datewise_confirmed.drop(['observationDate'],axis=1)
datewise_confirmed.info()
train_confirmed_reg, test_confirmed_reg = datewise_confirmed[0:train_size], datewise_confirmed[train_size:len(datewise_confirmed)]

y_train= train_confirmed_reg.pop('confirmCases')

X_train = train_confirmed_reg

y_test= test_confirmed_reg.pop('confirmCases')

X_test = test_confirmed_reg
model_xgb = XGBRegressor()
parameters = {'learning_rate': [0.1, 0.2, 0.3], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [4,5,6],

              'subsample': [0.6,0.7,0.8],

              'colsample_bytree': [0.6,0.7,0.8],

              'n_estimators': [500,1000,1500,2000]}
xgb_grid = GridSearchCV(model_xgb,

                        parameters,

                        cv = 2,

                        n_jobs = 5,

                        verbose=True)
xgb_grid.fit(X_train,y_train)
xgb_grid.best_params_
model_xgb1 = XGBRegressor(colsample_bytree=0.6,learning_rate=0.2,max_depth=5,min_child_weight=4,n_estimators=2000,subsample= 0.7)
model_xgb1.fit(X_train[['daysSince','DateTime','week']],y_train)

prdeictions_xgb = model_xgb1.predict(X_test[['daysSince','DateTime','week']])

rmse = np.sqrt(mean_squared_error(

    y_test,prdeictions_xgb))

print("Root Mean Square Error XGBRegressor Model: ",rmse)

model_comparison.append(rmse)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prdeictions_xgb,

              mode='lines+markers',name="Exponential Smoothing Predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases Exponential Smoothing Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
poly = PolynomialFeatures(degree=5)
train_confirmed_reg, test_confirmed_reg = datewise_confirmed[0:train_size], datewise_confirmed[train_size:len(datewise_confirmed)]
poly_train_confirmed_reg = poly.fit_transform(train_confirmed_reg[['daysSince','DateTime']])

poly_test_confirmed_reg = poly.fit_transform(test_confirmed_reg[['daysSince','DateTime']])
model_linear=LinearRegression(normalize=True)

model_linear.fit(poly_train_confirmed_reg,y_train)
prediction_poly=model_linear.predict(poly_test_confirmed_reg)

rmse_poly=np.sqrt(mean_squared_error(y_test,prediction_poly))

model_comparison.append(rmse_poly)

print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_confirmed.index,y=train_confirmed['confirmCases'],

              mode='lines+markers',name="Train Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=test_confirmed['confirmCases'],

              mode='lines+markers',name="Test Data for Confirmed Cases"))

fig.add_trace(go.Scatter(x=test_confirmed.index,y=prediction_poly,

              mode='lines+markers',name="Exponential Smoothing Predictions for Confirmed Cases"))

fig.update_layout(title="Confirmed Cases Exponential Smoothing Prediction",

                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
models = ['AR','MA','ARIMA','SARIMA','Prophet','Holt"s Linear','Exponential Smoothing','XGBRegression','Polynomial Regression']
model_rmse = pd.DataFrame()

model_rmse['models'] = models

model_rmse['RMSE'] = model_comparison
model_rmse.sort_values('RMSE',ascending=True).reset_index(drop=True)