import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import folium
import os

import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
pd.set_option('display.max_rows',20000, 'display.max_columns',100)
df_corona_in_india = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_corona_india = pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")
df_ICMR = pd.read_csv("../input/covid19-in-india/ICMRTestingLabs.csv")
df_Individual = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")
df_Hospital = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_Age = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")
df_Italy = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")
df_daily_US = pd.read_csv("../input/covid19-in-usa/us_states_covid19_daily.csv")
df_corona_in_india.tail()
df_corona_india.tail()
#Total cases of carona in India
df_corona_in_india['Total Cases'] = df_corona_in_india['Cured'] + df_corona_in_india['Deaths'] + df_corona_in_india['Confirmed']
#Active cases of carona in India
df_corona_in_india['Active Cases'] = df_corona_in_india['Total Cases'] - df_corona_in_india['Cured'] - df_corona_in_india['Deaths']
df_corona_in_india.tail()
import matplotlib.pyplot as plt
#Till 8th April Cases in India
df1= df_corona_in_india[df_corona_in_india['Date']=='26/05/20']
fig = px.bar(df1, x='State/UnionTerritory', y='Total Cases', color='Total Cases', height=600)
fig.update_layout(
    title='Till 26/05/20 Total Cases in India')
fig.show()
#Till 8th April Active Cases in India
df1= df_corona_in_india[df_corona_in_india['Date']=='26/05/20']
fig = px.bar(df1, x='State/UnionTerritory', y='Active Cases', color='Active Cases',barmode='group', height=600)
fig.update_layout(
    title='Till 26/05/20 Active Cases in India')
fig.show()
df_corona_in_india['Date'] =pd.to_datetime(df_corona_in_india.Date,dayfirst=True)
df_corona_in_india.tail()
#Daily Cases in India Datewise
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index()#.sort_values('Total Cases',ascending = True)
corona_data.tail(10)
corona_data.shape
corona_data['Daily Cases'] = corona_data['Total Cases'].sub(corona_data['Total Cases'].shift())
corona_data['Daily Cases'].iloc[0] = corona_data['Total Cases'].iloc[0]
corona_data['Daily Cases'] = corona_data['Daily Cases'].astype(int)
fig = px.bar(corona_data, y='Daily Cases', x='Date',hover_data =['Daily Cases'], color='Daily Cases', height=500)
fig.update_layout(
    title='Daily Cases in India Datewise')
fig.show()
corona_data['Corona Growth Rate'] = corona_data['Total Cases'].pct_change().mul(100).round(2)
#Corona Growth Rate Comparison with Previous Day
fig = px.bar(corona_data, y='Corona Growth Rate', x='Date',hover_data =['Corona Growth Rate','Total Cases'], height=500)
fig.update_layout(
    title='Corona Growth Rate(in Percentage) Comparison with Previous Day')
fig.show()
# print("hey")
#Total Cases in India State Datewise
corona_data = df_corona_in_india.groupby(['Date','State/UnionTerritory','Total Cases'])['Cured','Deaths','Active Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)
fig = px.bar(corona_data, y='Total Cases', x='Date',hover_data =['State/UnionTerritory','Active Cases','Deaths','Cured'], color='Total Cases',barmode='group', height=700)
fig.update_layout(
    title='Indian States with Current Total Corona Cases')
fig.show()
df_corona_india.tail()
#Pie chart visualization of states effected by caronavirus
fig = px.pie(df_Age, values='TotalCases', names='AgeGroup')
fig.update_layout(
    title='Age Group affected with COVID-19')
fig.show()
# #Agewise Gender affected by COVID-19(There are lot of Nan Values)
# fig = px.histogram(df_Individual.dropna(), x="age",color ='gender')
# fig.update_layout(
#     title='Agewise Gender affected by COVID-19')
# fig.show()
# #Agewise Covid-19 patients in State(There are lot of Nan Values)
# fig = px.histogram(df_Individual.dropna(), x="age",color ='detected_state')
# fig.update_layout(
#     title='Agewise Covid-19 patients in State')
# fig.show()
#Genderwise current status of COVID-19(There are lot of Nan Values)
fig = px.histogram(df_Individual.dropna(), x="gender",color ='current_status')
fig.update_layout(
    title='Genderwise current status of COVID-19')
fig.show()
# #Total Cases Datewise of Foreign Nationals
# carona_data = df_carona_india.groupby(['Date','Name of State / UT','Total Confirmed cases ( Foreign National )'])['Total Confirmed cases','Cured/Discharged/Migrated',].sum().reset_index().sort_values('Total Confirmed cases ( Foreign National )',ascending = True)
# fig = px.bar(carona_data, y='Total Confirmed cases ( Foreign National )', x='Date',hover_data =['Name of State / UT','Total Confirmed cases','Cured/Discharged/Migrated'], color='Total Confirmed cases ( Foreign National )', height=700)
# fig.update_layout(
#     title='Total Cases Datewise of Foreign Nationals')
# fig.show()
# #Total Cases Datewise of Indian Nationals
# carona_data = df_carona_india.groupby(['Date','Name of State / UT','Total Confirmed cases (Indian National)'])['Total Confirmed cases','Cured/Discharged/Migrated',].sum().reset_index().sort_values('Total Confirmed cases (Indian National)',ascending = True)
# fig = px.bar(carona_data, y='Total Confirmed cases (Indian National)', x='Date',hover_data =['Name of State / UT','Total Confirmed cases','Cured/Discharged/Migrated'], color='Total Confirmed cases (Indian National)', height=700)
# fig.update_layout(
#     title='Total Cases Datewise of Indian  Nationals')
# fig.show()
df_population_of_india = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_population_of_india.head()
#Statewise Total Cases
df_new = df1.groupby(['State/UnionTerritory','Cured','Deaths','Active Cases'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)
df_new.head()
#Using Merge two join two diffrent data frames and then sorting them in ascending
df_population_with_carona_case = df_population_of_india.merge(df_new, left_on='State / Union Territory', right_on='State/UnionTerritory')
df_population_with_carona_case=df_population_with_carona_case.drop(labels=['State/UnionTerritory','Sno'],axis=1)
df_population_with_carona_case=df_population_with_carona_case.sort_values('Total Cases',ascending=False)
df_population_with_carona_case.head()
#Pie chart visualization of states effected by caronavirus
fig = px.pie(df_population_with_carona_case, values='Total Cases', names='State / Union Territory')
fig.update_layout(
    title='Pie chart visualization of states effected by caronavirus')
fig.show()
#Latitude and Longitude of Indian State
df_with_utm = pd.read_csv("../input/utm-of-india/UTM ZONES of INDIA.csv")
df_with_utm.head()
#Data of Population,Cases of Carona and UTM 
df_pop_caro_utm = df_population_with_carona_case.merge(df_with_utm , left_on='State / Union Territory', right_on='State / Union Territory')
df_pop_caro_utm.head()
#Total Case in Indian States
fig = go.Figure(data=[go.Scatter(
    x=df_pop_caro_utm['State / Union Territory'][0:36],
    y=df_pop_caro_utm['Total Cases'][0:36],
    mode='markers',
    marker=dict(
        size=[100,90,80, 70, 60, 50, 40,35,35,35,35,35,35,35,35,35,35,30,28,28,25,25,20,15,15,15,15,10,10,10],
        showscale=True
        )
)])
fig.update_layout(
    title='Total Case in Indian States',
    xaxis_title="States",
    yaxis_title="Total Cases",
)
fig.show()
#Active Case in Indian States
fig = go.Figure(data=[go.Scatter(
    x=df_pop_caro_utm['State / Union Territory'][0:36],
    y=df_pop_caro_utm['Active Cases'][0:36],
    mode='markers',
    marker=dict(
        size=[100,90,80, 70, 60, 50, 40,35,35,35,35,35,35,35,35,35,35,30,28,28,25,25,20,15,15,15,15,10,10,10],
        showscale=True
        )
)])
fig.update_layout(
    title='Active Case in Indian States',
    xaxis_title="States",
    yaxis_title="Active Cases",
)
fig.show()
#Total Cases,Active Cases,Cured,Deaths from Corona Virus in India
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases','Active Cases','Cured','Deaths'].sum().reset_index().sort_values('Date',ascending=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Total Cases'],
                    mode='lines+markers',name='Total Cases'))
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Active Cases'], 
                mode='lines+markers',name='Active Cases'))
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Cured'], 
                mode='lines+markers',name='Cured'))
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Deaths'], 
                mode='lines+markers',name='Deaths'))
fig.update_layout(title_text='Curve Showing Different Cases from COVID-19 in India',plot_bgcolor='rgb(225,230,255)')
fig.show()
df_ICMR.head(4)
# #Testing till 26/05/20
# df_ICMR ['DateTime'] =pd.to_datetime(df_ICMR .DateTime,dayfirst=True)
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df_ICMR['DateTime'], y=df_ICMR['TotalSamplesTested'],
#                     mode='lines+markers',name='TotalSamplesTested'))
# fig.add_trace(go.Scatter(x=df_ICMR['DateTime'], y=df_ICMR['TotalIndividualsTested'], 
#                 mode='lines+markers',name='TotalIndividualsTested'))
# fig.add_trace(go.Scatter(x=df_ICMR['DateTime'], y=df_ICMR['TotalPositiveCases'], 
#                 mode='lines+markers',name='TotalPositiveCases'))
# fig.update_layout(title_text='ICMR TEST for COVID-19',plot_bgcolor='rgb(225,230,255)')
# fig.show()
# #Current Status of Patient wrt state he/she is QUARTINE and his/her Nationality
# df_Individual = df_Individual.fillna({
#     'nationality': 'Unknown','current_status': 'Unknown'})
# df_Individual_new= df_Individual.drop(labels=['id','government_id','diagnosed_date','age','gender','detected_city','detected_district','status_change_date','notes'],axis=1)
# df_Individual_new = df_Individual.groupby(['current_status','nationality','detected_state'])['id'].count().reset_index(name='count')
# fig = px.bar(df_Individual_new, x='count', y='detected_state', orientation='h',hover_data =['current_status','nationality','detected_state'], color='current_status',height=700)
# fig.update_layout(
#     title='Current Status of Patient wrt state he/she is QUARTINE and his/her Nationality')
# fig.show()
#Total Confirmed Cases INDIA vs ITALY
df_Italy_p = pd.read_csv("../input/covid19-in-italy/covid19_italy_province.csv")
corona_data_ITALY = df_Italy_p.groupby(['Date'])['TotalPositiveCases'].sum().reset_index().sort_values('TotalPositiveCases',ascending = True)

corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Date',ascending = True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=corona_data_ITALY['Date'], y=corona_data_ITALY['TotalPositiveCases'],
                    mode='lines+markers',name='Italy'))
fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Confirmed'],
                    mode='lines+markers',name='India'))
fig.update_layout(title_text='Total Confirmed Cases INDIA vs ITALY',plot_bgcolor='rgb(225,230,255)')
fig.show()
#COVID-19 Recovered Cases in INDIA Vs ITALY
df_Italy_r = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")
df_Italy_r['Total Cases'] = df_Italy_r['HospitalizedPatients']+df_Italy_r['IntensiveCarePatients'] + df_Italy_r['TotalHospitalizedPatients'] + df_Italy_r['HomeConfinement'] + df_Italy_r['CurrentPositiveCases'] + df_Italy_r['NewPositiveCases'] + df_Italy_r['Recovered'] + df_Italy_r['Deaths'] + df_Italy_r['TotalPositiveCases']
corona_data_ITALY = df_Italy_r.groupby(['Date'])['Recovered'].sum().reset_index().sort_values('Date',ascending = True)
df_corona_in_india['Date'] =pd.to_datetime(df_corona_in_india.Date,dayfirst=True)
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Date',ascending = True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=corona_data_ITALY['Date'], y=corona_data_ITALY['Recovered'],
                    mode='lines+markers',name='Italy'))
fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Cured'],
                    mode='lines+markers',name='India'))
fig.update_layout(title_text='COVID-19 Recovered Cases in INDIA Vs ITALY',plot_bgcolor='rgb(225,230,255)')
fig.show()
#COVID-19 Death Cases in ITALY VS INDIA
df_Italy_r = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")
df_Italy_r['Total Cases'] = df_Italy_r['HospitalizedPatients']+df_Italy_r['IntensiveCarePatients'] + df_Italy_r['TotalHospitalizedPatients'] + df_Italy_r['HomeConfinement'] + df_Italy_r['CurrentPositiveCases'] + df_Italy_r['NewPositiveCases'] + df_Italy_r['Recovered'] + df_Italy_r['Deaths'] + df_Italy_r['TotalPositiveCases']
corona_data_ITALY = df_Italy_r.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Date',ascending = True)
df_corona_in_india['Date'] =pd.to_datetime(df_corona_in_india.Date,dayfirst=True)
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Date',ascending = True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=corona_data_ITALY['Date'], y=corona_data_ITALY['Deaths'],
                    mode='lines+markers',name='Italy'))
fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Deaths'],
                    mode='lines+markers',name='India'))
fig.update_layout(title_text='COVID-19 Death Cases in INDIA Vs ITALY',plot_bgcolor='rgb(225,230,255)')
fig.show()
#Daily Deaths in Italy
corona_data = df_Italy_r.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Deaths',ascending = True)
corona_data['Daily Deaths'] = corona_data['Deaths'].sub(corona_data['Deaths'].shift())
corona_data['Daily Deaths'].iloc[0] = corona_data['Deaths'].iloc[0]
corona_data['Daily Deaths'] = corona_data['Daily Deaths'].astype(int)
fig = px.bar(corona_data, y='Daily Deaths', x='Date',hover_data =['Daily Deaths'], color='Daily Deaths', height=500)
fig.update_layout(
    title='DAILY DEATHS IN ITALY')
fig.show()
#Total Corona Cases INDIA vs USA
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Date',ascending = True)
corona_data_USA= df_daily_US.groupby(['dateChecked'])['positive'].sum().reset_index().sort_values('dateChecked',ascending = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona_data_USA['dateChecked'], y=corona_data_USA['positive'],
                    mode='lines+markers',name='USA'))

fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Total Cases'],
                    mode='lines+markers',name='India'))

fig.update_layout(title_text='Total Corona Cases INDIA vs USA',plot_bgcolor='rgb(225,230,255)')
fig.show()
#COVID-19 Recovered Cases in INDIA vs USA
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Date',ascending = True)
corona_data_USA= df_daily_US.groupby(['dateChecked'])['recovered'].sum().reset_index().sort_values('dateChecked',ascending = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona_data_USA['dateChecked'], y=corona_data_USA['recovered'],
                    mode='lines+markers',name='USA'))

fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Cured'],
                    mode='lines+markers',name='India'))

fig.update_layout(title_text='COVID-19 Recovered Cases in INDIA vs USA',plot_bgcolor='rgb(225,230,255)')
fig.show()
#COVID-19 Death Cases in INDIA vs USA
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Date',ascending = True)
corona_data_USA= df_daily_US.groupby(['dateChecked'])['death'].sum().reset_index().sort_values('dateChecked',ascending = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona_data_USA['dateChecked'], y=corona_data_USA['death'],
                    mode='lines+markers',name='USA'))

fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Deaths'],
                    mode='lines+markers',name='India'))

fig.update_layout(title_text='COVID-19 Death Cases in INDIA vs USA',plot_bgcolor='rgb(225,230,255)')
fig.show()
#DAILY DEATHS IN USA
corona_data = df_daily_US.groupby(['dateChecked'])['death'].sum().reset_index().sort_values('death',ascending = True)
corona_data['Daily Cases'] = corona_data['death'].sub(corona_data['death'].shift())
corona_data['Daily Cases'].iloc[0] = corona_data['death'].iloc[0]
corona_data['Daily Cases'] = corona_data['Daily Cases'].astype(int)
fig = px.bar(corona_data, y='Daily Cases', x='dateChecked',hover_data =['Daily Cases'], color='Daily Cases', height=500)
fig.update_layout(
    title='DAILY DEATHS IN USA')
fig.show()
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index()#.sort_values('Total Cases',ascending = True)
corona_data['Daily Cases'] = corona_data['Total Cases'].sub(corona_data['Total Cases'].shift())
corona_data['Daily Cases'].iloc[0] = corona_data['Total Cases'].iloc[0]
corona_data['Daily Cases'] = corona_data['Daily Cases'].astype(int)

x= corona_data['Date']
x.tail()
x= pd.DataFrame(x)
x['Daily Cases']=corona_data['Daily Cases']
x.tail()
from datetime import date, timedelta

sdate = date(2020, 1, 30)   # start date
edate = date(2020, 5, 26) 

dd = [sdate + timedelta(days=x) for x in range((edate-sdate).days + 1)]

dd = pd.Series(dd)

dd = pd.to_datetime(dd)
x.set_index(dd, inplace=True)
print(x.index)
ts = x['Daily Cases']
ts.tail(10)
plt.figure(figsize=(10,6))
plt.plot(ts)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
#     rolmean = pd.rolling_mean(timeseries, window=12)
#     rolstd = pd.rolling_std(timeseries, window=12)
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()
#Plot rolling statistics:
    plt.figure(figsize=(15,7))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
test_stationarity(ts)
ts_log=np.log(ts)
plt.figure(figsize=(15,7))
plt.plot(ts_log)
moving_avg = pd.Series(ts_log).rolling(window=2).mean()
plt.figure(figsize=(15,7))
plt.plot(ts_log)
plt.plot(moving_avg,color='red')
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(10)
ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.head()
test_stationarity(ts_log_moving_avg_diff)
expwighted_avg = ts_log.ewm(halflife=2).mean()
plt.figure(figsize=(15,7))
plt.plot(ts_log)
plt.plot(expwighted_avg,color='red')
from numpy import inf

ts_log[ts_log == -inf] = 0

ts_log_ewma_diff = ts_log - expwighted_avg



test_stationarity(ts_log_ewma_diff)
ts_log_diff = ts_log - ts_log.shift()

plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)
ts_log = pd.Series(ts_log)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,period=2)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


plt.figure(figsize=(15,7))
plt.subplot(411)
plt.plot(ts_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='residuals')
plt.legend(loc='best')
plt.tight_layout()
ts_log_decompose = residual #trend#residual needs to be used for modelling
ts_log_decompose.dropna(inplace = True)
test_stationarity(ts_log_decompose)
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf

lag_acf = acf(ts_log_diff,nlags=20)
lag_pacf = pacf(ts_log_diff,nlags=20,method='ols')
#plot ACF

plt.figure(figsize=(15,7))
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='grey')
plt.title('Autocorrelation Function')


plt.figure(figsize=(15,7))
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='grey')
plt.title('Partial Autocorrelation Function')
model = ARIMA(ts_log,order=(11,1,0))
results_AR = model.fit(method='css',solver='cg')
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
model = ARIMA(ts_log,order=(0,1,11))
results_MA = model.fit(method='css',solver='bfgs')
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
model = ARIMA(ts_log,order=(6,1,6))
results_ARIMA = model.fit()
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues,copy=True)
# print(predictions_ARIMA_diff)
predictions_ARIMA_diff_cum_sum = predictions_ARIMA_diff.cumsum()
# print(predictions_ARIMA_diff_cum_sum.head())
predictions_ARIMA_log = pd.Series(ts_log.ix[0],index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cum_sum,fill_value=0)
# predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
results_AR.summary()
sdate = date(2020, 5, 26)   # start date
edate = date(2020, 6, 26) 

dd = [sdate + timedelta(days=x) for x in range((edate-sdate).days + 1)]

dd = pd.Series(dd)

dd = pd.to_datetime(dd)

dd =  pd.DataFrame(dd)
X = dd.values

forecast = results_AR.predict(start=sdate,
                                end=edate,)
from fbprophet import Prophet
confirmed = x
confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
m = Prophet(interval_width=0.95,daily_seasonality=True,
           seasonality_mode= 'multiplicative')
m.fit(confirmed)
future = m.make_future_dataframe(periods=30)
future_confirmed = future.copy() # for non-baseline predictions later on
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
forecast_components = m.plot_components(forecast)
