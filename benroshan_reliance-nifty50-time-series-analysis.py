!pip install pmdarima
#Basic Libraries

import pandas as pd

import numpy as np

import datetime as dt

from datetime import datetime    

from pandas import Series 

import statsmodels.api as sm



#Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

sns.set_style("whitegrid")

%matplotlib inline

import altair as alt 

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



#Timeseries model libraries

from statsmodels.tsa.stattools import adfuller

from pmdarima import auto_arima

from fbprophet import Prophet



#Performance metric libraries

from sklearn.metrics import mean_absolute_error, mean_squared_error

from fbprophet.diagnostics import cross_validation

from fbprophet.diagnostics import performance_metrics

from fbprophet.plot import plot_cross_validation_metric





import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
reliance_raw=pd.read_csv("../input/nifty50-stock-market-data/RELIANCE.csv")



## print shape of dataset with rows and columns and information 

print ("The shape of the  data is (row, column):"+ str(reliance_raw.shape))

print (reliance_raw.info())
reliance_raw.head()
reliance_raw.dtypes
#Checking out the statistical measures

reliance_raw.describe()
#Creating a copy

reliance_analysis=reliance_raw.copy()



#Coverting date column to datetime data type

reliance_analysis['Date'] = reliance_analysis['Date'].apply(pd.to_datetime)



#Extracting Month, Week, Day,Day of week

reliance_analysis["Month"] = reliance_analysis.Date.dt.month

reliance_analysis["Week"] = reliance_analysis.Date.dt.week

reliance_analysis["Day"] = reliance_analysis.Date.dt.day

reliance_analysis["Day of week"] = reliance_analysis.Date.dt.dayofweek





#Setting date column as index

reliance_analysis.set_index("Date", drop=False, inplace=True)

reliance_analysis.iloc[:,15:19].head()
#Imputing null values with mean 

reliance_analysis.fillna(reliance_analysis.mean(),inplace=True)



#Checking for null values

reliance_analysis.isnull().sum()
#Size and style of the plot

plt.figure(figsize = (15, 7))

plt.style.use('seaborn-white')



#Subplots of distplot

plt.subplot(231)

sns.distplot(reliance_analysis['Prev Close'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(232)

sns.distplot(reliance_analysis['Open'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(233)

sns.distplot(reliance_analysis['High'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(234)

sns.distplot(reliance_analysis['Low'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(235)

sns.distplot(reliance_analysis['Close'])

fig = plt.gcf()

fig.set_size_inches(10,10)



plt.subplot(236)

sns.distplot(reliance_analysis['VWAP'])

fig = plt.gcf()

fig.set_size_inches(10,10)
fig = px.line(reliance_analysis, x='Date', y='VWAP',title='VWAP over Years(Use Rangeslider to slide over time)')

fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="YTD", step="year", stepmode="todate"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

fig.show()
cols_plot = ['Open', 'Close', 'High','Low']

axes = reliance_analysis[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

for ax in axes:

    ax.set_ylabel('Daily trade')
ax=reliance_analysis[['Volume']].plot(stacked=True)

ax.set_title('Volume over years',fontsize= 30)

ax.set_xlabel('Year',fontsize = 20)

ax.set_ylabel('Volume',fontsize = 20)

plt.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

         x=reliance_analysis['Date'],

         y=reliance_analysis['Open'],

         name='Open',

    line=dict(color='blue'),

    opacity=0.8))



fig.add_trace(go.Scatter(

         x=reliance_analysis['Date'],

         y=reliance_analysis['Close'],

         name='Close',

    line=dict(color='red'),

    opacity=0.8))



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="YTD", step="year", stepmode="todate"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

        

    

fig.update_layout(title_text='Open Vs Close',plot_bgcolor='rgb(248, 248, 255)',yaxis_title='Value')



fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(

         x=reliance_analysis['Date'],

         y=reliance_analysis['High'],

         name='High',

    line = dict(color='green', width=4, dash='dot'),

    opacity=0.8))



fig.add_trace(go.Scatter(

         x=reliance_analysis['Date'],

         y=reliance_analysis['Low'],

         name='Low',

    line=dict(color='orange', width=4, dash='dot'),

    opacity=0.8))



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="YTD", step="year", stepmode="todate"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

        

    

fig.update_layout(title_text='High Vs Low',plot_bgcolor='rgb(248, 248, 255)',yaxis_title='Value')



fig.show()

#Making a copy

reliance_lag=reliance_analysis.copy()

#Reset index

reliance_lag.reset_index(drop=True, inplace=True)

#Creating lag features

lag_features = ["High", "Low", "Volume", "VWAP"]



# Taking the number of days in window

window1 = 3

window2 = 7

window3 = 30



#Rolling mean

df_rolled_3d = reliance_lag[lag_features].rolling(window=window1, min_periods=0)

df_rolled_7d = reliance_lag[lag_features].rolling(window=window2, min_periods=0)

df_rolled_30d = reliance_lag[lag_features].rolling(window=window3, min_periods=0)



#Moving average

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)

df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)

df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)



#Standard deviation

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)

df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)

df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)



# Adding the features to the dataframe

for feature in lag_features:

    reliance_lag[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]

    reliance_lag[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]

    reliance_lag[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]

    

    reliance_lag[f"{feature}_std_lag{window1}"] = df_std_3d[feature]

    reliance_lag[f"{feature}_std_lag{window2}"] = df_std_7d[feature]

    reliance_lag[f"{feature}_std_lag{window3}"] = df_std_30d[feature]



reliance_lag.fillna(reliance_lag.mean(), inplace=True)



#Setting Date as index

reliance_lag.set_index("Date", drop=False, inplace=True)

reliance_lag.head()
#Printing the high curve

fig = go.Figure()

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['High'],

         name='High',

    line=dict(color='green'),

    opacity=0.8))



#Printing the low curve

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['Low'],

         name='Low',

    line=dict(color='orange'),

    opacity=0.8))



#Printing the high lag mean-30 days curve

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['High_mean_lag30'],

         name='High_mean_lag30',

    line=dict(color='red'),

    opacity=0.8))



#Printing the high lag standard deviation-30 days curve

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['High_std_lag30'],

         name='High_std_lag30',

    line=dict(color='royalblue'),

    opacity=0.8))



#Printing the low lag mean-30 days curve

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['Low_mean_lag30'],

         name='Low_mean_lag30',

    line=dict(color='yellow'),

    opacity=0.8))



#Printing the low lag standard deviation-30 days curve

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['Low_std_lag30'],

         name='Low_std_lag30',

    line=dict(color='pink'),

    opacity=0.8))



#Updating the time axis

fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="YTD", step="year", stepmode="todate"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

        

#Update the title   

fig.update_layout(title_text='High Vs Low with mean lag and standard deviation lag',plot_bgcolor='rgb(248, 248, 255)',yaxis_title='Value')



fig.show()
#Printing the Volume curve

fig = go.Figure()

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['Volume'],

         name='Volume',

    line=dict(color='green'),

    opacity=0.8))



#Printing the Volume_mean_lag-30 days curve

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['Volume_mean_lag30'],

         name='Volume_mean_lag30',

    line=dict(color='yellow'),

    opacity=0.8))

#Printing the Volume_std_lag30 curve

fig.add_trace(go.Scatter(

         x=reliance_lag['Date'],

         y=reliance_lag['Volume_std_lag30'],

         name='Volume_std_lag30',

    line=dict(color='blue'),

    opacity=0.8))



#Updating time axis

fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="YTD", step="year", stepmode="todate"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

        

#Updating layout 

fig.update_layout(title_text='Volume with mean lag and standard deviation lag',plot_bgcolor='rgb(248, 248, 255)',yaxis_title='Value')



fig.show()

#Setting the range of base plot

fig = px.line(reliance_analysis, x='Date', y='Volume',title='Volume during Phase 1 Lockdown(25 March – 14 April) and Phase 2 Lockdown (15 April – 3 May)', range_x=['2020-01-01','2020-06-30'])



# Adding the shape in the dates

fig.update_layout(

    shapes=[

        # First phase Lockdown

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-03-23",

            y0=0,

            x1="2020-04-14",

            y1=1,

            fillcolor="LightSalmon",

            opacity=0.5,

            layer="below",

            line_width=0,

        ),

        # Second phase Lockdown

        dict(

            type="rect",

            xref="x",

            yref="paper",

            x0="2020-04-15",

            y0=0,

            x1="2020-05-03",

            y1=1,

            fillcolor="Green",

            opacity=0.5,

            layer="below",

            line_width=0,

        )],

    annotations=[dict(x='2020-04-15', y=0.99, xref='x', yref='paper',

                    showarrow=False, xanchor='right', text='Phase 1 Lockdown'),

                 dict(x='2020-05-12', y=0.99, xref='x', yref='paper',

                    showarrow=False, xanchor='right', text='Phase 2 Lockdown')])



fig.show()
fig = px.line(reliance_analysis, x='Date', y='VWAP',title='VWAP after lockdown', range_x=['2020-03-23','2020-06-30'])

fig.show()
reliance_analysis_lockdown = reliance_analysis[reliance_analysis['Date'] >= '2020-03-23']

fig = go.Figure(data=[go.Candlestick(x=reliance_analysis_lockdown['Date'],

                open=reliance_analysis_lockdown['Open'],

                high=reliance_analysis_lockdown['High'],

                low=reliance_analysis_lockdown['Low'],

                close=reliance_analysis_lockdown['Close'])])



fig.show()
#Setting the date range for the base plot

fig = px.line(reliance_analysis, x='Date', y='VWAP', title='Major Corporate Announcements 2020(Till June 30)',range_x=['2020-01-01','2020-06-30'])



#Creating the line and news on announcement dates

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)',

    yaxis_title='NIFTY 50 VWAP',

    shapes = [dict(x0='2020-03-23', x1='2020-03-23', y0=0, y1=1, xref='x', yref='paper', line_width=2,opacity=0.3,line_color='green',editable=False),

             dict(x0='2020-04-22', x1='2020-04-22', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='green'),

             dict(x0='2020-05-04', x1='2020-05-04', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='green'),

             dict(x0='2020-05-17', x1='2020-05-17', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='green'),

             dict(x0='2020-03-30', x1='2020-03-30', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='green')],

    annotations=[dict(x='2020-03-23', y=0.54, xref='x', yref='paper',

                    showarrow=False, xanchor='right', text='Reliance support to Fight Against Coronavirus'),

                 dict(x='2020-04-22', y=0.5, xref='x', yref='paper',

                    showarrow=False, xanchor='left', text='Facebook invested in Jio Platforms '),

                dict(x='2020-05-04', y=0.08, xref='x', yref='paper',

                    showarrow=False, xanchor='left', text='Silver Lake invested in Jio Platforms'),

                 dict(x='2020-05-17', y=0.05, xref='x', yref='paper',

                    showarrow=False, xanchor='left', text='General Atlantic invested in Jio Platforms'),

                 dict(x='2020-03-30', y=0.09, xref='x', yref='paper',

                    showarrow=False, xanchor='right', text='Rs. 500 Crore Contribution to PM CARES Fund'),

               ]

)

fig.show()
reliance_stationarity=reliance_analysis[['Close']]



reliance_stationarity.plot()
test_result=adfuller(reliance_stationarity['Close'])



#Ho: Data is non stationary

#H1: Data is stationary



def adfuller_test(price):

    result=adfuller(price)

    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):

        print(label+' : '+str(value) )

    if result[1] <= 0.05:

        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")

    else:

        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")



adfuller_test(reliance_stationarity['Close'])
reliance_stationarity['Close First Difference']=reliance_stationarity['Close']-reliance_stationarity['Close'].shift(1)

reliance_stationarity['Close First Difference'].plot()
train = reliance_lag[reliance_lag.Date < "2019"]

valid = reliance_lag[reliance_lag.Date >= "2019"]
exogenous_features = ['High_mean_lag3','High_mean_lag7', 'High_mean_lag30', 'High_std_lag3', 'High_std_lag7',

       'High_std_lag30', 'Low_mean_lag3', 'Low_mean_lag7', 'Low_mean_lag30',

       'Low_std_lag3', 'Low_std_lag7', 'Low_std_lag30', 'Volume_mean_lag3',

       'Volume_mean_lag7', 'Volume_mean_lag30', 'Volume_std_lag3',

       'Volume_std_lag7', 'Volume_std_lag30', 'VWAP_mean_lag3',

       'VWAP_mean_lag7', 'VWAP_mean_lag30', 'VWAP_std_lag3', 'VWAP_std_lag7',

       'VWAP_std_lag30','Month', 'Week', 'Day', 'Day of week']
model = auto_arima(train.Close, exogenous=train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)

model.fit(train.Close, exogenous=train[exogenous_features])



valid["Forecast_ARIMAX"] = model.predict(n_periods=len(valid), exogenous=valid[exogenous_features])
valid[["Close", "Forecast_ARIMAX"]].plot()
print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(valid.Close, valid.Forecast_ARIMAX)))

print("\nMAE of Auto ARIMAX:", mean_absolute_error(valid.Close, valid.Forecast_ARIMAX))
### intiialize the Model

model=Prophet()



#Fitting the model and renaming the columns based on prophe requirements

model.fit(reliance_analysis[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}))



#Making future dataframe for forecasting, we have given 365 days which can calculate VWAP till 2021

reliance_future=model.make_future_dataframe(periods=365)



#Checking the future dates

reliance_future.tail()
### Prediction of future values

reliance_prediction=model.predict(reliance_future)



reliance_prediction.tail()
#Forecast plot

model.plot(reliance_prediction)
#Forecast components

model.plot_components(reliance_prediction)
#Cross validation for the parameter days

reliance_cv=cross_validation(model,initial='1095 days',period='180 days',horizon="365 days")
#Checking the parameters

reliance_performance=performance_metrics(reliance_cv)

reliance_performance.head()





#Plotting for root mean squared metric

fig=plot_cross_validation_metric(reliance_cv,metric='rmse')