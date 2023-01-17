import pandas as pd
import numpy as np
import io
import requests

import cufflinks as cf
cf.go_offline()
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
# india_covid_dataset = pd.read_csv('https://raw.githubusercontent.com/dhruvipatel14/Covid-2019-India-Analysis/master/datasets_549966_1189152_complete.csv',parse_dates=['Date'])
india_covid_dataset = pd.read_csv('../input/datasets_549966_1189152_complete.csv',parse_dates=['Date'])
covid_df = pd.DataFrame(india_covid_dataset)
covid_df.head()
daily_covid = pd.read_csv('../input/datasets_549966_1189152_nation_level_daily.csv')
daily_df = pd.DataFrame(daily_covid)
daily_df.tail()
state_level= pd.read_csv('../input/datasets_549966_1189152_state_level_latest.csv')
state_df = pd.DataFrame(state_level)
state_df.head()
final_state_df = state_df.drop([0])
confirm = final_state_df['confirmed'].values.sum()
print(f'Total number of Confirmed COVID 2019 cases across India:', confirm)
print(f'Total number of Active COVID 2019 cases across India:', final_state_df['active'].sum())
print(f'Total number of recovered COVID 2019 cases across India:', final_state_df['recovered'].sum())
print(f'Total number of Deaths due to COVID 2019  across India:', final_state_df['deaths'].sum())
print(f'Total number of States/UTs affected:', final_state_df['state'].nunique())
state_active = final_state_df.groupby('state')['active'].sum().sort_values(ascending=False).to_frame()
state_active.style.background_gradient(cmap='Reds')
state_active_df = final_state_df.groupby('state')['active'].sum().sort_values(ascending=True).reset_index()
data = px.data.gapminder()

fig = px.bar(state_active_df, x='active', y='state',title='Total Active Cases', 
             text='active', 
             orientation='h', 
             height=1000,
             color='active',
             range_x = [0, max(state_active_df['active'])]
            )                                                
fig.show()
state_conf_active = final_state_df.groupby('state')['active','confirmed','deaths'].sum().sort_values(ascending=True,by='active').reset_index()


fig = go.Figure(data=[
    go.Bar(name='Confirmed', y=final_state_df['confirmed'], x=final_state_df['state']),
    go.Bar(name='Active', y=final_state_df['active'], x=final_state_df['state'])
])
# Change the bar mode
fig.update_layout(barmode='group',title = 'Comparision of confimed VS Active cases')
fig.show()
daily_df['date'] = pd.date_range(start= '2020-01-30', end = '2020-05-25', freq='D')
daily_df.head()
new_daily  = daily_df.copy()
new_daily = new_daily.set_index('date')
new_daily.head()
fig = go.Figure()
fig.add_trace(go.Scatter(x=new_daily.index, y=new_daily['totalconfirmed'],
                    mode='lines+markers',name='Total Cases'))

fig.add_trace(go.Scatter(x=new_daily.index, y=new_daily['dailyconfirmed'], 
                mode='lines',name='New Cases'))
fig.add_trace(go.Scatter(x=new_daily.index, y=new_daily['dailyrecovered'], 
                mode='lines',name='Daily recovered'))
        
    
fig.update_layout(title_text='Trend of Coronavirus Cases on daily basis')

fig.show()
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import statsmodels.api as sm
import warnings
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf( new_daily['totalconfirmed'].values.squeeze(), lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(new_daily['totalconfirmed'], lags=20, ax=ax2)
def gridSearch(itemObj):
    # Define the p, d and q parameters to take any value between 0 and 3
    p = d = q = range(1, 5)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]

    bestAIC = np.inf
    bestParam = None
    bestSParam = None
    
    print('Running GridSearch')
    
    #use gridsearch to look for optimial arima parameters
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(itemObj,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                #if current run of AIC is better than the best one so far, overwrite it
                if results.aic<bestAIC:
                    if results.aic >= 1:
                        bestAIC = results.aic
                        bestParam = param
                        bestSParam = param_seasonal

            except:
                continue
                
    print('the best ones are:',bestAIC,bestParam,bestSParam)
    return bestAIC,bestParam,bestSParam
import warnings
warnings.filterwarnings("ignore")

# bestAIC,bestParam,bestSParam = gridSearch(new_daily['totalconfirmed'])
bestAIC = 94036.38
order = (2,1,2)
seasonal_order = (2,1,2,12)
mod = sm.tsa.statespace.SARIMAX(new_daily['totalconfirmed'],
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

results = mod.fit()
print(results.summary())
forecast = results.get_forecast(steps=20)

# Get confidence intervals of forecasts
pred_ci = forecast.conf_int()

forecast_df = pd.DataFrame(forecast.predicted_mean,columns=['Predicted Confirmed cases'])

forecast_plot = pd.concat([new_daily['totalconfirmed'],forecast_df,pred_ci])


layout = go.Layout(title = "Prediction of covid-2019 for next 20 days",
     xaxis=dict(title='Date'),
     yaxis=dict(title="Total confirmed cases"))
fig = forecast_plot.iplot(kind = "scatter",  layout = layout)