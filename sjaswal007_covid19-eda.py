# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
%matplotlib inline
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
#init_notebook_mode{connected=True}
import folium 
from folium import plugins
from tqdm.notebook import tqdm as tqdm
import datetime
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
cleaned_data=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
cleaned_data.head(15)
# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = confirmed - deaths - recovered
cleaned_data['Active'] = cleaned_data['Confirmed'] - cleaned_data['Deaths'] - cleaned_data['Recovered']

# filling missing values 
cleaned_data[['Province/State']] = cleaned_data[['Province/State']].fillna('')
cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.head()
Date_df=cleaned_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
Date_df=Date_df.sort_values(by=['Confirmed'])
Date_df.head(40)
temp = Date_df.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='case', value_name='count')
fig = px.area(temp, x="Date", y="count", color='case',
             title='Cases over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
fig.update_xaxes(tick0=4, dtick=4)
fig.show()
for column in Date_df[['Confirmed']]:
   # Select column contents by column name using [] operator
   columnSeriesObj = Date_df[column]
#    print('Colunm Name : ', column)
#    print('Column Contents : ', columnSeriesObj.values)
   new_cases=columnSeriesObj.values
my_list=[]
for i in range(0,len(new_cases)-1):
    new_value=new_cases[i+1]-new_cases[i]
    my_list.append(new_value)
my_list.insert(0,0)
print(len(my_list))
print(Date_df.shape)
Date_df['New_cases'] = np.array(my_list)

import plotly.express as px
fig = go.Figure()
fig.add_trace(go.Bar(
    x=Date_df['Date'],
    y=Date_df['New_cases'],
    name='New_cases',
    marker_color='grey'
))
fig.show()
for column in Date_df[['Recovered']]:
   # Select column contents by column name using [] operator
   columnSeriesObj = Date_df[column]
#    print('Colunm Name : ', column)
#    print('Column Contents : ', columnSeriesObj.values)
   new_recover=columnSeriesObj.values
my_list1=[]
for i in range(0,len(new_recover)-1):
    new_value=new_recover[i+1]-new_recover[i]
    my_list1.append(new_value)
my_list1.insert(0,0)
print(len(my_list1))
print(Date_df.shape)
Date_df['New_recover'] = np.array(my_list1)
Date_df.head(5)
fig = go.Figure()
fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['New_cases'],
                    mode='lines+markers',
                    name='New_cases',line=dict(color='red')))
fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['New_recover'],
                    mode='lines+markers',
                    name='New_recover',line=dict(color='green')))
fig.update_xaxes(tick0=4, dtick=4)
fig.show()
Date_df['Closed_cases']=Date_df['Recovered']+Date_df['Deaths']
Date_df['Recover_percent']=(Date_df['Recovered']/Date_df['Closed_cases'])*100
Date_df['Death_percent']=(Date_df['Deaths']/Date_df['Closed_cases'])*100
Date_df.head(5)

fig = go.Figure()
fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['Recover_percent'],
                    mode='lines+markers',
                    name='Recover %',line=dict(color='green')))
fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['Death_percent'],
                    mode='lines+markers',
                    name='Death %',line=dict(color='red')))
fig.update_xaxes(tick0=4, dtick=4)
fig.update_layout(title='Output of Closed Cases(Recovery OR Death)',
                   xaxis_title='Date',
                   yaxis_title='Percentage')
fig.show()
new_df=cleaned_data.loc[cleaned_data['Country/Region'] == 'China'] 
# v=cleaned_data.groupby('Country/Region')['Active'].sum().reset_index()
# v = v.melt(id_vars="Country/Region", value_vars=['Active'],
#                  var_name='case', value_name='count')
new_df1=cleaned_data.loc[cleaned_data['Date'] == '4/7/20'] 
res_df=new_df1.groupby('Country/Region')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
fig = px.pie(res_df, values='Confirmed', names='Country/Region', title='Countries Cases Distribution')
fig.show()
top_10=res_df.sort_values(by=['Confirmed'],ascending=False)[0:10]
import plotly.express as px
fig = px.bar(top_10, x='Country/Region', y='Confirmed')
fig.show()

top_10_death=res_df.sort_values(by=['Deaths'],ascending=False)[0:10]
top_10_death.head(5)
import plotly.express as px
fig = px.bar(top_10_death, x='Country/Region', y='Deaths')
fig.show()
us_data=cleaned_data.loc[cleaned_data['Country/Region']=='US']
us_data=us_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
us_data=us_data.sort_values(by=['Confirmed'])


Spain_data=cleaned_data.loc[cleaned_data['Country/Region']=='Spain']
Spain_data=Spain_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
Spain_data=Spain_data.sort_values(by=['Confirmed'])



Italy_data=cleaned_data.loc[cleaned_data['Country/Region']=='Italy']
Italy_data=Italy_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
Italy_data=Italy_data.sort_values(by=['Confirmed'])



France_data=cleaned_data.loc[cleaned_data['Country/Region']=='France']
France_data=France_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
France_data=France_data.sort_values(by=['Confirmed'])



Germany_data=cleaned_data.loc[cleaned_data['Country/Region']=='Germany']
Germany_data=Germany_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
Germany_data=Germany_data.sort_values(by=['Confirmed'])


fig = go.Figure()
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=us_data['Confirmed'][30:],
                    mode='markers',
                    name='US'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Spain_data['Confirmed'][30:],
                    mode='markers',
                    name='Spain'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Italy_data['Confirmed'][30:],
                    mode='markers',
                    name='Italy'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=France_data['Confirmed'][30:],
                    mode='markers',
                    name='France'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Germany_data['Confirmed'][30:],
                    mode='markers',
                    name='Germany'))
fig.update_xaxes(tick0=4, dtick=4)
fig.update_layout(height=500, width=1000, title_text="Case progrssion of top countries")
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=us_data['Deaths'][30:],
                    mode='markers',
                    name='US'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Spain_data['Deaths'][30:],
                    mode='markers',
                    name='Spain'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Italy_data['Deaths'][30:],
                    mode='markers',
                    name='Italy'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=France_data['Deaths'][30:],
                    mode='markers',
                    name='France'))
fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Germany_data['Deaths'][30:],
                    mode='markers',
                    name='Germany'))
fig.update_xaxes(tick0=4, dtick=4)

fig.update_layout(height=500, width=1000, title_text="Death progrssion of top countries")
fig.show()

train_dataset = pd.read_csv('../input/timeseries-confirmedcovid19/old/time_series_covid_19_confirmed_global.csv')
drop_clo = ['Province/State','Country/Region','Lat','Long']
train_dataset=train_dataset.drop(drop_clo,axis=1)
datewise= list(train_dataset.columns)
val_dataset = train_dataset[datewise[-15:]]
date_array=np.asarray(Date_df['Date'])
fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[0].values[:-15], marker=dict(color="dodgerblue"), showlegend=False,),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=val_dataset.loc[0].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False,),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[1].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=val_dataset.loc[1].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[2].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=val_dataset.loc[2].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Train (blue) vs. Validation (orange) sales")
fig.show()
predictions = []
for i in range(len(val_dataset.columns)):
    if i == 0:
        predictions.append(train_dataset[train_dataset.columns[-16]].values)
    else:
        predictions.append(val_dataset[val_dataset.columns[i-1]].values)
    
predictions = np.transpose(np.array([row.tolist() for row in predictions]))
error_naive = np.linalg.norm(predictions[:] - val_dataset.values[:])/len(predictions[:])
pred_1 = predictions[0]
pred_2 = predictions[1]
pred_3 = predictions[2]

fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[0].values[:-15], marker=dict(color="dodgerblue"),name="Train"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=val_dataset.loc[0].values, mode='lines+markers', marker=dict(color="darkorange"), name="Validation"),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=pred_1, mode='lines', marker=dict(color="seagreen"),
               name="Pred"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[1].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=val_dataset.loc[1].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),
    row=2, col=1
)


fig.add_trace(
    go.Scatter(x=date_array[-15:], y=pred_2, mode='lines', marker=dict(color="seagreen"), showlegend=False,
               name="Denoised signal"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[2].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=val_dataset.loc[2].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=date_array[-15:], y=pred_3, mode='lines', marker=dict(color="seagreen"), showlegend=False,
               name="Denoised signal"),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Naive approach")
fig.show()
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(predictions[:] ,val_dataset.values[:]))
print(rms)
model_train=Date_df.iloc[:int(Date_df.shape[0]*0.90)]
valid=Date_df.iloc[int(Date_df.shape[0]*0.90):]
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=0.2, smoothing_slope=0.8)
y_pred=valid.copy()
import matplotlib.pyplot as plt
y_pred["Holt"]=holt.forecast(len(valid))
#model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))
print("Root Mean Square Error Holt's Linear Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))
fig = go.Figure()
fig.add_trace(go.Scatter(x=Date_df['Date'], y=model_train['Confirmed'],
                    mode='lines+markers',
                    name='Train ',line=dict(color='green')))
fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Confirmed'],
                    mode='lines+markers',
                    name='validation ',line=dict(color='red')))
fig.add_trace(go.Scatter(x=valid['Date'], y=y_pred["Holt"],
                    mode='lines+markers',
                    name='predicted ',line=dict(color='white')))
fig.update_xaxes(tick0=4, dtick=4)
fig.update_layout(title='Holt Linear Model',
                   xaxis_title='Date',
                   yaxis_title='Percentage')
fig.show()
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline 
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-09-2020.csv')
cols = confirmed_df.keys()
dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 

china_cases = [] 
italy_cases = []
us_cases = [] 
spain_cases = [] 
france_cases = [] 

china_deaths = [] 
italy_deaths = []
us_deaths = [] 
spain_deaths = [] 
france_deaths = [] 

china_recoveries = [] 
italy_recoveries = []
us_recoveries = [] 
spain_recoveries = [] 
france_recoveries = [] 

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)
    
    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    # case studies 
    china_cases.append(confirmed_df[confirmed_df['Country/Region']=='China'][i].sum())
    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())
    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())
    spain_cases.append(confirmed_df[confirmed_df['Country/Region']=='Spain'][i].sum())
    france_cases.append(confirmed_df[confirmed_df['Country/Region']=='France'][i].sum())
    
    china_deaths.append(deaths_df[deaths_df['Country/Region']=='China'][i].sum())
    italy_deaths.append(deaths_df[deaths_df['Country/Region']=='Italy'][i].sum())
    us_deaths.append(deaths_df[deaths_df['Country/Region']=='US'][i].sum())
    spain_deaths.append(deaths_df[deaths_df['Country/Region']=='Spain'][i].sum())
    france_deaths.append(deaths_df[deaths_df['Country/Region']=='France'][i].sum())
    
    china_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='China'][i].sum())
    italy_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Italy'][i].sum())
    us_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='US'][i].sum())
    spain_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Spain'][i].sum())
    france_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='France'][i].sum())
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.05, shuffle=False) 
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)
bayesian_search.best_params_
bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'Polynomial Bayesian Ridge Regression Predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Bayesian Ridge Predicted # of Confirmed Cases Worldwide': np.round(bayesian_pred[-10:])})
svm_df


us_data=cleaned_data.loc[cleaned_data['Country/Region']=='US']
us_data=us_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
us_data=us_data.sort_values(by=['Confirmed'])

state_data=cleaned_data.loc[cleaned_data['Country/Region']=='US']
state_data.head(5)
fig = go.Figure()
fig.add_trace(go.Scatter(x=us_data['Date'], y=us_data['Active'],
                    mode='lines+markers',
                    name='Active'))
fig.add_trace(go.Scatter(x=us_data['Date'], y=us_data['Deaths'],
                    mode='lines+markers',
                    name='Deaths'))
fig.update_xaxes(tick0=4, dtick=4)
fig.show()

new_df=cleaned_data.loc[cleaned_data['Country/Region'] == 'China'] 
# v=cleaned_data.groupby('Country/Region')['Active'].sum().reset_index()
# v = v.melt(id_vars="Country/Region", value_vars=['Active'],
#                  var_name='case', value_name='count')
new_df1=cleaned_data.loc[cleaned_data['Date'] == '4/7/20'] 
res_df=new_df1.groupby('Country/Region')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
new_df1=cleaned_data.loc[cleaned_data['Date'] == '4/7/20'] 
res_df=new_df1.groupby('Country/Region')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

import plotly.graph_objects as go



fig = go.Figure()
fig.add_trace(go.Bar(
    x=top_10['Country/Region'],
    y=top_10['Confirmed'],
    name='Confirmed',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=top_10['Country/Region'],
    y=top_10['Active'],
    name='Active',
    marker_color='lightsalmon'
))

fig.add_trace(go.Bar(
    x=top_10['Country/Region'],
    y=top_10['Deaths'],
    name='Deaths',
    marker_color='red'
))
fig.add_trace(go.Bar(
    x=top_10['Country/Region'],
    y=top_10['Recovered'],
    name='Recovered',
    marker_color='green'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()
import plotly.express as px
fig = px.bar(top_10, x="sex", y="total_bill", color="smoker", barmode="group",
             facet_row="time", facet_col="day",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],
                              "time": ["Lunch", "Dinner"]})
fig.show()