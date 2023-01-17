# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
test_d=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
test_d
train_d=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
train_d
train_d['Province_State'].unique()
test_d['Province_State'].unique()
countryorstates=train_d['Country_Region'].unique()
countryorstates
train_d['Country_Region'].value_counts()
column=train_d.keys()
print(column)
fatality=train_d.loc[:,column[5]:column[-1]]
fatality
confirmed_cases=train_d.loc[:,column[4]:column[-2]]
confirmed_cases
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import datetime
import time
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objs as go

grp=train_d.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()
grp=grp.melt(id_vars="Date",value_vars=["Fatalities","ConfirmedCases"],var_name='Case',value_name='Count')
fig=px.area(grp,x='Date',y='Count',color='Case',height=600,title='Global: Cases over time')
fig.show()
#Globally: Confirmed cases and fatalities since March
combined=train_d.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()
fatal_date=combined.sort_values(by='Fatalities',ascending=False)
fatal_date=fatal_date.reset_index(drop=True)
fatal_date.style.background_gradient(cmap='Reds')
'Total number of active COVID 2019 cases across US'
df_usa = train_d.query("Country_Region=='US'")
US_cases = df_usa.groupby('Province_State', as_index=False)['ConfirmedCases', 'Fatalities'].sum()
US_cases.head()
px.area(US_cases, x='Province_State', y='ConfirmedCases', title='Confirmed Cases in US')
from fbprophet import Prophet
def fit_model(data_,interval_width_=0.95,periods_=10):
    data_.columns = ['ds', 'y']
    data_['ds'] = pd.to_datetime(data_['ds'])
    
    model = Prophet(interval_width=interval_width_)
    model.fit(data_)  
    return model

def predict(model,data_):
    data_=data_.rename(columns={'Date':'ds'})
    forecast = model.predict(data_)
    return forecast

def forecast_state(training_data,testing_data,state_name,interval_width=0.95):
    train_confirmed = training_data.groupby('Date').sum()['ConfirmedCases'].reset_index().copy()
    train_fatalities = training_data.groupby('Date').sum()['Fatalities'].reset_index().copy()
    
    model_confirmed=fit_model(train_confirmed)
    confirmed_predictions = predict(model_confirmed,testing_data[['Date']].copy())
    testing_data['ConfirmedCases']=confirmed_predictions['yhat'].astype(np.uint64).tolist()

    model_fatalities=fit_model(train_fatalities)
    fatalities_predictions = predict(model_fatalities,testing_data[['Date']].copy())
    testing_data['Fatalities']=fatalities_predictions['yhat'].astype(np.uint64).tolist()

    return testing_data
EMPTY_VAL = "EMPTY_VAL"

def fillState(Province_State, Country_Region):
    if Province_State == EMPTY_VAL: return Country_Region
    return Province_State

X_train=train_d.copy()
X_test=test_d.copy()
X_train['Province_State'].fillna(EMPTY_VAL, inplace=True)
X_test['Province_State'].fillna(EMPTY_VAL, inplace=True)
X_train['Province_State'] = X_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
X_test['Province_State'] = X_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
world_output=test_d.copy()
world_output['ConfirmedCases']=int(0)
world_output['Fatalities']=int(0)
count=0
total=X_train['Country_Region'].nunique()
for country,grp_country in X_train.groupby(['Country_Region']):
    country_output={}
    for state,grp_state in grp_country.groupby(['Province_State']):
        print(f'{count}/{total} : {country}\t{state}')
        state_test=X_test.loc[X_test.Province_State == state].copy()
        output=forecast_state(grp_state,state_test,state,0.95)
        world_output.update(output)
    count+=1
    world_output=world_output.astype({"ForecastId":int,"ConfirmedCases":int,"Fatalities":int})
    world_output[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)
df=pd.read_csv('submission.csv')
df