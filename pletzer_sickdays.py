import time
time.asctime()
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
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

# remove rows that have fewer than 100 cases (the SIR model is only valid for large S, I and N)
df_100 = df.loc[df.ConfirmedCases >= 100]

# combine country and region into a single column
df_100 = df_100.fillna('-')

df_100['country_state'] = df_100.Country_Region + ' ' + df_100.Province_State
import plotly.express as px
import plotly.graph_objects as go
fig = px.line(df_100, x='Date', y=df_100.ConfirmedCases, color='country_state')
fig.update_layout(yaxis_type='log')
fig.show()
# focus on a few countries and provinces/states
countries = ['Italy -', 'Spain -', 'Germany -', 'Iran -', 'Switzerland -', 'Korea, South -', 'Netherlands -',
      'Singapore -', 'France -', 'Austria -', 'New Zealand -', 'United Kingdom -',
      'US New York', 'US New Jersey', 'US Washington', 'US California', 'China Hubei', 'China Hong Kong']

fig = go.Figure()
for country in countries:
    d = df_100.loc[df_100.country_state == country]
    fig.add_trace(go.Scatter(x=d.Date, y=d.ConfirmedCases, mode='lines', name=country))
fig.update_layout(yaxis_type='log', yaxis_title='ConfirmedCases')
fig.show()
def getDailyIncrement(C):
    """
    Estimate the Delta C by taking second order accurate differences
    """
    
    dC = np.zeros(C.shape, np.float32)
    n = C.shape[0]

    if n >= 3:
        # need at least three points
        
        # in the interior, apply second order differencing
        dC[1:-1] = 0.5*(C[2:] - C[:-2])
        
        # extrapolate for the first and last pointds
        dC[0] = dC[1]
        dC[-1] = dC[-2]
    
    return dC

def addDailyIncrementColumn(df):
    """
    Add column with time derivative of the cummulative number of cases
    """
    # get the list of country/regions
    countries = df['Country_Region'].unique()
    
    # initialize
    df['dC'] = np.zeros(df.shape[0], np.float32)  # d C/dt
    
    for country in countries:
        
        df2 = df.loc[df.Country_Region == country]
        
        # get the list of states
        states = df2['Province_State'].unique()
        mskCountry = (df.Country_Region == country)
        
        if len(states) == 1:
            # no states, just one country
            C = df.loc[mskCountry, 'ConfirmedCases'].array
            dC = getDailyIncrement(C)
            df.loc[mskCountry, 'dC'] = dC
        else:
            # treat each state separately
            for state in states:
                msk = mskCountry & (df.Province_State == state)
                C = df.loc[msk, 'ConfirmedCases'].array
                dC = getDailyIncrement(C)
                df.loc[msk, 'dC'] = dC
  
# add "dC" (Delta C) column
addDailyIncrementColumn(df_100)
fig = go.Figure()
for country in countries:
    d = df_100.loc[df_100.country_state == country]
    fig.add_trace(go.Scatter(x=d.ConfirmedCases, y=d.dC/d.ConfirmedCases, mode='lines', name=country))
fig.update_layout(xaxis_title="C = # of confirmed cases", 
                  yaxis_title="Delta C/C = relative daily increase of confirmed cases",)
fig.show()
# extract beta and N*p from linear regression
from sklearn import linear_model
lm = linear_model.LinearRegression()
# choose gamma
gamma = 1/15.
res = {'country': [],
       'y-intercept': [],
       'slope': [],
       'beta': [],
       'N*p': []}
for country in countries:
    d = df_100.loc[df_100.country_state == country]
    x = d.ConfirmedCases.to_numpy()
    y = d.dC.to_numpy()/x
    model = lm.fit(x.reshape(-1, 1), y)
    res['country'].append(country)
    res['y-intercept'].append(lm.intercept_)
    res['slope'].append(lm.coef_[0])
    beta = lm.intercept_ + gamma
    res['beta'].append(beta)
    Np = - beta**2 / (2*lm.coef_[0])
    res['N*p'].append(Np)
df_res = pd.DataFrame(res)
df_res