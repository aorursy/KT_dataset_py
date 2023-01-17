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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import os # accessing directory structure



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

import seaborn as sns; sns.set()

# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno

import numpy as np 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator

plt.style.use('seaborn')

%matplotlib inline

import seaborn as sns; sns.set()

# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)

filename = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"

confirmed_df = pd.read_csv(filename)

confirmed_df.head()
confirmed_df.iloc[:,4:]
countries = confirmed_df['Country/Region'].unique().tolist()

print(countries)



print("\nTotal countries affected by virus: ",len(countries))
# Creating a dataframe with total no of confirmed cases for every country

Number_of_countries = len(confirmed_df['Country/Region'].value_counts())





cases = pd.DataFrame(confirmed_df.groupby('Country/Region'))

cases['Country/Region'] = cases.index

cases.index=np.arange(1,Number_of_countries+1)



global_cases = cases[['Country/Region']]

global_cases


#Get Correlation between different variables

corr = confirmed_df.corr(method='kendall')

plt.figure(figsize=(18,12))

sns.heatmap(corr, annot=True)


countries=['India', 'Italy','Brazil', 'Canada', 'Germany']

y=confirmed_df.loc[confirmed_df['Country/Region']=='Italy'].iloc[0,4:]

s = pd.DataFrame({'Canada':y})

for c in countries:    

    s[c] = confirmed_df.loc[confirmed_df['Country/Region']==c].iloc[0,4:]

pyplot.plot(range(y.shape[0]), s)
s.tail(10)
for r in confirmed_df['Country/Region']:

    if r != 'China':

        pyplot.plot(range(len(confirmed_df.columns)-4), confirmed_df.loc[confirmed_df['Country/Region']==r].iloc[0,4:])

#         pyplot.legend()
def confirmed_US_case():

  confirmed_cases_US = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"

  df = pd.read_csv(confirmed_cases_US)

  df = df[df['Country/Region'] == "US"]

  df_new = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'])

  df_new.rename(columns={"variable":"Date","value":"confirmed_cases"},inplace=True)

  confirmed_per_day = df_new.groupby("Date")['confirmed_cases'].sum()

  confirmed_per_day = confirmed_per_day.reset_index()

  print(confirmed_per_day)

  confirmed_per_day = confirmed_per_day[['Date','confirmed_cases']]

  return confirmed_per_day



confirmed_cases = confirmed_US_case()
confirmed_cases.tail()


confirmed_cases.rename(columns={"Date":"ds","confirmed_cases":"y"},inplace=True)

confirmed_cases['ds'] = pd.to_datetime(confirmed_cases['ds'])

confirmed_cases.sort_values(by='ds',inplace=True)
#Plotting number of cases with day

plt_confirmed = confirmed_cases.reset_index()['y'].plot(title="#Confirmed Cases Vs Day");

plt_confirmed.set(xlabel="Date", ylabel="#Confirmed Cases");
#Doing train test split

X_train = confirmed_cases[:-4]

X_test = confirmed_cases[-4:]



X_test = X_test.set_index("ds")

X_test = X_test['y']
# Model Initialize

from fbprophet import Prophet

m = Prophet()

m.fit(X_train)

future_dates = m.make_future_dataframe(periods=7)

# Prediction

forecast =  m.predict(future_dates)

pd.plotting.register_matplotlib_converters()

ax = forecast.plot(x='ds',y='yhat',label='Predicted confirmed cases in US',legend=True,figsize=(12,8))

X_test.plot(y='y',label='Actual Confirmed case counts in US',legend=True)
filename = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv"

death_df = pd.read_csv(filename)

death_df.head()
filename = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv"

recovered_df = pd.read_csv(filename)

recovered_df.head()
dates = ['1/22/20', '1/23/20', '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', 

         '1/29/20', '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', 

         '2/5/20', '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', 

         '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',

         '2/20/20','2/21/20','2/22/20','2/23/20','2/24/20','2/25/20','2/26/20',

'2/27/20','2/28/20','2/29/20','3/1/20','3/2/20','3/3/20','3/4/20','3/5/20','3/6/20',

'3/7/20','3/8/20','3/9/20','3/10/20','3/11/20','3/12/20','3/13/20','3/14/20','3/15/20',

'3/16/20','3/17/20','3/18/20','3/19/20','3/20/20','3/21/20','3/22/20','3/23/20','3/24/20','3/25/20','3/26/20',

        '3/27/20', '3/28/20', '3/29/20', '3/30/20', '3/31/20', '4/1/20',

       '4/2/20', '4/3/20', '4/4/20', '4/5/20', '4/6/20', '4/7/20', '4/8/20',

       '4/9/20', '4/10/20']
conf_df_long = confirmed_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Confirmed')

conf_df_long

deaths_df_long = death_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Deaths')

deaths_df_long
recovered_df_long = recovered_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')

recovered_df_long


full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recovered_df_long['Recovered']], 

                       axis=1, sort=False)

full_table.head()
full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp = full_latest.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered'].max()

temp.style.background_gradient(cmap='Pastel1_r')
temp_case = full_latest_grouped[['Country/Region', 'Confirmed']]

temp_case = temp_case.sort_values(by='Confirmed', ascending=False)

temp_case = temp_case.reset_index(drop=True)

temp_case.head(10).style.background_gradient(cmap='Pastel1_r')
temp_deaths = full_latest_grouped[['Country/Region', 'Deaths']]

temp_deaths = temp_deaths.sort_values(by='Deaths', ascending=False)

temp_deaths = temp_deaths.reset_index(drop=True)

temp_deaths = temp_deaths[temp_deaths['Deaths']>0]

temp_deaths.style.background_gradient(cmap='Pastel1_r')
import plotly.express as px

def location(row):

    if row['Country/Region']=='Canada':

        if row['Province/State']=='Quebec':

            return 'Quebec'

        else:

            return 'Other Canadian Provinces'

    else:

        return 'Rest of the World'



temp = full_latest.copy()

temp['Region'] = temp.apply(location, axis=1)

temp = temp.groupby('Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp.melt(id_vars='Region', value_vars=['Confirmed', 'Deaths', 'Recovered'], 

                 var_name='Case', value_name='Count').sort_values('Count')

temp.head()



fig = px.bar(temp, y='Region', x='Count', color='Case', barmode='group', orientation='h',

             height=500, width=1000, text='Count', title='Quebec - Canada - World', 

             color_discrete_sequence= ['#EF553B', '#00CC96', '#636EFA'])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
full_latest.groupby('Date').sum()
confirmed = full_latest.groupby('Date').sum()['Confirmed'].reset_index()

deaths = full_latest.groupby('Date').sum()['Deaths'].reset_index()

recovered = full_latest.groupby('Date').sum()['Recovered'].reset_index()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(x=confirmed['Date'],

                y=confirmed['Confirmed'],

                name='Confirmed',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=deaths['Date'],

                y=deaths['Deaths'],

                name='Deaths',

                marker_color='Red'

                ))

fig.add_trace(go.Bar(x=recovered['Date'],

                y=recovered['Recovered'],

                name='Recovered',

                marker_color='Green'

                ))



fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered (Bar Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()