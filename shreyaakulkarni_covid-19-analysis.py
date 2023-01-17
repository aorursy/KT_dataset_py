# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#@title Map : Daily Spread of Coronavirus

import pandas as pd

import plotly.express as px

import numpy as np

df=pd.read_csv("../input/covid19/owid-covid-data (16).csv")

df1=df[df["location"].isin(["India","Italy","Australia",

                            "Canada","United States"])]

fig = px.choropleth(df, 

                    locations="location", 

                    locationmode = "country names",

                    color="new_cases", 

                    hover_name="location", 

                    animation_frame="date"

                   )



fig.update_layout(

    title_text = 'Daily Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
#@title Graph : stringency_index





fig = px.line(df1, x="date", y="stringency_index", color='location',

              title='stringency_index')

fig.show()
fig = px.line(df1, x="date", y="new_cases_per_million", color='location',

              title='No of cases per million')

fig.show()
fig = px.line(df1, x="date", y="new_cases", color='location',

              title='New Cases')

fig.show()
#@title Grath : Death_rate

df1['death_rate']=(df1['total_deaths']/df1['total_cases'])*100

fig = px.line(df1, x="date", y="death_rate", color='location',

              title='Death Rate')

fig.show()
#@title Pie Chart : Percentage of total cases

fig = px.pie(df1, values = 'new_cases',names='location', height=600,

             title = 'Percentage of total cases')

fig.update_traces(textposition='inside', textinfo='percent+label')



fig.update_layout(

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



fig.show()
#@title Graph : Daily New_Cases of covid19 in India



y=df[df["location"].isin(["India"])][['new_cases','date']]



train=y[:(len(y['new_cases'])-10)]

valid=y[(len(y['new_cases'])-10):]



fig = px.line(y, x="date", y="new_cases",

              title='New_Cases of covid19 in India')

fig.show()
#@title Forecasting new cases in india for next 10 days using Arima Model

from statsmodels.tsa.arima_model import ARIMA



model = ARIMA(train['new_cases'],order=(8,1,1))

#model.fit(train['new_cases'])

model_fit=model.fit()







#forecast1 = model_fit.predict(start=(len(y['new_cases'])-10),end=len(y['new_cases'])-1)

forecast1 = np.round_(model_fit.forecast(steps =10)[0])

print(forecast1)
forecast = pd.DataFrame()

forecast['date']=valid['date']

forecast['new_cases']=(list(forecast1))



forecast2 = pd.DataFrame()

forecast2['date']=valid['date']

forecast2['new_cases(actual)']=valid['new_cases']

forecast2['new_cases(forecasted)']=(list(forecast1))

forecast2
import matplotlib.pyplot as plt

plt.plot(valid['date'],valid['new_cases'])

plt.plot(forecast['date'],forecast['new_cases'],color='red')
#@title Display Forecased and Actual Value using line plot

import plotly.graph_objs as go

import matplotlib. pyplot as plt

#plot the predictions for validation set





train['Line']=np.repeat("Data", len(train['new_cases']))



forecast['Line']=np.repeat("Forecasted", len(forecast['new_cases']))



valid['Line']=np.repeat("Actual", len(valid['new_cases']))







df3=valid.append(forecast, ignore_index=True)

fig = px.line(df3, x="date", y="new_cases", color='Line',

              title='forecasted new cases of covid19')



fig1 = go.Figure(fig.add_traces(

                 data=px.line(train, x='date', y='new_cases')._data,))



fig1.show()
#@title Accuracy & Error

#calculate rmse

from math import sqrt

from sklearn.metrics import mean_squared_error





ac=[]

for i in range(1,len(valid)):

  ac.append(min(forecast['new_cases'].iloc[i],valid['new_cases'].iloc[i])/max(forecast['new_cases'].iloc[i],valid['new_cases'].iloc[i]))



print("Min-Max accuracy of model is : ",np.mean(ac)*100)



rms = sqrt(mean_squared_error(valid['new_cases'],forecast['new_cases']))

print("Error: ",rms)
#@title Forecasting for next 15 days









model = ARIMA(y['new_cases'],order=(8,2,2))

model_fit=model.fit()

forecast1 = np.round_(model_fit.forecast(steps =15)[0])



forecast = pd.DataFrame()

#forecast=pd.DataFrame()

date=pd.date_range(start=y['date'].iloc[-1], periods=16, freq='D')

forecast['date']=date[1:]

forecast['new_cases']=list(forecast1)

forecast
#@title Forecased and Actual Value using line plot



y['Line']=np.repeat("Actual", len(y['new_cases']))

forecast['Line']=np.repeat("Forecasted", 15)



df3=y.append(forecast, ignore_index=True)



fig = px.line(df3, x="date", y="new_cases", color='Line',

              title='forecasted new cases of covid19')



fig1 = go.Figure(fig.add_traces(

                 data=px.line(train, x='date', y='new_cases')._data,))



fig1.show()