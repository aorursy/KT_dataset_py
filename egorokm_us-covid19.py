# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

import plotly.graph_objects as go

import warnings

import datetime

import math

from scipy.optimize import minimize



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ca_test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

ca_train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

ca_submission = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')
ca_test.tail()
df = ca_train

df = df.groupby(['Date','Country/Region']).sum().reset_index()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

df = df.set_index('Date')[['ConfirmedCases']]

df_comp = df.copy()
df_comp.head()
def model(N, a, alpha, t0, t):

    """Инициализируем модель."""

    return N * (1 - math.e ** (-a * (t-t0))) ** alpha



def model_loss(params):

    "Считаем потери."

    N, a, alpha, t0 = params

    global df

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t0, t) - df.iloc[t, 0]) ** 2

    return r 

try:

    N = df['Confirmed'][-1]

    T = -df['Confirmed'][0]

except:

    N = 10000

    T = 0
# оптимизируем значения

days_forecast = 60

opt = minimize(model_loss, x0=np.array([N, 0.1, 5, T]), method='Nelder-Mead', tol=1e-6).x



x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

x_actual =list(x_actual)

y_actual = list(df.reset_index().iloc[:,1])



start_date = pd.to_datetime(df.index[0])



x_model = []

y_model = []



# получим значения модели для того же временного ряда, что и фактические

for t in range(len(df) + days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))
country = "US"

state = "California"

fig = go.Figure()

fig.update_layout(title=country + ' - ' + state,

                  xaxis_title='Дата',

                  yaxis_title="Количество людей",

                  autosize=False,

                  width=700,

                  height=500,

                 )



fig.add_trace(go.Line(x=x_actual,

                      y=y_actual,

                      mode='markers',

                      name='Actual',

                      marker=dict(symbol='circle-open-dot', 

                                  size=9, 

                                  color='yellow', 

                                  line_width=1.5,)))    



fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Прогноз",

                      line=dict(color='blue', 

                                width=2.5)))
# Сравнение с данными тестового набора

df = ca_train

df = df.groupby(['Date','Country/Region']).sum().reset_index()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

df = df.set_index('Date')[['ConfirmedCases']]

df_comp = df.copy()



# define the models to forecast the growth of cases

def model(N, a, alpha, t0, t):

    return N * (1 - math.e ** (-a * (t-t0))) ** alpha



def model_loss(params):

    N, a, alpha, t0 = params

    global df

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t0, t) - df.iloc[t, 0]) ** 2

    return r 

try:

    N = df['ConfirmedCases'][-1]

    T = -df['ConfirmedCases'][0]

except:

    N = 10000

    T = 0



opt = minimize(model_loss, x0=np.array([N, 0.1, 5, T]), method='Nelder-Mead', tol=1e-6).x

print(opt)



# create series to be plotted 

x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

x_actual =list(x_actual)

y_actual = list(df.reset_index().iloc[:,1])



start_date = pd.to_datetime(df.index[0])

days_forecast = len(df)+len(ca_test)-7

x_model = []

y_model = []



# get the model values for the same time series as the actuals

for t in range(days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))



fig = go.Figure()

fig.update_layout(title=country + ' - ' + state,

                  xaxis_title='Дата',

                  yaxis_title="Количество людей",

                  autosize=False,

                  width=700,

                  height=500,)



fig.add_trace(go.Line(x=x_actual,

                      y=y_actual,

                      mode='markers',

                      name='Actual',

                      marker=dict(symbol='circle-open-dot', 

                                  size=9, 

                                  color='yellow', 

                                  line_width=1.5,)))    

fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Прогноз",

                      line=dict(color='blue', 

                                width=2.5))) 



df2 = pd.DataFrame(y_model,index=pd.to_datetime(x_model),columns=['ConfirmedCases'])

df2.index.name = 'Date'

df_comp = df.rename(columns={'ConfirmedCases': 'Actuals'})

df_comp = pd.merge(df_comp,

                     df2,

                     how='outer',

                     left_on=['Date'],

                     right_on=['Date'])

df_comp = df_comp[['ConfirmedCases']]





# define the models to forecast the growth of cases

def model(N, a, alpha, t):

    return N * (1 - math.e ** (-a * (t))) ** alpha



def model_loss(params):

    N, a, alpha = params

    global df

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2

    return r 



try:

    N = df['ConfirmedCases'][-1]

except:

    N = 10000



opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-6).x

print(opt)



try:

    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(days_forecast):

        x_model.append(start_date + datetime.timedelta(days=t))

        y_model.append(round(model(*opt,t)))





    # now plot the new series

    fig.add_trace(go.Line(x=x_model,

                          y=y_model,

                          mode='lines',

                          name="Prediction without offset",

                          line=dict(color='Red', 

                                    width=1.5,

                                    dash='dot'

                                   )

                         ) 

                 )

    

except:

    pass



fig.show()  
df = ca_train

df = df.groupby(['Date','Country/Region']).sum().reset_index()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

df = df.set_index('Date')[['Fatalities']]

def model(N, a, alpha, t):

    return N * (1 - math.e ** (-a * (t))) ** alpha



def model_loss(params):

    N, a, alpha = params

    global df

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2

    return r 



try:

    N = df['ConfirmedCases'][-1]

except:

    N = 10000



opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-6).x

print(opt)



x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

x_actual =list(x_actual)

y_actual = list(df.reset_index().iloc[:,1])



start_date = pd.to_datetime(df.index[0])

days_forecast = len(df)+len(ca_test)-7

x_model = []

y_model = []



for t in range(days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))



fig = go.Figure()

fig.update_layout(title=country + ' - ' + state,

                  xaxis_title='Дата',

                  yaxis_title="Количество людей",

                  autosize=False,

                  width=700,

                  height=500,)



fig.add_trace(go.Line(x=x_actual,

                      y=y_actual,

                      mode='markers',

                      name='Actual',

                      marker=dict(symbol='circle-open-dot', 

                                  size=9, 

                                  color='black', 

                                  line_width=1.5,)))    



fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Прогноз",

                      line=dict(color='blue', 

                                width=2.5))) 





df2 = pd.DataFrame(y_model,index=pd.to_datetime(x_model),columns=['Fatalities'])

df2.index.name = 'Date'

df_comp = pd.merge(df_comp,

                     df2,

                     how='outer',

                     left_on=['Date'],

                     right_on=['Date'])



def model(N, a, alpha, t):

    return N * (1 - math.e ** (-a * (t))) ** alpha



def model_loss(params):

    N, a, alpha = params

    global df

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2

    return r 



try:

    N = df['ConfirmedCases'][-1]

except:

    N = 10000



opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-6).x

print(opt)



try:

    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(days_forecast):

        x_model.append(start_date + datetime.timedelta(days=t))

        y_model.append(round(model(*opt,t)))





    # now plot the new series

    fig.add_trace(go.Line(x=x_model,

                          y=y_model,

                          mode='lines',

                          name="Prediction without offset",

                          line=dict(color='Red', 

                                    width=1.5,

                                    dash='dot')))

    

except:

    pass

fig.show()
df2 = pd.DataFrame(y_model,index=pd.to_datetime(x_model),columns=['Fatalities'])

df2.index.name = 'Date'

df_comp = pd.merge(df_comp,

                     df2,

                     how='outer',

                     left_on=['Date'],

                     right_on=['Date'])



def model(N, a, alpha, t):

    return N * (1 - math.e ** (-a * (t))) ** alpha



def model_loss(params):

    N, a, alpha = params

    global df

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2

    return r 



try:

    N = df['ConfirmedCases'][-1]

except:

    N = 10000



opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-6).x

print(opt)



try:

    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    for t in range(days_forecast):

        x_model.append(start_date + datetime.timedelta(days=t))

        y_model.append(round(model(*opt,t)))





    # now plot the new series

    fig.add_trace(go.Line(x=x_model,

                          y=y_model,

                          mode='lines',

                          name="Прогноз",

                          line=dict(color='Red', 

                                    width=1.5,

                                    dash='dot')))

except:

    pass



fig.show()
df_comp = df_comp.reset_index()

df_comp.tail()
ca_test.head()
ca_test['Date'] = pd.to_datetime(ca_test['Date'])
ca_test.info()
df_sub = pd.merge(ca_test,

                  df_comp,

                  how='left',

                  on=['Date']

                 )
df_sub.tail()
df_sub.drop('Fatalities_x', axis=1, inplace=True)
df_sub = df_sub.rename(columns={'Fatalities_y': 'Fatalities'})
df_sub = df_sub[['ForecastId','ConfirmedCases','Fatalities']]

df_sub.tail()
ca_submission.tail()
df_sub.to_csv('submission.csv',index=False)
df_sub