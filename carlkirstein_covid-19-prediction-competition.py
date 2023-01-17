import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

import plotly.graph_objects as go

import warnings

import datetime

import math

from scipy.optimize import minimize

# Configure Jupyter Notebook

pd.set_option('display.max_columns', None) 

pd.set_option('display.max_rows', 500) 

pd.set_option('display.expand_frame_repr', False)

# pd.set_option('max_colwidth', -1)

display(HTML("<style>div.output_scroll { height: 35em; }</style>"))



%matplotlib inline

%config InlineBackend.figure_format ='retina'



warnings.filterwarnings('ignore')
# the number of days into the future for the forecast

days_forecast = 30
# download the latest data sets

conf_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

recv_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
# create full table

dates = conf_df.columns[4:]



conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Confirmed')



deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Deaths')



recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')



full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 

                       axis=1, sort=False)

# avoid double counting

full_table = full_table[full_table['Province/State'].str.contains(',')!=True]
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')
# Display the number cases globally

df = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()

df = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

df =  df[df['Date']==max(df['Date'])].reset_index(drop=True)

df
# count the number cases per country

df = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

df = df.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

df = df.sort_values(by='Confirmed', ascending=False)

df = df.reset_index(drop=True)

df.style.background_gradient(cmap='coolwarm')
country = 'US'

cluster = 'California'



df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]

df = df.groupby(['Date','Country/Region']).sum().reset_index()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

df = df.set_index('Date')[['Confirmed']]

df_result = df.copy()

# df_result = df_result[['Date','Confirmed']]



# ensure that the model starts from when the first case is detected

# NOTE: its better not to truncate the dataset like this 

# df = df[df[df.columns[0]]>0]



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

    N = df['Confirmed'][-1]

    T = -df['Confirmed'][0]

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



x_model = []

y_model = []



# get the model values for the same time series as the actuals

for t in range(len(df) + days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))



# instantiate the figure and add the two series - actual vs modelled    

fig = go.Figure()

fig.update_layout(title=country + ' - ' + cluster,

                  xaxis_title='Date',

                  yaxis_title="nr People",

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

                                  color='black', 

                                  line_width=1.5,

                                 )

                     ) 

             )    



fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Prediction with offset",

                      line=dict(color='blue', 

                                width=2.5

                               )

                     ) 

             ) 



# now add the results of the model to the dataframe

df2 = pd.DataFrame(y_model,index=x_model,columns=['Offset'])

df2.index.name = 'Date'

df_result = pd.merge(df_result,

                     df2,

                     how='outer',

                     left_on=['Date'],

                     right_on=['Date'])



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

    N = df['Confirmed'][-1]

except:

    N = 10000



opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-6).x

print(opt)



try:

    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

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

    

    # now add the results of the model to the dataframe

    df2 = pd.DataFrame(y_model,index=x_model,columns=['No Offset'])

    df2.index.name = 'Date'

    df_result = pd.merge(df_result,

                         df2,

                         how='outer',

                         left_on=['Date'],

                         right_on=['Date'])

    

except:

    pass



fig.show()
df_result['Offset error'] = (df_result['Confirmed']-df_result['Offset'])/df_result['Confirmed']*100

df_result['Offset error'][df_result['Confirmed']==0]=0



df_result['No Offset error'] = (df_result['Confirmed']-df_result['No Offset'])/df_result['Confirmed']*100

df_result['No Offset error'][df_result['Confirmed']==0]=0



def highlight_max(s):

    '''

    highlight the absolute maximum value in a Series with red font.

    '''

    is_min = abs(s) == abs(s).max()

    return ['color: red' if v else '' for v in is_min]



df_result.style.apply(highlight_max,axis=1,subset=['Offset error', 'No Offset error'])
# now plot the prediction for the country

# x_actual = pd.to_datetime(df_result['Date'].reset_index())

x_actual = list(df_result.reset_index()['Date'])



x_model = x_actual

y_model_clus = list(df_result['Offset error'])

y_model_glob = list(df_result['No Offset error'])



# instantiate the figure and add the two series - actual vs modelled    

fig = go.Figure()



fig.update_layout(title=country,

                  xaxis_title='Date',

                  yaxis_title="% error",

                  autosize=False,

                  width=700,

                  height=500,

                  #yaxis_type='log'

                 )



fig.add_trace(go.Line(x=x_model,

                      y=y_model_clus,

                      mode='lines',

                      name='Offset error',

                      line=dict(color='blue', 

                                width=1

                               )

                     ) 

             )



fig.add_trace(go.Line(x=x_model,

                      y=y_model_glob,

                      mode='lines',

                      name='No Offset error',

                      line=dict(color='red', 

                                width=1.0,

                                dash='dot'

                               )

                     ) 

             )



fig.show()
country = 'US'

cluster = 'California'



df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]

df = df.groupby(['Date','Country/Region']).sum().reset_index()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

df = df.set_index('Date')[['Deaths']]

df_result = df.copy()

# df_result = df_result[['Date','Deaths']]



# ensure that the model starts from when the first case is detected

# NOTE: its better not to truncate the dataset like this 

# df = df[df[df.columns[0]]>0]



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

    N = df['Deaths'][-1]

    T = -df['Deaths'][0]

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



x_model = []

y_model = []



# get the model values for the same time series as the actuals

for t in range(len(df) + days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))



# instantiate the figure and add the two series - actual vs modelled    

fig = go.Figure()

fig.update_layout(title=country + ' - ' + cluster,

                  xaxis_title='Date',

                  yaxis_title="nr People",

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

                                  color='black', 

                                  line_width=1.5,

                                 )

                     ) 

             )    



fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Prediction with offset",

                      line=dict(color='blue', 

                                width=2.5

                               )

                     ) 

             ) 



# now add the results of the model to the dataframe

df2 = pd.DataFrame(y_model,index=x_model,columns=['Offset'])

df2.index.name = 'Date'

df_result = pd.merge(df_result,

                     df2,

                     how='outer',

                     left_on=['Date'],

                     right_on=['Date'])



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

    N = df['Deaths'][-1]

except:

    N = 10000



opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-6).x

print(opt)



try:

    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

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

    

    # now add the results of the model to the dataframe

    df2 = pd.DataFrame(y_model,index=x_model,columns=['No Offset'])

    df2.index.name = 'Date'

    df_result = pd.merge(df_result,

                         df2,

                         how='outer',

                         left_on=['Date'],

                         right_on=['Date'])

    

except:

    pass



fig.show()
df_result['Offset error'] = (df_result['Deaths']-df_result['Offset'])/df_result['Deaths']*100

df_result['Offset error'][df_result['Deaths']==0]=0



df_result['No Offset error'] = (df_result['Deaths']-df_result['No Offset'])/df_result['Deaths']*100

df_result['No Offset error'][df_result['Deaths']==0]=0



def highlight_max(s):

    '''

    highlight the absolute maximum value in a Series with red font.

    '''

    is_min = abs(s) == abs(s).max()

    return ['color: red' if v else '' for v in is_min]



df_result.style.apply(highlight_max,axis=1,subset=['Offset error', 'No Offset error'])
# now plot the prediction for the country

# x_actual = pd.to_datetime(df_result['Date'].reset_index())

x_actual = list(df_result.reset_index()['Date'])



x_model = x_actual

y_model_clus = list(df_result['Offset error'])

y_model_glob = list(df_result['No Offset error'])



# instantiate the figure and add the two series - actual vs modelled    

fig = go.Figure()



fig.update_layout(title=country,

                  xaxis_title='Date',

                  yaxis_title="% error",

                  autosize=False,

                  width=700,

                  height=500,

                  #yaxis_type='log'

                 )



fig.add_trace(go.Line(x=x_model,

                      y=y_model_clus,

                      mode='lines',

                      name='Offset error',

                      line=dict(color='blue', 

                                width=1

                               )

                     ) 

             )



fig.add_trace(go.Line(x=x_model,

                      y=y_model_glob,

                      mode='lines',

                      name='No Offset error',

                      line=dict(color='red', 

                                width=1.0,

                                dash='dot'

                               )

                     ) 

             )



fig.show()
df_ca_train = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df_ca_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

df_ca_submission = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')
df_ca_train.tail(10)
country = 'US'

cluster = 'California'



df = full_table[(full_table['Country/Region'] == country)&(full_table['Province/State'] == cluster)]

df = df.groupby(['Date','Country/Region']).sum().reset_index()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

df = df.set_index('Date')[['Confirmed']]

df_result = df.copy()

# drop the last three row of dataframe to model competitions results

df.drop(df.tail(3).index,inplace=True)



# ensure that the model starts from when the first case is detected

# NOTE: its better not to truncate the dataset like this 

# df = df[df[df.columns[0]]>0]



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

    N = df['Confirmed'][-1]

    T = -df['Confirmed'][0]

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



x_model = []

y_model = []



# get the model values for the same time series as the actuals

for t in range(len(df) + days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))



# instantiate the figure and add the two series - actual vs modelled    

fig = go.Figure()

fig.update_layout(title=country + ' - ' + cluster,

                  xaxis_title='Date',

                  yaxis_title="nr People",

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

                                  color='black', 

                                  line_width=1.5,

                                 )

                     ) 

             )    



fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Prediction with offset",

                      line=dict(color='blue', 

                                width=2.5

                               )

                     ) 

             ) 



# now add the results of the model to the dataframe

df2 = pd.DataFrame(y_model,index=x_model,columns=['Offset'])

df2.index.name = 'Date'

df_result = pd.merge(df_result,

                     df2,

                     how='outer',

                     left_on=['Date'],

                     right_on=['Date'])



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

    N = df['Confirmed'][-1]

except:

    N = 10000



opt = minimize(model_loss, x0=np.array([N, 0.1, 5]), method='Nelder-Mead', tol=1e-6).x

print(opt)



try:

    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

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

    

    # now add the results of the model to the dataframe

    df2 = pd.DataFrame(y_model,index=x_model,columns=['No Offset'])

    df2.index.name = 'Date'

    df_result = pd.merge(df_result,

                         df2,

                         how='outer',

                         left_on=['Date'],

                         right_on=['Date'])

    

except:

    pass



fig.show()
df = df_ca_train

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

days_forecast = len(df)+len(df_ca_test)-7

x_model = []

y_model = []



# get the model values for the same time series as the actuals

for t in range(days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))



# instantiate the figure and add the two series - actual vs modelled    

fig = go.Figure()

fig.update_layout(title=country + ' - ' + cluster,

                  xaxis_title='Date',

                  yaxis_title="nr People",

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

                                  color='black', 

                                  line_width=1.5,

                                 )

                     ) 

             )    



fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Prediction with offset",

                      line=dict(color='blue', 

                                width=2.5

                               )

                     ) 

             ) 



# now add the results of the model to the dataframe



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

df = df_ca_train

df = df.groupby(['Date','Country/Region']).sum().reset_index()

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'])

df = df.set_index('Date')[['Fatalities']]





# # define the models to forecast the growth of cases

# def model(N, a, alpha, t0, t):

#     return N * (1 - math.e ** (-a * (t-t0))) ** alpha



# def model_loss(params):

#     N, a, alpha, t0 = params

#     global df

#     r = 0

#     for t in range(len(df)):

#         r += (model(N, a, alpha, t0, t) - df.iloc[t, 0]) ** 2

#     return r 

# try:

#     N = df['Fatalities'][-1]

#     T = -df['Fatalities'][0]

# except:

#     N = 10000

#     T = 0



# opt = minimize(model_loss, x0=np.array([N, 0.1, 5, T]), method='Nelder-Mead', tol=1e-6).x



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



# create series to be plotted 

x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

x_actual =list(x_actual)

y_actual = list(df.reset_index().iloc[:,1])



start_date = pd.to_datetime(df.index[0])

days_forecast = len(df)+len(df_ca_test)-7

x_model = []

y_model = []



# get the model values for the same time series as the actuals

for t in range(days_forecast):

    x_model.append(start_date + datetime.timedelta(days=t))

    y_model.append(round(model(*opt,t)))



# instantiate the figure and add the two series - actual vs modelled    

fig = go.Figure()

fig.update_layout(title=country + ' - ' + cluster,

                  xaxis_title='Date',

                  yaxis_title="nr People",

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

                                  color='black', 

                                  line_width=1.5,

                                 )

                     ) 

             )    



fig.add_trace(go.Line(x=x_model,

                      y=y_model,

                      mode='lines',

                      name="Prediction with offset",

                      line=dict(color='blue', 

                                width=2.5

                               )

                     ) 

             ) 



# now add the results of the model to the dataframe



df2 = pd.DataFrame(y_model,index=pd.to_datetime(x_model),columns=['Fatalities'])

df2.index.name = 'Date'

df_comp = pd.merge(df_comp,

                     df2,

                     how='outer',

                     left_on=['Date'],

                     right_on=['Date'])



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

df_comp = df_comp.reset_index()

df_comp.tail()
df_ca_test.head()
df_ca_test['Date'] = pd.to_datetime(df_ca_test['Date'])
df_ca_test.info()
df_sub = pd.merge(df_ca_test,

                  df_comp,

                  how='left',

                  on=['Date']

                 )
df_sub.tail()
df_sub = df_sub[['ForecastId','ConfirmedCases','Fatalities']]

df_sub.tail()
df_ca_submission.tail()
df_sub.to_csv('submission.csv',index=False)