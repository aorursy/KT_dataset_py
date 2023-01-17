import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

import plotly.graph_objects as go

import warnings

import datetime

import math

from scipy.optimize import minimize

from scipy import stats

from datetime import datetime, timedelta
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
countries = ['South Africa',

             

#              Americas

#              'Colombia',

             'US',

             'Canada',

#              'Mexico',

             'Chile', 

             'Brazil',

#              'Peru',

#              'Argentina',

             

#              Europe

             'United Kingdom',

             'France',

             'Spain',

             'Portugal',

#              'Italy',

             'Germany', 

             'Belgium',

             'Netherlands',             



#             Nordic

             'Sweden',

             'Norway',

#              'Finland',

#              'Denmark',

#              'Greenland',

#              'Iceland',

             

#              East Block

             'Russia',

#              'Belarus',

             

#              South Pacific

             'Australia',

#              'New Zealand',

#              'Indonesia',

             

#              Asia

#              'Japan',

#              'Korea, South',

#              'China',

             

#              South Asia

#              'India',

#              'Bangladesh',

             

#              Middle East

#              'Iran',

#              'Turkey',

#              'United Arab Emirates',

                        

#              Africa

             'Mauritius',

#              'Botswana',

#              'Zimbabwe',

#              'Angola',

#              'Swaziland',

#              'Namibia',

#              'Lesotho'

             ]



pd.DataFrame(countries,columns=['Country'])
# download the latest data sets



# conf_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

conf_df = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv')



# deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

deaths_df = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv')



# recv_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

recv_df = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv')



pop_df = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')

pop_df['Country (or dependency)'][pop_df['Country (or dependency)']=='United States']="US"

pop_df['Country (or dependency)'][pop_df['Country (or dependency)']=='Taiwan']="Taiwan*"

pop_df['Country (or dependency)'][pop_df['Country (or dependency)']=='South Korea']="Korea, South"

pop_df['Country (or dependency)'][pop_df['Country (or dependency)']=='Czech Republic (Czechia)']="Czechia"

pop_df['Country (or dependency)'][pop_df['Country (or dependency)']=='DR Congo']="Congo (Kinshasa)"

pop_df['Country (or dependency)'][pop_df['Country (or dependency)']=='Congo']="Congo (Brazzaville)"





pop_df = pop_df.rename(columns={'Country (or dependency)':'Country/Region',

                                'Population (2020)':'Population',

                                'Density (P/Km²)':'Density (P/sqkm)'})



# test_df = pd.read_excel('https://covid.ourworldindata.org/data/owid-covid-data.xlsx')

# population = pd.read_csv('http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv')

# create full table

dates = conf_df.columns[4:]



conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Confirmed')



deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Deaths')



recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')



# full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'],recv_df_long['Recovered']], axis=1, sort=False)

full_table = pd.concat([conf_df_long, deaths_df_long['Deaths']], axis=1, sort=False)

full_table = pd.merge(full_table,

                      recv_df_long,

                      how='left',

                      on=['Date','Province/State', 'Country/Region', 'Lat', 'Long']

                     )



full_table = pd.merge(full_table,

                      pop_df,

                      how='left',

                      on=['Country/Region']

                     )



full_table['Recovered'] = full_table['Recovered'].fillna(0)

full_table['Recovered'] = full_table['Recovered'].astype('int')
# avoid double counting

full_table = full_table[full_table['Province/State'].str.contains(',')!=True]
# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

full_table['Case Fatality Rate est.'] = full_table['Deaths'] / (full_table['Deaths'] + full_table['Recovered'])



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')
# Display the number cases globally

# df['Recovered'].fillna(0)

df = full_table.groupby('Date')['Confirmed', 'Deaths','Recovered'].sum().reset_index()

df['Active'] = df['Confirmed']-df['Deaths']-df['Recovered']

df['CFR % est.'] = df['Deaths']/(df['Deaths']+df['Recovered'])

df['CFR % est.'] = df['CFR % est.'].fillna(0.00123)

df['CFR % est.'][df['Recovered'] == 0] = 0.00123

yesterday = datetime.strptime(full_table['Date'].iloc[-1], '%m/%d/%y').date() - timedelta(days=1)

yesterday = yesterday.strftime("%m/%d/%y").lstrip("0").replace("/0", "/")

# print(yesterday)

# if yesterday[0]=='0':

#     yesterday = yesterday[1:]

    



df2 =  df[df['Date']==yesterday].reset_index(drop=True)

df =  df[df['Date']==full_table['Date'].iloc[-1]].reset_index(drop=True)

df['New Confirmed'] = df['Confirmed']-df2['Confirmed']

df['New Deaths'] = df['Deaths']-df2['Deaths']



df.style.format({'CFR % est.': '{:.2%}',

                 'Confirmed':'{:,.0f}',

                 'Deaths':'{:,.0f}',

                 'Recovered':'{:,.0f}',

                 'Active':'{:,.0f}', 

                 'New Confirmed':'{:,.0f}',

                 'New Deaths':'{:,.0f}'

                })



# df2.style.format({'CFR % est.': '{:.2%}',

#                  'Confirmed':'{:,.0f}',

#                  'Deaths':'{:,.0f}',

#                  'Recovered':'{:,.0f}',

#                  'Active':'{:,.0f}'})
# Global view

try:

    df = full_table

    df = df.groupby(['Date']).sum().reset_index()

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by=['Date'])

    df = df.set_index('Date')[['Confirmed']]

    

    # ensure that the model starts from when the first case is detected

#         df = df[df[df.columns[0]]>20]

#         df['Confirmed']=df['Confirmed']-19



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



#         df['Confirmed'] = df['Confirmed']+19



    # create series to be plotted 

    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

    x_actual =list(x_actual)

    y_actual = list(df.reset_index().iloc[:,1])



    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

        x_model.append(start_date + timedelta(days=t))

        y_model.append(round(model(*opt,t)))



    # instantiate the figure and add the two series - actual vs modelled    

    fig = go.Figure()

    fig.update_layout(title='Global Cumulative',

                      xaxis_title='Date',

                      yaxis_title="nr People",

                      autosize=False,

                      width=900,

                      height=500,

#                           yaxis_type='log'

                     )



    fig.add_trace(go.Scatter(x=x_actual,

                          y=y_actual,

                          mode='markers',

                          name='Actual',

                          marker=dict(symbol='circle-open-dot', 

                                      size=6, 

                                      color='black', 

                                      line_width=1.5,

                                     )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_model,

                          y=y_model,

                          mode='lines',

                          name="Today's Prediction",

                          line=dict(color='blue', 

                                    width=2.5

                                   )

                         ) 

                 ) 





    # drop the last row of dataframe to model yesterday's results

    df.drop(df.tail(1).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))





        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Yesterday's Prediction",

                              line=dict(color='Red', 

                                        width=1.5,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    # drop the last row of dataframe to model results from a week ago

    df.drop(df.tail(6).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))





        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Last week's Prediction",

                              line=dict(color='green', 

                                        width=1.1,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    fig.show()

except:

    pass


try:

    df = full_table

    df = df.groupby(['Date']).sum().reset_index()

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by=['Date'])

    df = df.set_index('Date')[['Confirmed']]



    # ensure that the model starts from when the first case is detected

#         df = df[df[df.columns[0]]>20]

#         df['Confirmed']=df['Confirmed']-19



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



#         df['Confirmed'] = df['Confirmed']+19



    # create series to be plotted 

    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

    x_actual =list(x_actual)

    y_actual = list(df.reset_index().iloc[:,1])

    y_act_daily = list(df.reset_index().iloc[:,1].diff())



    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

        x_model.append(start_date + timedelta(days=t))

        y_model.append(round(model(*opt,t)))

    y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

    y_model.insert(0,0)



    # instantiate the figure and add the two series - actual vs modelled    

    fig = go.Figure()

    fig.update_layout(title='Global Daily',

                      xaxis_title='Date',

                      yaxis_title="nr People",

                      autosize=False,

                      width=900,

                      height=500,

#                           yaxis_type='log'

                     )



    fig.add_trace(go.Scatter(x=x_actual,

                          y=y_act_daily,

                          mode='lines+markers',

                          name='Actual',

                          marker=dict(symbol='circle-open-dot', 

                                      size=6, 

                                      color='black', 

                                      line_width=1.5,

                                     )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_model,

                          y= y_model,

                          mode='lines',

                          name="Today's Prediction",

                          line=dict(color='blue', 

                                    width=2.5

                                   )

                         ) 

                 ) 





    # drop the last row of dataframe to model yesterday's results

    df.drop(df.tail(1).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))

        y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

        y_model.insert(0,0)



        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Yesterday's Prediction",

                              line=dict(color='Red', 

                                        width=1.5,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    # drop the last row of dataframe to model results from a week ago

    df.drop(df.tail(6).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))

        y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

        y_model.insert(0,0)



        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Last week's Prediction",

                              line=dict(color='green', 

                                        width=1.1,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    fig.show()

except:

    pass


try:

    df = full_table

    df = df.groupby(['Date']).sum().reset_index()

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by=['Date'])

    df = df.set_index('Date')[['Deaths']]



    # ensure that the model starts from when the first case is detected

#         df = df[df[df.columns[0]]>0]



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    # create series to be plotted 

    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

    x_actual =list(x_actual)

    y_actual = list(df.reset_index().iloc[:,1])



    try:

        start_date = pd.to_datetime(df.index[0])

    except:

        pass



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

        x_model.append(start_date + timedelta(days=t))

        y_model.append(round(model(*opt,t)))



    # instantiate the figure and add the two series - actual vs modelled    

    fig = go.Figure()

    fig.update_layout(title='Global Cumulative Deaths',

                      xaxis_title='Date',

                      yaxis_title="nr People",

                      autosize=False,

                      width=900,

                      height=500,

#                           yaxis_type='log'

                     )



    fig.add_trace(go.Scatter(x=x_actual,

                          y=y_actual,

                          mode='markers',

                          name='Actual',

                          marker=dict(symbol='circle-open-dot', 

                                      size=6, 

                                      color='black', 

                                      line_width=1.5,

                                     )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_model,

                          y=y_model,

                          mode='lines',

                          name="Today's Prediction",

                          line=dict(color='red', 

                                    width=2.5

                                   )

                         ) 

                 ) 





    # drop the last row of dataframe to model yesterday's results

    df.drop(df.tail(1).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))





        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Yesterday's Prediction",

                              line=dict(color='blue', 

                                        width=1.5,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    # drop the last row of dataframe to model results from a week ago

    df.drop(df.tail(6).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))





        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Last week's Prediction",

                              line=dict(color='green', 

                                        width=1.1,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass





    fig.show()

except:

    pass


try:

    df = full_table

    df = df.groupby(['Date']).sum().reset_index()

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by=['Date'])

    df = df.set_index('Date')[['Deaths']]



    df = get_time_series(country)



    # ensure that the model starts from when the first case is detected

#         df = df[df[df.columns[0]]>20]

#         df['Deaths']=df['Deaths']-19



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



#         df['Deaths'] = df['Deaths']+19



    # create series to be plotted 

    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

    x_actual =list(x_actual)

    y_actual = list(df.reset_index().iloc[:,1])

    y_act_daily = list(df.reset_index().iloc[:,1].diff())



    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

        x_model.append(start_date + timedelta(days=t))

        y_model.append(round(model(*opt,t)))

    y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

    y_model.insert(0,0)



    # instantiate the figure and add the two series - actual vs modelled    

    fig = go.Figure()

    fig.update_layout(title='Global Daily Deaths',

                      xaxis_title='Date',

                      yaxis_title="nr People",

                      autosize=False,

                      width=900,

                      height=500,

#                           yaxis_type='log'

                     )



    fig.add_trace(go.Scatter(x=x_actual,

                          y=y_act_daily,

                          mode='lines+markers',

                          name='Actual',

                          marker=dict(symbol='circle-open-dot', 

                                      size=6, 

                                      color='black', 

                                      line_width=1.5,

                                     )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_model,

                          y= y_model,

                          mode='lines',

                          name="Today's Prediction",

                          line=dict(color='red', 

                                    width=2.5

                                   )

                         ) 

                 ) 





    # drop the last row of dataframe to model yesterday's results

    df.drop(df.tail(1).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))

        y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

        y_model.insert(0,0)



        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Yesterday's Prediction",

                              line=dict(color='Red', 

                                        width=1.5,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    # drop the last row of dataframe to model results from a week ago

    df.drop(df.tail(6).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))

        y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

        y_model.insert(0,0)



        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Last week's Prediction",

                              line=dict(color='green', 

                                        width=1.1,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    fig.show()

except:

    pass
# count the number cases per country

df = full_table[full_table['Date'] == full_table['Date'].iloc[-1]].reset_index()

df = df.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()



# df['Active'] = df['Confirmed']-df['Deaths']-df['Recovered']

# df['CFR % est.'] = df['Deaths']/(df['Deaths']+df['Recovered'])

# df['CFR % est.'] = df['CFR % est.'].fillna(0.00123)

# df['CFR % est.'][df['Recovered'] == 0] = 0.00123



df2 = full_table[full_table['Date'] == yesterday].reset_index()

df2 = df2.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()



df['New Confirmed'] = df['Confirmed']-df2['Confirmed']

df['New Deaths'] = df['Deaths']-df2['Deaths']



df = pd.merge(df,

              pop_df[['Country/Region','Population']],

              how='left',

              on=['Country/Region']

              )



df['% Pop. Confirmed'] = df['Confirmed']/df['Population']

df['% Pop. Deaths'] = df['Deaths']/df['Population']





df = df.sort_values(by=['Deaths','Confirmed'], ascending=False)

df = df.reset_index(drop=True)

df.fillna(0,inplace=True)

df.style.background_gradient(cmap='coolwarm').format({'CFR % est.': '{:.2%}',

                                                      'Confirmed':'{:,.0f}',

                                                      'Deaths':'{:,.0f}',

                                                      'Recovered':'{:,.0f}',

                                                      'Active':'{:,.0f}', 

                                                      'New Confirmed':'{:,.0f}',

                                                      'New Deaths':'{:,.0f}',

                                                      'Population':'{:,.0f}',

                                                      'Density (P/sqkm)':'{:,.0f}',

                                                      '% Pop. Confirmed': '{:.2%}',

                                                      '% Pop. Deaths': '{:.3%}',})

for country in countries:

    try:

        def get_time_series(country):

            df = full_table[(full_table['Country/Region'] == country)]

            df = df.groupby(['Date','Country/Region']).sum().reset_index()

            df['Date'] = pd.to_datetime(df['Date'])

            df = df.sort_values(by=['Date'])

            return df.set_index('Date')[['Confirmed']]



        df = get_time_series(country)



        # ensure that the model starts from when the first case is detected

#         df = df[df[df.columns[0]]>20]

#         df['Confirmed']=df['Confirmed']-19

        

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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x

        

#         df['Confirmed'] = df['Confirmed']+19

        

        # create series to be plotted 

        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

        x_actual =list(x_actual)

        y_actual = list(df.reset_index().iloc[:,1])



        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))



        # instantiate the figure and add the two series - actual vs modelled    

        fig = go.Figure()

        fig.update_layout(title=country,

                          xaxis_title='Date',

                          yaxis_title="nr People",

                          autosize=False,

                          width=900,

                          height=500,

#                           yaxis_type='log'

                         )



        fig.add_trace(go.Scatter(x=x_actual,

                              y=y_actual,

                              mode='markers',

                              name='Actual',

                              marker=dict(symbol='circle-open-dot', 

                                          size=6, 

                                          color='black', 

                                          line_width=1.5,

                                         )

                             ) 

                     )    



        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Today's Prediction",

                              line=dict(color='blue', 

                                        width=2.5

                                       )

                             ) 

                     ) 





        # drop the last row of dataframe to model yesterday's results

        df.drop(df.tail(1).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))





            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Yesterday's Prediction",

                                  line=dict(color='Red', 

                                            width=1.5,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass

        

        

        

        # drop the last row of dataframe to model results from a week ago

        df.drop(df.tail(6).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))





            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Last week's Prediction",

                                  line=dict(color='green', 

                                            width=1.1,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass

        

        

        

        fig.show()

    except:

        pass
for country in countries:

    try:

        def get_time_series(country):

            df = full_table[(full_table['Country/Region'] == country)]

            df = df.groupby(['Date','Country/Region']).sum().reset_index()

            df['Date'] = pd.to_datetime(df['Date'])

            df = df.sort_values(by=['Date'])

            return df.set_index('Date')[['Confirmed']]



        df = get_time_series(country)



        # ensure that the model starts from when the first case is detected

    #         df = df[df[df.columns[0]]>20]

    #         df['Confirmed']=df['Confirmed']-19



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    #         df['Confirmed'] = df['Confirmed']+19



        # create series to be plotted 

        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

        x_actual =list(x_actual)

        y_actual = list(df.reset_index().iloc[:,1])

        y_act_daily = list(df.reset_index().iloc[:,1].diff())



        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))

        y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

        y_model.insert(0,0)



        # instantiate the figure and add the two series - actual vs modelled    

        fig = go.Figure()

        fig.update_layout(title=country,

                          xaxis_title='Date',

                          yaxis_title="nr People",

                          autosize=False,

                          width=900,

                          height=500,

    #                           yaxis_type='log'

                         )



        fig.add_trace(go.Scatter(x=x_actual,

                              y=y_act_daily,

                              mode='lines+markers',

                              name='Actual',

                              marker=dict(symbol='circle-open-dot', 

                                          size=6, 

                                          color='black', 

                                          line_width=1.5,

                                         )

                             ) 

                     )    



        fig.add_trace(go.Scatter(x=x_model,

                              y= y_model,

                              mode='lines',

                              name="Today's Prediction",

                              line=dict(color='blue', 

                                        width=2.5

                                       )

                             ) 

                     ) 





        # drop the last row of dataframe to model yesterday's results

        df.drop(df.tail(1).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))

            y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

            y_model.insert(0,0)



            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Yesterday's Prediction",

                                  line=dict(color='Red', 

                                            width=1.5,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass







        # drop the last row of dataframe to model results from a week ago

        df.drop(df.tail(6).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))

            y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

            y_model.insert(0,0)



            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Last week's Prediction",

                                  line=dict(color='green', 

                                            width=1.1,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass







        fig.show()

    except:

        pass
for country in countries:

    try:

        def get_time_series(country):

            df = full_table[(full_table['Country/Region'] == country)]

            df = df.groupby(['Date','Country/Region']).sum().reset_index()

            df['Date'] = pd.to_datetime(df['Date'])

            df = df.sort_values(by=['Date'])

            return df.set_index('Date')[['Deaths']]



        df = get_time_series(country)



        # ensure that the model starts from when the first case is detected

#         df = df[df[df.columns[0]]>0]



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        # create series to be plotted 

        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

        x_actual =list(x_actual)

        y_actual = list(df.reset_index().iloc[:,1])



        try:

            start_date = pd.to_datetime(df.index[0])

        except:

            continue



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))



        # instantiate the figure and add the two series - actual vs modelled    

        fig = go.Figure()

        fig.update_layout(title=country,

                          xaxis_title='Date',

                          yaxis_title="nr People",

                          autosize=False,

                          width=900,

                          height=500,

#                           yaxis_type='log'

                         )



        fig.add_trace(go.Scatter(x=x_actual,

                              y=y_actual,

                              mode='markers',

                              name='Actual',

                              marker=dict(symbol='circle-open-dot', 

                                          size=6, 

                                          color='black', 

                                          line_width=1.5,

                                         )

                             ) 

                     )    



        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Today's Prediction",

                              line=dict(color='red', 

                                        width=2.5

                                       )

                             ) 

                     ) 





        # drop the last row of dataframe to model yesterday's results

        df.drop(df.tail(1).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))





            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Yesterday's Prediction",

                                  line=dict(color='blue', 

                                            width=1.5,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass

        

        

        

        # drop the last row of dataframe to model results from a week ago

        df.drop(df.tail(6).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))





            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Last week's Prediction",

                                  line=dict(color='green', 

                                            width=1.1,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass

        

        

        fig.show()

    except:

        pass
for country in countries:

    try:

        def get_time_series(country):

            df = full_table[(full_table['Country/Region'] == country)]

            df = df.groupby(['Date','Country/Region']).sum().reset_index()

            df['Date'] = pd.to_datetime(df['Date'])

            df = df.sort_values(by=['Date'])

            return df.set_index('Date')[['Deaths']]



        df = get_time_series(country)



        # ensure that the model starts from when the first case is detected

    #         df = df[df[df.columns[0]]>20]

    #         df['Deaths']=df['Deaths']-19



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    #         df['Deaths'] = df['Deaths']+19



        # create series to be plotted 

        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

        x_actual =list(x_actual)

        y_actual = list(df.reset_index().iloc[:,1])

        y_act_daily = list(df.reset_index().iloc[:,1].diff())



        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))

        y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

        y_model.insert(0,0)



        # instantiate the figure and add the two series - actual vs modelled    

        fig = go.Figure()

        fig.update_layout(title=country,

                          xaxis_title='Date',

                          yaxis_title="nr People",

                          autosize=False,

                          width=900,

                          height=500,

    #                           yaxis_type='log'

                         )



        fig.add_trace(go.Scatter(x=x_actual,

                              y=y_act_daily,

                              mode='lines+markers',

                              name='Actual',

                              marker=dict(symbol='circle-open-dot', 

                                          size=6, 

                                          color='black', 

                                          line_width=1.5,

                                         )

                             ) 

                     )    



        fig.add_trace(go.Scatter(x=x_model,

                              y= y_model,

                              mode='lines',

                              name="Today's Prediction",

                              line=dict(color='red', 

                                        width=2.5

                                       )

                             ) 

                     ) 





        # drop the last row of dataframe to model yesterday's results

        df.drop(df.tail(1).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))

            y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

            y_model.insert(0,0)



            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Yesterday's Prediction",

                                  line=dict(color='Red', 

                                            width=1.5,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass







        # drop the last row of dataframe to model results from a week ago

        df.drop(df.tail(6).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))

            y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

            y_model.insert(0,0)



            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Last week's Prediction",

                                  line=dict(color='green', 

                                            width=1.1,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass







        fig.show()

    except:

        pass


period = 7

Case = "Confirmed"



fig = go.Figure()

fig.update_layout(title="COVID-19 Progression",

                  xaxis_title=Case + " % of Population",

                  yaxis_title="New " + Case + " % (Rolling "+ str(period) + " days)",

                  autosize=False,

                  width=900,

                  height=600,

                  yaxis_type='log',

                  xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    try:

        def get_time_series(country):

            df = full_table[(full_table['Country/Region'] == country)]

            df = df.groupby(['Date','Country/Region']).sum().reset_index()

            df['Date'] = pd.to_datetime(df['Date'])

            df = df.sort_values(by=['Date'])

            return df.set_index('Date')[[Case]]



        df = get_time_series(country)



        # create series to be plotted 

        x_confirmed = list(df[Case])

        y_new = list(df[Case].rolling(period).apply(lambda x: x[-1] - x[0]))



    #     x_confirmed = [x for x in x_confirmed if x > 20]

    #     y_new = y_new[-len(x_confirmed):]



        y_new = [1.0 if y==0 else y for y in y_new ]



        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

        x_confirmed = [x*100/population for x in x_confirmed]

        y_new = [x*100/population for x in y_new]



        # instantiate the figure and add the two series - actual vs modelled    

        fig.add_trace(go.Scatter(x=x_confirmed,

                              y=y_new,

                              mode='lines',

                              name=country,

                              text=country,

                              line=dict(

    #                               color='grey', 

                                        width=1

                                       )

                             ) 

                     )    



        fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                              y=y_new[-1:],

                              mode='markers+text',

                              name=country,

                              text=country,

                              textposition='bottom right',

                              marker=dict(symbol='circle', 

                                          size=5, 

                                          color='black'

                                         )

                             ) 

                     )



        try:

            x_log = [math.log(x) for x in x_confirmed]

            y_log = [1.0 if y==0 else y for y in y_new ]

            y_log = [math.log(y) for y in y_log]



            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log[period:-period],y_log[period:-period])

            line = [math.exp(slope*x+intercept) for x in x_log]

            line = [math.exp(1.0*x-0.2) for x in x_log]

            fig.add_trace(go.Scatter(x=x_confirmed,

                                  y=line,

                                  mode='lines',

                                  name="linear fit",

                                  line=dict(color='blue', 

                                            width=2.5,

    #                                         dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass

    except:

        pass

fig.show()
period = 7

Case = "Deaths"



fig = go.Figure()

fig.update_layout(title="COVID-19 Progression",

                  xaxis_title=Case + " % of Population",

                  yaxis_title="New " + Case + " % (Rolling "+ str(period) + " days)",

                  autosize=False,

                  width=900,

                  height=600,

                  yaxis_type='log',

                  xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    try:

        def get_time_series(country):

            df = full_table[(full_table['Country/Region'] == country)]

            df = df.groupby(['Date','Country/Region']).sum().reset_index()

            df['Date'] = pd.to_datetime(df['Date'])

            df = df.sort_values(by=['Date'])

            return df.set_index('Date')[[Case]]



        df = get_time_series(country)



        # create series to be plotted 

        x_confirmed = list(df[Case])

        y_new = list(df[Case].rolling(period).apply(lambda x: x[-1] - x[0]))



    #     x_confirmed = [x for x in x_confirmed if x > 20]

    #     y_new = y_new[-len(x_confirmed):]



        y_new = [1.0 if y==0 else y for y in y_new ]



        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

        x_confirmed = [x*100/population for x in x_confirmed]

        y_new = [x*100/population for x in y_new]



        # instantiate the figure and add the two series - actual vs modelled    

        fig.add_trace(go.Scatter(x=x_confirmed,

                              y=y_new,

                              mode='lines',

                              name=country,

                              text=country,

                              line=dict(

    #                               color='grey', 

                                        width=1

                                       )

                             ) 

                     )    



        fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                              y=y_new[-1:],

                              mode='markers+text',

                              name=country,

                              text=country,

                              textposition='bottom right',

                              marker=dict(symbol='circle', 

                                          size=5, 

                                          color='black'

                                         )

                             ) 

                     )



        try:

            x_log = [math.log(x) for x in x_confirmed]

            y_log = [1.0 if y==0 else y for y in y_new ]

            y_log = [math.log(y) for y in y_log]



            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log[period:-period],y_log[period:-period])

            line = [math.exp(slope*x+intercept) for x in x_log]

            line = [math.exp(1.0*x-0.2) for x in x_log]

            fig.add_trace(go.Scatter(x=x_confirmed,

                                  y=line,

                                  mode='lines',

                                  name="linear fit",

                                  line=dict(color='blue', 

                                            width=2.5,

    #                                         dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass

    except:

        pass

fig.show()
nr_from = 0.0001

Case = "Confirmed"



peak_df = pd.DataFrame(countries,columns=['Country'])

peak_df['Peak '+Case]=0



fig = go.Figure()

fig.update_layout(title="Cumulative - Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + "% of population",

                  yaxis_title="% Population "+ Case,

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=population*nr_from/100]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case])

    peak_df['Peak '+Case][peak_df['Country']==country]=y_new.index(max(y_new))

#         y_new = [1.0 if y==0 else y for y in y_new ]





    y_new = [x*100/population for x in y_new]



    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
period = 10

nr_from = 0.0001

Case = "Confirmed"



peak_df = pd.DataFrame(countries,columns=['Country'])

peak_df['Peak '+Case]=0

# peak_df['Peak '+Case][peak_df['Country']==country]=y_new.index(max(y_new[1:]))



fig = go.Figure()

fig.update_layout(title="Daily - Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + "% of population",

                  yaxis_title= str(period) + "d moving average for % Population "+ Case,

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=population*nr_from/100]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case].diff().rolling(window=period,min_periods=1).mean())

    y_new = [x*100/population for x in y_new]

    

    peak_df['Peak '+Case][peak_df['Country']==country]=y_new.index(max(y_new[1:]))

    

    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
peak_df
nr_from = 0.0001

Case = "Deaths"



fig = go.Figure()

fig.update_layout(title="Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + "% of population",

                  yaxis_title="% Population "+ Case,

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=population*nr_from/100]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case])

    y_new = [x*100/population for x in y_new]



    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
period = 10

nr_from = 0.000001

Case = "Deaths"



peak_df = pd.DataFrame(countries,columns=['Country'])

peak_df['Peak '+Case]=0

# peak_df['Peak '+Case][peak_df['Country']==country]=y_new.index(max(y_new[1:]))





fig = go.Figure()

fig.update_layout(title="Daily - Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + "% of population",

                  yaxis_title= str(period) + "d moving average for % Population "+ Case,

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=population*nr_from/100]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case].diff().rolling(window=period,min_periods=1).mean())

    y_new = [x*100/population for x in y_new]

    peak_df['Peak '+Case][peak_df['Country']==country]=y_new.index(max(y_new[1:]))

    

    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
peak_df
nr_from = 50

Case = "Confirmed"



fig = go.Figure()

fig.update_layout(title="Cumulative - Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + " cases",

                  yaxis_title="Nr People",

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=nr_from]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case])



    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
period = 10

nr_from = 50

Case = "Confirmed"



fig = go.Figure()

fig.update_layout(title="Daily - Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + " cases",

                  yaxis_title= str(period) + "d moving average for nr "+ Case,

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=nr_from]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case].diff().rolling(window=period,min_periods=1).mean())



    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
nr_from = 10

Case = "Deaths"



fig = go.Figure()

fig.update_layout(title="Cumulative - Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + " cases",

                  yaxis_title="Nr People",

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=nr_from]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case])



    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
nr_from = 10

Case = "Deaths"



fig = go.Figure()

fig.update_layout(title="Daily - Comparison of " + Case,

                  xaxis_title="Days since " + Case + " were " + str(nr_from) + " cases",

                  yaxis_title= str(period) + "d moving average for nr "+ Case,

                  autosize=False,

                  width=900,

                  height=600,

#                   yaxis_type='log',

#                   xaxis_type='log',

                  showlegend=False,

                 )



for country in countries:

    

    try:

        population = list(pop_df['Population'][pop_df['Country/Region']==country])[0]

    except:

        poplation = 1400000

    

    def get_time_series(country):

        df = full_table[(full_table['Country/Region'] == country)]

        df = df.groupby(['Date','Country/Region']).sum().reset_index()

        df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values(by=['Date'])

        df = df[df[Case]>=nr_from]

        return df.set_index('Date')[[Case]]



    df = get_time_series(country)



    # create series to be plotted 

    x_confirmed = list(range(len(df[Case])))

    y_new = list(df[Case].diff().rolling(window=period,min_periods=1).mean())



    # instantiate the figure and add the two series - actual vs modelled    

    fig.add_trace(go.Scatter(x=x_confirmed,

                          y=y_new,

                          mode='lines',

                          name=country,

                          text=country,

                          line=dict(

#                               color='grey', 

                                    width=1.5

                                   )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_confirmed[-1:],

                          y=y_new[-1:],

                          mode='markers+text',

                          name=country,

                          text=country,

                          textposition='bottom right',

                          marker=dict(symbol='circle', 

                                      size=5, 

                                      color='black'

                                     )

                         ) 

                 )



fig.show()
# check the following link for more info: https://github.com/dsfsi/covid19za



# Provincial Confirmed

conf_df = pd.read_csv('https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_confirmed.csv')



# Provincial Testing

# test_prov_df = pd.read_csv('https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_testing.csv')

test_df = pd.read_csv('https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_timeline_testing.csv')



# Provincial Deaths

death_df = pd.read_csv('https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_deaths.csv')

# death2_df = pd.read_csv('https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_timeline_deaths.csv')



# Hospital info 

hosp_df = pd.read_csv('https://raw.githubusercontent.com/dsfsi/covid19za/master/data/health_system_za_public_hospitals.csv')



# Transmission type

# trans_df = pd.read_csv('https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_timeline_transmission_type.csv')
region = conf_df.columns[2:-1]



conf_df_long = conf_df.melt(id_vars=['date', 'YYYYMMDD'], 

                            value_vars=region, var_name='Region', value_name='Confirmed')



deaths_df_long = death_df.melt(id_vars=['date', 'YYYYMMDD'], 

                            value_vars=region, var_name='Region', value_name='Deaths')





# test_df_long = test_df.melt(id_vars=['date', 'YYYYMMDD'], 

#                             value_vars=region, var_name='Type', value_name='Tests')



# full_table = pd.concat([conf_df_long, deaths_df_long['Deaths']], axis=1, sort=False)
full_table2 = pd.merge(conf_df_long,

                      deaths_df_long,

                      how='left',

                      on=['date','YYYYMMDD','Region'])
full_table2['Confirmed'] = full_table2['Confirmed'].interpolate()

full_table2['Deaths'] = full_table2['Deaths'].interpolate()

provinces = ['WC','GP','KZN','EC','FS','LP','MP','NW','NC']
# count the number cases per country

df = full_table2[full_table2['date'] == full_table2['date'].iloc[-1]].reset_index()

df = df.groupby('Region')['Confirmed', 'Deaths'].sum().reset_index()



yesterday = datetime.strptime(full_table2['date'].iloc[-1], '%d-%m-%Y').date() - timedelta(days=1)

yesterday = yesterday.strftime("%d-%m-%Y")



df2 = full_table2[full_table2['date'] == yesterday].reset_index()

df2 = df2.groupby('Region')['Confirmed', 'Deaths'].sum().reset_index()



df['New Confirmed'] = df['Confirmed']-df2['Confirmed']

df['New Deaths'] = df['Deaths']-df2['Deaths']



df = df.sort_values(by=['Deaths','Confirmed'], ascending=False)

df = df.reset_index(drop=True)

df.style.background_gradient(cmap='coolwarm').format({'Confirmed':'{:,.0f}',

                                                      'Deaths':'{:,.0f}', 

                                                      'New Confirmed':'{:,.0f}',

                                                      'New Deaths':'{:,.0f}'})


for province in provinces:



    def get_time_series(province):

        df = full_table2[(full_table2['Region'] == province)]

        df = df.groupby(['date','Region']).sum().reset_index()

        df['date'] = pd.to_datetime(df['date'],format='%d-%m-%Y')

        df = df.sort_values(by=['date'])

        return df.set_index('date')[['Confirmed']]



    df = get_time_series(province)

#     print(df)

    # ensure that the model starts from when the first case is detected

#         df = df[df[df.columns[0]]>20]

#         df['Confirmed']=df['Confirmed']-19



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



#         df['Confirmed'] = df['Confirmed']+19



    # create series to be plotted 

    x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

    x_actual =list(x_actual)

    y_actual = list(df.reset_index().iloc[:,1])



    start_date = pd.to_datetime(df.index[0])



    x_model = []

    y_model = []



    # get the model values for the same time series as the actuals

    for t in range(len(df) + days_forecast):

        x_model.append(start_date + timedelta(days=t))

        y_model.append(round(model(*opt,t)))



    # instantiate the figure and add the two series - actual vs modelled    

    fig = go.Figure()

    fig.update_layout(title=province,

                      xaxis_title='Date',

                      yaxis_title="nr People",

                      autosize=False,

                      width=900,

                      height=500,

#                           yaxis_type='log'

                     )



    fig.add_trace(go.Scatter(x=x_actual,

                          y=y_actual,

                          mode='markers',

                          name='Actual',

                          marker=dict(symbol='circle-open-dot', 

                                      size=6, 

                                      color='black', 

                                      line_width=1.5,

                                     )

                         ) 

                 )    



    fig.add_trace(go.Scatter(x=x_model,

                          y=y_model,

                          mode='lines',

                          name="Today's Prediction",

                          line=dict(color='blue', 

                                    width=2.5

                                   )

                         ) 

                 ) 





    # drop the last row of dataframe to model yesterday's results

    df.drop(df.tail(1).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))





        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Yesterday's Prediction",

                              line=dict(color='Red', 

                                        width=1.5,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    # drop the last row of dataframe to model results from a week ago

    df.drop(df.tail(6).index,inplace=True)



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

    opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    try:

        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))





        # now plot the new series

        fig.add_trace(go.Scatter(x=x_model,

                              y=y_model,

                              mode='lines',

                              name="Last week's Prediction",

                              line=dict(color='green', 

                                        width=1.1,

                                        dash='dot'

                                       )

                             ) 

                     )

    except:

        pass







    fig.show()
for province in provinces:

    try:

        def get_time_series(province):

            df = full_table2[(full_table2['Region'] == province)]

            df = df.groupby(['date','Region']).sum().reset_index()

            df['date'] = pd.to_datetime(df['date'],format='%d-%m-%Y')

            df = df.sort_values(by=['date'])

            return df.set_index('date')[['Confirmed']]



        df = get_time_series(province)



        # ensure that the model starts from when the first case is detected

    #         df = df[df[df.columns[0]]>20]

    #         df['Confirmed']=df['Confirmed']-19



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



    #         df['Confirmed'] = df['Confirmed']+19



        # create series to be plotted 

        x_actual = pd.to_datetime(df.reset_index().iloc[:,0])

        x_actual =list(x_actual)

        y_actual = list(df.reset_index().iloc[:,1])

        y_act_daily = list(df.reset_index().iloc[:,1].diff().rolling(window=3).mean())



        start_date = pd.to_datetime(df.index[0])



        x_model = []

        y_model = []



        # get the model values for the same time series as the actuals

        for t in range(len(df) + days_forecast):

            x_model.append(start_date + timedelta(days=t))

            y_model.append(round(model(*opt,t)))

        y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

        y_model.insert(0,0)



        # instantiate the figure and add the two series - actual vs modelled    

        fig = go.Figure()

        fig.update_layout(title=province,

                          xaxis_title='Date',

                          yaxis_title="nr People",

                          autosize=False,

                          width=900,

                          height=500,

    #                           yaxis_type='log'

                         )



        fig.add_trace(go.Scatter(x=x_actual,

                              y=y_act_daily,

                              mode='lines+markers',

                              name='Actual',

                              marker=dict(symbol='circle-open-dot', 

                                          size=6, 

                                          color='black', 

                                          line_width=1.5,

                                         )

                             ) 

                     )    



        fig.add_trace(go.Scatter(x=x_model,

                              y= y_model,

                              mode='lines',

                              name="Today's Prediction",

                              line=dict(color='blue', 

                                        width=2.5

                                       )

                             ) 

                     ) 





        # drop the last row of dataframe to model yesterday's results

        df.drop(df.tail(1).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))

            y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

            y_model.insert(0,0)



            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Yesterday's Prediction",

                                  line=dict(color='Red', 

                                            width=1.5,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass







        # drop the last row of dataframe to model results from a week ago

        df.drop(df.tail(6).index,inplace=True)



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

        opt = minimize(model_loss, x0=np.array([N, 0.2, 30]), method='Nelder-Mead', tol=1e-6).x



        try:

            start_date = pd.to_datetime(df.index[0])



            x_model = []

            y_model = []



            # get the model values for the same time series as the actuals

            for t in range(len(df) + days_forecast):

                x_model.append(start_date + timedelta(days=t))

                y_model.append(round(model(*opt,t)))

            y_model = [j-i for i, j in zip(y_model[:-1], y_model[1:])]

            y_model.insert(0,0)



            # now plot the new series

            fig.add_trace(go.Scatter(x=x_model,

                                  y=y_model,

                                  mode='lines',

                                  name="Last week's Prediction",

                                  line=dict(color='green', 

                                            width=1.1,

                                            dash='dot'

                                           )

                                 ) 

                         )

        except:

            pass







        fig.show()

    except:

        pass