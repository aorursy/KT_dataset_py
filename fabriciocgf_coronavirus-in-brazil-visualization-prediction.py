import numpy as np 

import pandas as pd 

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

from collections import OrderedDict



# plotly packages

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs import *
def Bayesian_Reg(Lin_index, Lin_df, future_forcast):

    MAE = {}

    tol = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1]

    alpha_1 = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1]

    alpha_2 = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1]

    lambda_1 = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1]

    lambda_2 = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

    

    for size in np.arange(.05, .51, .05):

        X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(Lin_index, Lin_df, test_size=size, shuffle=False) 

    

        for deg in np.arange(3, 8):

            # transform our data for polynomial regression

            poly = PolynomialFeatures(degree=deg)

            poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

            poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

            poly_future_forcast = poly.fit_transform(future_forcast)

    

            # bayesian ridge polynomial regression

            bayesian = BayesianRidge(fit_intercept=False, normalize=True)

            bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40)

            bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)

            bayesian_confirmed = bayesian_search.best_estimator_

            test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)

            bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)

            MAE.update( {mean_absolute_error(test_bayesian_pred, y_test_confirmed) : [deg, size]} )

    sort = list(sorted(MAE.items()))

    mae = sort[0][0]

    size = sort[0][1][1]

    deg = sort[0][1][0]

    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(Lin_index, Lin_df, test_size=size, shuffle=False) 

    

    # transform our data for polynomial regression

    poly = PolynomialFeatures(degree=deg)

    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

    poly_future_forcast = poly.fit_transform(future_forcast)

    

    # bayesian ridge polynomial regression

    bayesian = BayesianRidge(fit_intercept=False, normalize=True)

    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40)

    bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)

    bayesian_confirmed = bayesian_search.best_estimator_

    test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)

    bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)

    return bayesian_pred, y_test_confirmed, test_bayesian_pred, mae, deg
def Linear_Reg(Lin_index, Lin_df, future_forcast):

    MAE = {}

    

    for size in np.arange(.01, .51, .01):

        X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(Lin_index, Lin_df, test_size=size, shuffle=False) 

    

        for deg in np.arange(3, 7):

            # transform our data for polynomial regression

            poly = PolynomialFeatures(degree=deg)

            poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

            poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

            poly_future_forcast = poly.fit_transform(future_forcast)

    

            # polynomial regression

            linear_model = LinearRegression(normalize=True, fit_intercept=False)

            linear_model.fit(poly_X_train_confirmed, y_train_confirmed)

            test_linear_pred = linear_model.predict(poly_X_test_confirmed)

            linear_pred = linear_model.predict(poly_future_forcast)

            MAE.update( {mean_absolute_error(test_linear_pred, y_test_confirmed) : [deg, size]} )

    sort = list(sorted(MAE.items()))

    mae = sort[0][0]

    size = sort[0][1][1]

    deg = sort[0][1][0]

    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(Lin_index, Lin_df, test_size=size, shuffle=False) 

    

    # transform our data for polynomial regression

    poly = PolynomialFeatures(degree=deg)

    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

    poly_future_forcast = poly.fit_transform(future_forcast)

    

    # polynomial regression

    linear_model = LinearRegression(normalize=True, fit_intercept=False)

    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)

    test_linear_pred = linear_model.predict(poly_X_test_confirmed)

    linear_pred = linear_model.predict(poly_future_forcast)

    return linear_pred, y_test_confirmed, test_linear_pred, mae, deg
confirmed_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

brazil_df = pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")
brazil_df.head()
cols = confirmed_df.keys()

brazil_df['date'] = pd.to_datetime(brazil_df['date'])
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()

world_cases = []

brazil_cases = [] 

italy_cases = []

us_cases = [] 

spain_cases = [] 

brazil_deaths = [] 



for i in dates:

    confirmed_sum = confirmed[i].sum()

    world_cases.append(confirmed_sum)

    brazil_deaths.append(deaths_df[deaths_df['Country/Region']=='Brazil'][i].sum())

    brazil_cases.append(confirmed_df[confirmed_df['Country/Region']=='Brazil'][i].sum())

    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())

    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())

    spain_cases.append(confirmed_df[confirmed_df['Country/Region']=='Spain'][i].sum())
def daily_increase(data):

    d = [] 

    for i in range(len(data)):

        if i == 0:

            d.append(data[0])

        else:

            d.append(data[i]-data[i-1])

    return d 



brazil_daily_increase = daily_increase(brazil_cases)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)
days_in_future = 7
adjusted_dates = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)[:-days_in_future]

start = '22/1/2020'

start_date = datetime.datetime.strptime(start, '%d/%m/%Y')

adjusted_dates_dates = []

for i in range(len(adjusted_dates)):

    adjusted_dates_dates.append((start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y'))
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Confirmed cases over time"

)





fig = go.Figure(data=[

    

    go.Scatter(name='Brazil', x = adjusted_dates_dates, y=brazil_cases)

    , go.Scatter(name='Italy', x = adjusted_dates_dates, y=italy_cases)

    , go.Scatter(name='Spain', x = adjusted_dates_dates, y=spain_cases)

    , go.Scatter(name='US', x=adjusted_dates_dates, y=us_cases)

    ])



fig['layout'].update(layout)



fig.show()
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Confirmed cases over time around the world"

)





fig = go.Figure(data=[

    

    go.Scatter(name='World', x = adjusted_dates_dates, y=world_cases.reshape(-1))

    ])



fig['layout'].update(layout)



fig.show()
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Confirmed cases over time"

)





fig = go.Figure(data=[

    

    go.Scatter(name='Brazil', x = adjusted_dates_dates[34:], y=brazil_cases[34:])

    ])



fig['layout'].update(layout)



fig.show()
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Deaths over time"

)





fig = go.Figure(data=[

    

    go.Scatter(name='Deaths', x = adjusted_dates_dates[53:], y=brazil_deaths[53:])

    ])



fig['layout'].update(layout)



fig.show()
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Daily increase of confirmed cases in Brazil"

)





fig = go.Figure(data=[

    go.Bar(x=adjusted_dates_dates[34:], y=brazil_daily_increase[34:]),

    ])



fig['layout'].update(layout)



fig.show()
days_since_2_25 = np.array([i for i in range(len(days_since_1_22[34:]))]).reshape(-1, 1)

brazil_cases_since_2_25 = brazil_cases[34:]

future_forcast = np.array([i for i in range(len(days_since_2_25)+days_in_future)]).reshape(-1, 1)

linear_pred, test_data, reg_data, mae, deg  = Linear_Reg(days_since_2_25, brazil_cases_since_2_25, future_forcast)

print('Linear Regression mean absolute error:')

print(mae)

bayesian_pred, test_data_B, reg_data_B, mae_B, deg_B  = Bayesian_Reg(days_since_2_25, brazil_cases_since_2_25, future_forcast)

print('Bayesian Ridge Regression mean absolute error:')

print(mae_B)
start = '25/2/2020'

start_date = datetime.datetime.strptime(start, '%d/%m/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y'))
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Confirmed cases predictions over time"

)





fig = go.Figure(data=[

    

    go.Scatter(name='Confirmed cases', x = future_forcast_dates, y=brazil_cases_since_2_25)

    , go.Scatter(name='Polynomial Regression', x = future_forcast_dates, y=linear_pred, line = dict(dash='dot'))

    , go.Scatter(name='Polynomial Bayesian Ridge', x = future_forcast_dates, y=bayesian_pred, line = dict(dash='dash'))

    ])



fig['layout'].update(layout)



fig.show()
brazil_deaths_since_2_25 = brazil_deaths[53:]

linear_pred, test_data, reg_data, mae, deg  = Linear_Reg(days_since_2_25[19:], brazil_deaths_since_2_25, future_forcast[19:])

print('Linear Regression mean absolute error:')

print(mae)

bayesian_pred, test_data_B, reg_data_B, mae_B, deg_B  = Bayesian_Reg(days_since_2_25[19:], brazil_deaths_since_2_25, future_forcast[19:])

print('Bayesian Ridge Regression mean absolute error:')

print(mae_B)
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Deaths predictions over time"

)





fig = go.Figure(data=[

    

    go.Scatter(name='Deaths', x = future_forcast_dates[19:], y=brazil_deaths_since_2_25)

    , go.Scatter(name='Polynomial Regression', x = future_forcast_dates[19:], y=linear_pred, line = dict(dash='dot'))

    , go.Scatter(name='Polynomial Bayesian Ridge', x = future_forcast_dates[19:], y=bayesian_pred, line = dict(dash='dash'))

    ])



fig['layout'].update(layout)



fig.show()
other_states_df = brazil_df[['cases','deaths']][brazil_df['state']!='São Paulo'].groupby(brazil_df['date']).sum().sort_values(by = 'date', ascending=True)



sp_df = brazil_df[['cases','deaths']][brazil_df['state']=='São Paulo'].groupby(brazil_df['date']).sum().sort_values(by = 'date', ascending=True)



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Total cases in São Paulo vs other states"

)



fig = go.Figure(data=[

    

    go.Scatter(name='Other states', x=other_states_df.index[26:] , y=other_states_df['cases'][26:])

    , go.Scatter(name='São Paulo', x=sp_df.index[26:] , y=sp_df['cases'][26:])

    

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Deaths in São Paulo vs other states"

)



fig = go.Figure(data=[

    

    go.Scatter(name='Other states', x=other_states_df.index[27:] , y=other_states_df['deaths'][27:])

    , go.Scatter(name='São Paulo', x=sp_df.index[27:] , y=sp_df['deaths'][27:])

    

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()
sp_index = np.array([i for i in range(len(sp_df.index[41:]))]).reshape(-1, 1)

future_forcast = np.array([i for i in range(len(sp_index)+days_in_future)]).reshape(-1, 1) 

linear_pred, test_data, reg_data, mae, deg  = Linear_Reg(sp_index, sp_df['deaths'][41:], future_forcast)

print('Linear Regression mean absolute error:')

print(mae)

bayesian_pred, test_data_B, reg_data_B, mae_B, deg_B  = Bayesian_Reg(sp_index, sp_df['deaths'][41:], future_forcast)

print('Bayesian Ridge Regression mean absolute error:')

print(mae_B)
start = '11/3/2020'

start_date = datetime.datetime.strptime(start, '%d/%m/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y'))
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Deaths predictions over time"

)





fig = go.Figure(data=[

    

    go.Scatter(name='Deaths', x = future_forcast_dates, y=sp_df['deaths'][41:])

    , go.Scatter(name='Polynomial Regression', x = future_forcast_dates, y=linear_pred, line = dict(dash='dot'))

    , go.Scatter(name='Polynomial Bayesian Ridge', x = future_forcast_dates, y=bayesian_pred, line = dict(dash='dash'))

    ])



fig['layout'].update(layout)



fig.show()
sp_index = np.array([i for i in range(len(sp_df.index[26:]))]).reshape(-1, 1)

future_forcast = np.array([i for i in range(len(sp_index)+days_in_future)]).reshape(-1, 1)

linear_pred, test_data, reg_data, mae, deg  = Linear_Reg(sp_index, sp_df['cases'][26:], future_forcast)

print('Linear Regression mean absolute error:')

print(mae)

bayesian_pred, test_data_B, reg_data_B, mae_B, deg_B  = Bayesian_Reg(sp_index, sp_df['cases'][26:], future_forcast)

print('Bayesian Ridge Regression mean absolute error:')

print(mae_B)
start = '26/3/2020'

start_date = datetime.datetime.strptime(start, '%d/%m/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y'))
layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Confirmed cases predictions over time"

)





fig = go.Figure(data=[

    

    go.Scatter(name='Cases', x = future_forcast_dates, y=sp_df['cases'][26:])

    , go.Scatter(name='Polynomial Regression', x = future_forcast_dates, y=linear_pred, line = dict(dash='dot'))

    , go.Scatter(name='Polynomial Bayesian Ridge', x = future_forcast_dates, y=bayesian_pred, line = dict(dash='dash'))

    ])



fig['layout'].update(layout)



fig.show()