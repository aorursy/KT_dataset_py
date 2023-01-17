import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # For Viz

import plotly.express as px

import plotly

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.preprocessing import PolynomialFeatures

import datetime as dt

'''

sklearn.linear_model.LinearRegression



'''





%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

confirmed_dates = df_confirmed.columns[4:]

cumulative_confirmed_data = []



for date in confirmed_dates:

    cumulative_confirmed_data.append(df_confirmed[date].sum())



confirmed_cases_per_day = [cumulative_confirmed_data[0]]

for i in range(1, len(cumulative_confirmed_data)):

    confirmed_cases_per_day.append(cumulative_confirmed_data[i] - cumulative_confirmed_data[i - 1])



days = len(confirmed_dates)



fig = make_subplots(rows=1, cols=2)

fig.append_trace(go.Scatter(x=confirmed_dates, y=cumulative_confirmed_data, text="Confirmed"), row=1, col=1)

fig.append_trace(go.Bar(x=confirmed_dates, y=confirmed_cases_per_day, text="Confirmed this day"), row=1, col=2)



fig.update_layout(height=600, width=1500, title_text =  "Number of cases confirmed total vs Each day")

fig.show()

df_confirmed.head()
df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

deaths_dates = df_deaths.columns[4:]

cumulative_deaths_data = []



for date in deaths_dates:

    cumulative_deaths_data.append(df_deaths[date].sum())



deaths_cases_per_day = [cumulative_deaths_data[0]]

for i in range(1, len(cumulative_deaths_data)):

    deaths_cases_per_day.append(cumulative_deaths_data[i] - cumulative_deaths_data[i - 1])



fig = make_subplots(rows=1, cols=2)

fig.append_trace(go.Scatter(x=deaths_dates, y=cumulative_deaths_data, text="Deaths"), row=1, col=1)

fig.append_trace(go.Bar(x=deaths_dates, y=deaths_cases_per_day, text="Deaths this day"), row=1, col=2)



fig.update_layout(height=600, width=1500, title_text="Deaths Cumulative vs Each Day")

fig.show()
df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

recovered_dates = df_recovered.columns[4:]

cumulative_recovered_data = []



for date in recovered_dates:

    cumulative_recovered_data.append(df_recovered[date].sum())



recovered_cases_per_day = [cumulative_recovered_data[0]]

for i in range(1, len(cumulative_recovered_data)):

    recovered_cases_per_day.append(cumulative_recovered_data[i] - cumulative_recovered_data[i - 1])



fig = make_subplots(rows=1, cols=2)

fig.append_trace(go.Scatter(x=recovered_dates, y=cumulative_recovered_data, text="Recovered"), row=1, col=1)

fig.append_trace(go.Bar(x=recovered_dates, y=recovered_cases_per_day, text="Recovered this day"), row=1, col=2)



fig.update_layout(height=600, width=1500, title_text="Recovered Cumulative vs Each Day")

fig.show()

X_train = [[day] for day in range(1, days + 1)]



DAYS_TO_PREDICT = 20

DATE_FORMAT = "%m/%d/%Y"



X_test = [[day] for day in range(1, days + DAYS_TO_PREDICT)]



dates = []

for i in confirmed_dates:

    dates.append(i)

current_date = dt.datetime.strptime(dates[-1] + '20', DATE_FORMAT).date()



for i in range(DAYS_TO_PREDICT):

    date = current_date + dt.timedelta(days=i + 1)

    dates.append([date.strftime(DATE_FORMAT)[:-2]])



Y_confirmed = cumulative_confirmed_data

Y_recovered = cumulative_recovered_data

Y_deaths = cumulative_deaths_data



def Linear(X, Y):

    model = LinearRegression()

    model.fit(X, Y)

    return model



def Poly(X, degree=2):

    poly = PolynomialFeatures(degree)

    new_X = poly.fit_transform(X)

    return new_X



def RidgeRegression(X, Y, _alpha=1):

    model = Ridge(alpha=_alpha)

    model.fit(X, Y)

    return model



def LassoRegression(X, Y, _alpha=1):

    model = Lasso(alpha=_alpha)

    model.fit(X, Y)

    return model



def Elastic(X, Y):

    model = ElasticNet(random_state=0)

    model.fit(X, Y)

    return model



def Predict(model, X):

    return model.predict(X)



linear = Linear(X_train, Y_confirmed)

linear_prediction = Predict(linear, X_test)



POLYNOMIAL_DEGREE = 4



poly = Linear(Poly(X_train, POLYNOMIAL_DEGREE), Y_confirmed)

poly_prediction = Predict(poly, Poly(X_test, POLYNOMIAL_DEGREE))



RIDGE_DEGREE = 2

RIDGE_ALPHA = 1



ridge = RidgeRegression(Poly(X_train, RIDGE_DEGREE), Y_confirmed, RIDGE_ALPHA)

ridge_prediction = Predict(ridge, Poly(X_test, RIDGE_DEGREE))



LASSO_DEGREE = 7

LASSO_ALPHA = 1



lasso = LassoRegression(Poly(X_train, LASSO_DEGREE), Y_confirmed, LASSO_ALPHA)

lasso_prediction = Predict(lasso, Poly(X_test, LASSO_DEGREE))



fig = go.Figure()



fig.add_trace(go.Scatter(x=recovered_dates, y=cumulative_confirmed_data, name="Actual Confirmed cases"))

fig.add_trace(go.Scatter(x=dates, y=linear_prediction, name="Linear Regression"))

fig.add_trace(go.Scatter(x=dates, y=poly_prediction, name="Polynomial Regression"))

fig.add_trace(go.Scatter(x=dates, y=ridge_prediction, name="Ridge Regression"))

fig.add_trace(go.Scatter(x=dates, y=lasso_prediction, name="Lasso Regression"))





fig.update_layout(

    title={

        'text': "Comparision of future cases using various linear models",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
linear = Linear(X_train, Y_deaths)

linear_prediction = Predict(linear, X_test)



POLYNOMIAL_DEGREE = 5



poly = Linear(Poly(X_train, POLYNOMIAL_DEGREE), Y_deaths)

poly_prediction = Predict(poly, Poly(X_test, POLYNOMIAL_DEGREE))



RIDGE_DEGREE = 2

RIDGE_ALPHA = 1



ridge = RidgeRegression(Poly(X_train, RIDGE_DEGREE), Y_deaths, RIDGE_ALPHA)

ridge_prediction = Predict(ridge, Poly(X_test, RIDGE_DEGREE))



LASSO_DEGREE = 7

LASSO_ALPHA = 1



lasso = LassoRegression(Poly(X_train, LASSO_DEGREE), Y_deaths, LASSO_ALPHA)

lasso_prediction = Predict(lasso, Poly(X_test, LASSO_DEGREE))



fig = go.Figure()



fig.add_trace(go.Scatter(x=recovered_dates, y=cumulative_confirmed_data, name="Actual Deaths"))

fig.add_trace(go.Scatter(x=dates, y=linear_prediction, name="Linear Regression"))

fig.add_trace(go.Scatter(x=dates, y=poly_prediction, name="Polynomial Regression"))

fig.add_trace(go.Scatter(x=dates, y=ridge_prediction, name="Ridge Regression"))

fig.add_trace(go.Scatter(x=dates, y=lasso_prediction, name="Lasso Regression"))



fig.update_layout(

    title={

        'text': "Comparision of future deaths using various linear models",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
linear = Linear(X_train, Y_recovered)

linear_prediction = Predict(linear, X_test)



POLYNOMIAL_DEGREE = 10



poly = Linear(Poly(X_train, POLYNOMIAL_DEGREE), Y_recovered)

poly_prediction = Predict(poly, Poly(X_test, POLYNOMIAL_DEGREE))



RIDGE_DEGREE = 2

RIDGE_ALPHA = 1



ridge = RidgeRegression(Poly(X_train, RIDGE_DEGREE), Y_recovered, RIDGE_ALPHA)

ridge_prediction = Predict(ridge, Poly(X_test, RIDGE_DEGREE))



LASSO_DEGREE = 2

LASSO_ALPHA = 1



lasso = LassoRegression(Poly(X_train, LASSO_DEGREE), Y_recovered, LASSO_ALPHA)

lasso_prediction = Predict(lasso, Poly(X_test, LASSO_DEGREE))



fig = go.Figure()



fig.add_trace(go.Scatter(x=recovered_dates, y=cumulative_confirmed_data, name="Actual Recorvered cases"))

fig.add_trace(go.Scatter(x=dates, y=linear_prediction, name="Linear Regression"))

fig.add_trace(go.Scatter(x=dates, y=poly_prediction, name="Polynomial Regression"))

fig.add_trace(go.Scatter(x=dates, y=ridge_prediction, name="Ridge Regression"))

fig.add_trace(go.Scatter(x=dates, y=lasso_prediction, name="Lasso Regression"))





fig.update_layout(

    title={

        'text': "Comparision of future recovery using various linear models",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()
TOP_COUNTRY_COUNT = 10



country_wise_total = []

top_countries = []

for i in range(len(df_confirmed)):

    row = df_confirmed.iloc[i]

    country = row['Country/Region']

    confirmed_cases = row[-1]

    country_wise_total.append((country, confirmed_cases))



country_wise_total.sort(key=lambda x: x[-1], reverse=True)

df_country_wise = pd.DataFrame(country_wise_total, columns=['Country', 'Cases'])

# p = plt.pie(df_country_wise['Cases'][:10], labels=df_country_wise['Country'][:10])

px.pie(df_country_wise, names=df_country_wise['Country'][:TOP_COUNTRY_COUNT], values=df_country_wise['Cases'][:TOP_COUNTRY_COUNT], title='Top 10 contires contribution so far')
store = {}



for i in range(TOP_COUNTRY_COUNT):

    top_countries.append(country_wise_total[i][0])

    

for i in range(len(df_confirmed)):

    row = df_confirmed.iloc[i]

    if not (row[1] in top_countries):

        continue

    series = list(row[4:])

    country = row[1]

    store[country] = list(series)



df_top_countries = pd.DataFrame(store)

corr = df_top_countries.corr()

corr
px.imshow(corr, labels=dict(x="Countries", y="Countries", color="Correlation"), x = corr.columns, y = corr.columns, title="Similar spread comparision between contries")