import numpy as np 

import pandas as pd 

from scipy.optimize import curve_fit
# import graph objects as "go"

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
confirmed_df.head()
# drop Lat and Long columns

cols = ["Lat", "Long"]

confirmed_df = confirmed_df.drop(cols, axis=1)

deaths_df = deaths_df.drop(cols, axis=1)
def get_country_data(df, country_name):

    sum_country = df[df["Country/Region"] == country_name].groupby(["Country/Region"]).sum()

    if sum_country.iloc[0]["3/11/20"] == sum_country.iloc[0]["3/12/20"]:

        sum_country.iloc[0]["3/12/20"] = (sum_country.iloc[0]["3/11/20"] + sum_country.iloc[0]["3/13/20"]) / 2

    return sum_country
croatia_df = get_country_data(confirmed_df, "Croatia")
x = np.arange(len(croatia_df.keys()))

y_real = croatia_df.iloc[0]
DAYS_IN_FUTURE=7



def exp_func(x, a, b, c, d):

    return a * np.exp(-b * x + c) + d



popt, pcov = curve_fit(exp_func, x, y_real, p0=(1, -0.07, 0, 0))

print(popt)

print(pcov)

x_for_fit_graph = np.linspace(0, x[-1] + DAYS_IN_FUTURE, (len(x) + DAYS_IN_FUTURE - 1) * 5 + 1)

y_fitted = exp_func(x_for_fit_graph, *popt)
trace1 = go.Scatter(

            name="Data",

            mode="lines+markers",

            x=x,

            y=y_real,

            text=croatia_df.keys()

        )

trace2 = go.Scatter(

            name="Fit",

            mode="lines",

            x=x_for_fit_graph,

            y=y_fitted

        )

data = [trace1, trace2]

# specify the layout of our figure

layout = dict(title = "Confirmed cases in Croatia",

              xaxis= dict(title='Days from 22/1',zeroline=True))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
# Italy

country_df = get_country_data(confirmed_df, "Italy")

x = np.arange(len(country_df.keys()))

y_real = country_df.iloc[0]



DAYS_IN_FUTURE=7



def exp_func(x, a, b, c, d):

    return a * np.exp(-b * x + c) + d



popt, pcov = curve_fit(exp_func, x, y_real, p0=(1, -0.15, 0, 0))

print(popt)

print(pcov)

x_for_fit_graph = np.linspace(0, x[-1] + DAYS_IN_FUTURE, (len(x) + DAYS_IN_FUTURE - 1) * 5 + 1)

y_fitted = exp_func(x_for_fit_graph, *popt)



# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

trace1 = go.Scatter(

            name="Data",

            mode="lines+markers",

            x=x,

            y=y_real,

            text=croatia_df.keys()

        )

trace2 = go.Scatter(

            name="Fit",

            mode="lines",

            x=x_for_fit_graph,

            y=y_fitted

        )

data = [trace1, trace2]

# specify the layout of our figure

layout = dict(title = "Confirmed cases in Italy",

              xaxis= dict(title='Days from 22/1',zeroline=True))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
def plot_new_cases(country_name):

    country_conf_df = get_country_data(confirmed_df, country_name)

    country_deth_df = get_country_data(deaths_df, country_name)

    

    # new cases

    new_cases_conf = np.diff(country_conf_df.values[0], prepend=[0])

    new_cases_deth = np.diff(country_deth_df.values[0], prepend=[0])

    

    trace1 = go.Scatter(

        name="New Confirmed",

        mode="lines+markers",

        x=x,

        y=new_cases_conf,

        text=country_conf_df.keys()

    )

    trace2 = go.Scatter(

        name="New Deaths",

        mode="lines+markers",

        x=x,

        y=new_cases_deth,

        text=country_conf_df.keys(),

        yaxis='y2'

    )

    data = [trace1, trace2]

    

    # specify the layout of our figure

    layout = dict(title = "New cases in {}".format(country_name),

                  xaxis = dict(title='Days from 22/1',zeroline=True),

                  yaxis = dict(title='New confirmed cases', zeroline=True),  # type=log

                  yaxis2 = dict(title='New Deaths', overlaying='y', side='right')

                 )



    # create and show our figure

    fig = dict(data = data, layout = layout)

    iplot(fig)
plot_new_cases("Croatia")
plot_new_cases("Italy")
plot_new_cases("China")
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()

world_cases = []

total_deaths = [] 

mortality_rate = []

total_recovered = [] 



for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    mortality_rate.append(death_sum/confirmed_sum)

    total_recovered.append(recovered_sum)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 5

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-5]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False) 
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
linear_model = LinearRegression(normalize=True, fit_intercept=True)

linear_model.fit(X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(X_test_confirmed)

linear_pred = linear_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)

print(linear_model.intercept_)
plt.plot(y_test_confirmed)

plt.plot(test_linear_pred)
tol = [1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}



bayesian = BayesianRidge()

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

bayesian_search.fit(X_train_confirmed, y_train_confirmed)
bayesian_search.best_params_