import numpy as np

import pandas as pd

import scipy.optimize as opt

from scipy.special import expit

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train_range = (train.Date.min(), train.Date.max())

train_days = train.Date.nunique()

print('training data: {} - {} ({} days)'.format(*train_range, train_days))



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test_range = (test.Date.min(), test.Date.max())

test_days = test.Date.nunique()

print('testing data: {} - {} ({} days)'.format(*test_range, test_days))



overlap = (test.Date.min(), train.Date.max())

n_overlap = len(set(test.Date) & set(train.Date))

print('train/test overlap: {} - {} ({} days)'.format(*overlap, n_overlap))



train.Province_State = train.Province_State.astype('category')

train.Country_Region = train.Country_Region.astype('category')

train.Date = train.Date.astype('datetime64')

train.ConfirmedCases = train.ConfirmedCases.astype('int')

train.Fatalities = train.Fatalities.astype('int')

train = train.set_index('Id')



test.Province_State = test.Province_State.astype('category')

test.Country_Region = test.Country_Region.astype('category')

test.Date = test.Date.astype('datetime64')

test = test.set_index('ForecastId')



submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

submission = submission.set_index('ForecastId')



validation = train[train.Date >= min(test.Date)]

train = train[train.Date < min(test.Date)]
province_marker = lambda row: '(' + row.Region + ')'



def scatter_colonies(df):

    """Free regions and dependencies from their countries."""

    provinces = df[df.Province_State.notna()]

    provinces = provinces.rename(columns={'Province_State': 'Region'})

    provinces.Region = provinces.apply(province_marker, axis=1)

    del provinces['Country_Region']

    others = df[df.Province_State.isna()]

    others = others.rename(columns={'Country_Region': 'Region'})

    del others['Province_State']

    union = pd.concat([provinces, others])

    index_name = df.index.name

    return union.sort_values([index_name, 'Date'])



validation = scatter_colonies(validation)

train = scatter_colonies(train)

test  = scatter_colonies(test)
popfacts = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

new_headers = {'Country (or dependency)': 'Region', 'Population (2020)': 'Population2020'}

popfacts = popfacts.rename(columns=new_headers)[['Region', 'Population2020']]

popfacts.Region = popfacts.Region.astype('category')



constituent_states = pd.read_csv('/kaggle/input/population-of-provinces-and-states-for-covid19/population.csv')

constituent_states.State = constituent_states.State.astype('category')

new_headers = {'State': 'Region', 'Pop': 'Population2020'}

constituent_states = constituent_states.rename(columns=new_headers)

constituent_states.Region = constituent_states.apply(province_marker, axis=1)

popfacts = pd.concat([popfacts, constituent_states])

popfacts = popfacts.sort_values(by='Population2020', ascending=False)

popfacts = popfacts.set_index('Region')

popfacts.loc['Burma'] = popfacts.loc['Myanmar']

popfacts.loc['Congo (Brazzaville)'] = popfacts.loc['Congo']

popfacts.loc['Congo (Kinshasa)'] = popfacts.loc['DR Congo']

popfacts.loc['Cote d\'Ivoire'] = popfacts.loc['Côte d\'Ivoire']

popfacts.loc['Czechia'] = popfacts.loc['Czech Republic (Czechia)']

popfacts.loc['Diamond Princess'] = 3711

popfacts.loc['Korea, South'] = popfacts.loc['South Korea']

popfacts.loc['Kosovo'] = 1810463

popfacts.loc['MS Zaandam'] = 1829

popfacts.loc['US'] = popfacts.loc['United States']

popfacts.loc['Taiwan*'] = popfacts.loc['Taiwan']

popfacts.loc['Saint Kitts and Nevis'] = popfacts.loc['Saint Kitts & Nevis']

popfacts.loc['West Bank and Gaza'] = 5072990

popfacts.loc['Saint Vincent and the Grenadines'] = popfacts.loc['St. Vincent & Grenadines']

popfacts.loc['Sao Tome and Principe'] = 219159
def top_countries_by(column, n=-1, df=train):

    """Get the top countries with respect to a column."""

    top_rows_by_country = df.groupby('Region').max()[column]

    sorted_rows = top_rows_by_country.sort_values(ascending=False)

    top_countries = sorted_rows[:n].index

    return top_countries.to_list()



top_deaths = top_countries_by('Fatalities', 5)

top_ncases = top_countries_by('ConfirmedCases', 5)

top_countries = set(top_deaths) | set(top_ncases)
def column_over_time(column, threshold=0, df=train):

    """Get the values of a column over time, per country."""

    pivot_args = {'values': column,

        'index': 'Date',

        'columns': 'Region'}

    included = df[df[column] >= threshold].Region.unique()

    pivoted_table = df.pivot_table(**pivot_args)[included]

    return pivoted_table



def plot_countries(countries, stat=None, df=train):

    """Plot infected and removed individuals for specified countries."""

    conf_cases = column_over_time('ConfirmedCases', df=df)[countries]

    fatalities = column_over_time('Fatalities', df=df)[countries]

    if stat is None:

        for c in countries:

            conf_cases_c = conf_cases[c]

            fatalities_c = fatalities[c]

            days = range(1, len(conf_cases_c) + 1)

            p1 = plt.plot(days, conf_cases_c, ls='-', label=c)

            color = p1[0].get_color()

            p2 = plt.plot(days, fatalities_c, ls='--', c=color)

    else:

        conf_cases_avg = conf_cases.apply(stat, axis=1)

        fatalities_avg = fatalities.apply(stat, axis=1)

        days = range(1, len(conf_cases) + 1)

        plt.plot(days, conf_cases_avg, label='confirmed cases')

        plt.plot(days, fatalities_avg, label='fatalities')

    plt.legend()

    plt.show()



plot_countries(top_countries)



train_countries = train.Region.unique()

plot_countries(train_countries, stat='sum')
def forecast(data, steps=1, country=None):

    if sum(data > 0) <= 3:

        return np.array([data[-1]] * steps)



    past = list(range(sum(data == 0), len(data)))

    future = list(range(len(data), len(data) + steps))



    data = data[data > 0]

    msle = mean_squared_log_error

    sigmoid = lambda t, M, beta, alpha: M * expit(beta * (t - alpha))

    cost = lambda params: np.sqrt(msle(data, sigmoid(past, *params)))



    pop = popfacts.loc[country, 'Population2020']

    guess = [min(pop, 1000 + 0.0001 * pop), 0.25, 80]



    bounds = [(1, pop), (0, None), (0, None)]

    opt_params = opt.minimize(cost, x0=guess, bounds=bounds).x

    prediction = sigmoid(future, *opt_params)



    return np.floor(prediction)



for country in train.Region.unique():

    country_data = train[train.Region == country]

    country_data = country_data.sort_values(by='Date')

    country_location = test.Region == country

    country_test_data = test[country_location].sort_values(by='Date')

    conf_cases = country_data.ConfirmedCases.values

    fatalities = country_data.Fatalities.values

    forecast_days = len(country_test_data)

    conf_cases_pred = forecast(conf_cases, forecast_days, country=country)

    fatalities_pred = forecast(fatalities, forecast_days, country=country)

    submission.ConfirmedCases[country_test_data.index] = conf_cases_pred

    submission.Fatalities[country_test_data.index] = fatalities_pred



submission.to_csv('submission.csv')
predictions = test.copy()

predictions['ConfirmedCases'] = submission.ConfirmedCases

predictions['Fatalities'] = submission.Fatalities



top_ncases = top_countries_by('ConfirmedCases', 3, df=predictions)

top_deaths = top_countries_by('Fatalities', 3, df=predictions)

top_countries = set(top_deaths) | set(top_ncases)

print('top countries:', top_countries)

plot_countries(top_countries, df=predictions)

plot_countries(train_countries, stat='sum', df=predictions)
overlap_dates = predictions.Date <= validation.Date.max()

cases_true = column_over_time('ConfirmedCases', df=validation)[top_countries]

cases_pred = column_over_time('ConfirmedCases', df=predictions[overlap_dates])[top_countries]

for c in top_countries:

    cases_true_c = cases_true[c]

    cases_pred_c = cases_pred[c]

    days = range(1, len(cases_true_c) + 1)

    p1 = plt.plot(days, cases_true_c, ls='-', label=c)

    color = p1[0].get_color()

    p2 = plt.plot(days, cases_pred_c, ls=':', c=color)

plt.legend()

plt.show()