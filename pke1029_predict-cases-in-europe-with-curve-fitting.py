import numpy as np

import pandas as pd

# from datetime import date, timedelta

from scipy.optimize import curve_fit

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = [12, 7]

matplotlib.rcParams.update({'font.size': 16})
# read csv

df_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

# add continent info

df_country = pd.read_csv('../input/country-to-continent/countryContinent.csv', encoding = 'ISO-8859-1')

df_country = df_country[['country', 'sub_region', 'continent']]

df_confirmed = pd.merge(df_country, df_confirmed, left_on = 'country', right_on = 'Country/Region')

del df_confirmed['Country/Region']

df_confirmed.head()
# group country by continent

df_confirmed_continent = df_confirmed.groupby('continent').sum()

df_confirmed_continent = df_confirmed_continent.loc[:, '1/22/20':]

display(df_confirmed_continent.head())

df_confirmed_continent.transpose().plot();
def logistic(x, x0, L, k):

    return L / (1 + np.exp(-k*(x-x0)))
# initialize x axis

nrow, ncol = df_confirmed_continent.shape

x = np.array(range(ncol))

x2 = np.array(range(2*ncol))



# confirmed cases in Europe

y_confirmed = df_confirmed_continent.loc['Europe'].to_numpy()

plt.scatter(x, y_confirmed, label = 'Europe-confirmed');



# fit model

popt_c, pcov_c = curve_fit(logistic, x, y_confirmed)

perr_c = np.sqrt(np.diag(pcov_c)) # std dev of fit parameter

print(popt_c)

# set curve maximum value

curve_max_value = popt_c[1]



# prediction

Y_confirmed = logistic(x2, *popt_c)

plt.fill_between(x2, logistic(x2, *popt_c + perr_c), logistic(x2, *popt_c - perr_c), alpha = 0.2, label = '1 sigma', color = 'r')

plt.plot(x2, Y_confirmed, 'r', label = 'logistic');

plt.legend();

# similar process with recovered time series

df_recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

df_recovered = pd.merge(df_country, df_recovered, left_on = 'country', right_on = 'Country/Region')

del df_recovered['Country/Region']

df_recovered_continent = df_recovered.groupby('continent').sum()

df_recovered_continent = df_recovered_continent.loc[:, '1/22/20':]

display(df_recovered_continent.head())

df_recovered_continent.transpose().plot();
def recovery_model(x, x0, p):

    return logistic(x, x0, curve_max_value, p)
# recovered cases in Europe

y_recovered = df_recovered_continent.loc['Europe'].to_numpy()

plt.scatter(x, y_recovered, label = 'Europe-recovered');



# fit model

popt_r, pcov_r = curve_fit(recovery_model, x, y_recovered, bounds = (0, [120, 10]))

perr_r = np.sqrt(np.diag(pcov_r)) # std dev of fit parameter

print(popt_r)



# predict

Y_recovered = recovery_model(x2, *popt_r)

plt.plot(x2, Y_recovered, 'r', label = 'logistic');

plt.fill_between(x2, recovery_model(x2, *popt_r + perr_r), recovery_model(x2, *popt_r - perr_r), alpha = 0.2, label = '1 sigma', color = 'r')

plt.legend();



# combining the two model

plt.scatter(x, y_confirmed - y_recovered, label = 'Europe-infected');

plt.plot(x2, Y_confirmed - Y_recovered, 'r', label = 'model');



plt.fill_between(x2, 

                 logistic(x2, *popt_c + perr_c) - recovery_model(x2, *popt_r - perr_r), # +1 sd away

                 logistic(x2, *popt_c - perr_c) - recovery_model(x2, *popt_r + perr_r), # -1 sd away

                 alpha = 0.2,

                 label = '1 sigma',

                 color = 'r')

plt.legend();
df_confirmed_europe = df_confirmed[df_confirmed['continent'] == 'Europe']

df_confirmed_europe = df_confirmed_europe.sort_values(by = df_confirmed_europe.columns[-1], ascending = False)

index = df_confirmed_europe.index

df_confirmed_europe = df_confirmed_europe.set_index('country')

df_confirmed_europe = df_confirmed_europe.head().loc[:, '1/22/20':]

display(df_confirmed_europe.head())

df_confirmed_europe.head().transpose().plot();
df_recovered_europe = df_recovered.iloc[index]

df_recovered_europe = df_recovered_europe.set_index('country')

df_recovered_europe = df_recovered_europe.head().loc[:, '1/22/20':]

df_recovered_europe.head()
def fit_model(y_confirmed, y_recovered):

    popt_c, pcov_c = curve_fit(logistic, x, y_confirmed)

    curve_max_value = popt_c[1]

    perr_c = np.sqrt(np.diag(pcov_c))

    Y_confirmed = logistic(x2, *popt_c)

    

    def recovery_model(x, x0, p):

        return logistic(x, x0, curve_max_value, p)

    

    popt_r, pcov_r = curve_fit(recovery_model, x, y_recovered, bounds = (0, [120, 10]))

    perr_r = np.sqrt(np.diag(pcov_r))

    Y_recovered = recovery_model(x2, *popt_r)

    plt.scatter(x, y_confirmed - y_recovered)

    plt.plot(x2, Y_confirmed - Y_recovered)

    plt.fill_between(x2, 

                     logistic(x2, *popt_c + perr_c) - recovery_model(x2, *popt_r - perr_r), # +1 sd away

                     logistic(x2, *popt_c - perr_c) - recovery_model(x2, *popt_r + perr_r), # -1 sd away

                     alpha = 0.1,

                     color = 'k')
n = 3

for i in range(n):

    y_confirmed = df_confirmed_europe.iloc[i].to_numpy()

    y_recovered = df_recovered_europe.iloc[i].to_numpy()

    fit_model(y_confirmed, y_recovered)

plt.legend(df_confirmed_europe.index[0:3]);