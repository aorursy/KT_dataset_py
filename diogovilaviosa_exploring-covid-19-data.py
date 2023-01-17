import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import numpy as np

import math

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Main file in this dataset is covid_19_data.csv and the detailed descriptions are below.

#Sno - Serial number

#ObservationDate - Date of the observation in MM/DD/YYYY

#Province/State - Province or state of the observation (Could be empty when missing)

#Country/Region - Country of observation

#Last Update - Time in UTC at which the row is updated for the given province or country. (Not standardised and so please clean before using it)

#Confirmed - Cumulative number of confirmed cases till that date

#Deaths - Cumulative number of of deaths till that date

#Recovered - Cumulative number of recovered cases till that date

main_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

main_data.tail()
confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')



confirmed['location'] = confirmed['Country/Region'].fillna(value='NAN') + '_' + confirmed['Province/State'].fillna(value='NAN')

recovered['location'] = recovered['Country/Region'].fillna(value='NAN') + '_' + recovered['Province/State'].fillna(value='NAN')

deaths['location'] = deaths['Country/Region'].fillna(value='NAN') + '_' + deaths['Province/State'].fillna(value='NAN')
plot_conf = np.sum(confirmed.iloc[:,4:confirmed.shape[1]-1])

plot_reco = np.sum(recovered.iloc[:,4:recovered.shape[1]-1])

plot_dead = np.sum(deaths.iloc[:,4:deaths.shape[1]-1])

plot_infe = plot_conf - plot_reco - plot_dead



plot_mort = (plot_dead / plot_conf)

day_ndx = [i for i in range(len(plot_conf))]

n_days = len(plot_conf)
sns.lineplot(x = day_ndx, y = plot_mort, label="mortality")
sns.lineplot(x = day_ndx, y = plot_conf, label="confirmed")

sns.lineplot(x = day_ndx, y = plot_infe, label="infected")

sns.lineplot(x = day_ndx, y = plot_reco, label="recovered")

sns.lineplot(x = day_ndx, y = plot_dead, label="deaths")
locations = ['Portugal', 'Spain', 'France', 'Germany', 'Belgium', 'Netherlands', 'Italy', 'Switzerland', 'Luxembourg', 'Austria']

plt.figure(figsize=(12,10))

for loc in locations:

    plot_conf = np.sum(confirmed.loc[confirmed['location'] == loc+'_NAN'].iloc[:,4:confirmed.shape[1]-1])

    if plot_conf[0] == 0 and plot_conf[-1] > 0:

        shifted = [i for i in plot_conf if i > 0]

        print(loc, shifted[0])

        x_data = day_ndx[0:len(shifted)]

        sns.lineplot(x = x_data, y = shifted, label=loc)

plt.ylim(0,500)
def growth(t, a, b):

    return a*b**t
plt.figure(figsize=(12,10))

plt.ylim(0,1000)

loc = 'Portugal'

plot_conf = np.sum(confirmed.loc[confirmed['location'] == loc+'_NAN'].iloc[:,4:confirmed.shape[1]-1])

shifted = [i for i in plot_conf if i > 0]

print(loc, shifted[0])

x_data = day_ndx[0:len(shifted)]

sns.lineplot(x = x_data, y = shifted, label='Confirmed')



params = curve_fit(growth, x_data, shifted)



x_new = np.linspace(0,20,1000)

y_new = growth(x_new,params[0][0],params[0][1])

sns.lineplot(x = x_new, y = y_new, label='Prediction')

print(params[0][0],params[0][1])