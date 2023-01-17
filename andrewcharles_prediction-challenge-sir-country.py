# https://population.un.org/wpp/Download/Standard/Population/

import os

import pandas as pd

import numpy as np

pd.set_option("display.precision", 3)

pd.set_option("display.expand_frame_repr", False)

pd.set_option("display.max_rows", 25)

import matplotlib.pyplot as plt

from scipy import integrate, optimize
dftrain = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

popdf = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')

print(dftrain.columns)

print(set(list(dftrain['Country/Region'])))
confirmed_by_date = dftrain.groupby(['Date','Country/Region'])[['ConfirmedCases']].agg("sum")

country_cols = confirmed_by_date['ConfirmedCases'].unstack()

#country_cols.head()
# Find the row where the number of cases is closest to ten.

# This indicates the start of the outbreak. We could use a more sophisticated method for this.

COL = 'Italy'

COL = "China"

COL = 'Australia'

val = 10

K1 = confirmed_by_date.groupby(['Country/Region']).agg(

    iK = pd.NamedAgg(column='ConfirmedCases', aggfunc=lambda x: abs(x-val).idxmin() )

)

# How to get this index for a particular country

idx = K1.loc['Italy']

val_index = confirmed_by_date.loc[idx].index
population = popdf.loc[popdf['Country (or dependency)'] == COL]['Population (2020)'].values[0]

dates = pd.DataFrame(country_cols[COL].reset_index()['Date'])

v_index = dates.loc[dates['Date']==val_index[0][0]].index.values[0]

v_index

a = country_cols[COL].copy()[v_index:]

a_reset = a.reset_index()

ydf = a_reset[COL].fillna(value=0)

x = ydf.index.values

y = ydf.values

N = population

inf0 = y[0] # starting population

sus0 = N - inf0 # susceptibles

rec0 = 0.0 # recovered (initial)

print(population)
def sir_model(y, x,beta, gamma, N=population):

    """ beta -- transistion rate S -> I (contact rate)

        gamma -- transition rate I -> R (recovery/mortality)

        k -- network parameter

        y1 = infected

        y0 = susceptible

        N -- total population

    """

    sus = -beta * y[0] * y[1] / N

    rec = gamma * y[1]

    inf = -(sus + rec)

    return sus, inf, rec



def fit_odeint(x, beta, gamma):

    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]
popt, pcov = optimize.curve_fit(fit_odeint, x, y)
#popt = np.array([0.08,0.05])

fitted = fit_odeint(x, *popt)

plt.plot(x, y, 'o')

plt.plot(x, fitted)

plt.title("Fit of SIR model to global infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
dftest = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

dates = dftest.groupby(['Country/Region','Date']).agg('mean')

dates_new = dates.index.droplevel().values

x_new = np.arange(0,len(x)+len(dates_new))

#x_new = np.arange(0,100)
x_new
#popt = np.array([0.7,0.2])

y_pred = fit_odeint(x_new, *popt)

# Prediction

#plt.plot(x, y, 'o')

plt.plot(x_new, y_pred)

plt.title("Fit of SIR model to global infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
y_pred.max(),population