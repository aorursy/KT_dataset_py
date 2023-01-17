import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# importing datasets

full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.head()
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

# full_table[cases] = full_table[cases].fillna(0)
def get_time_series(country):

    # for some countries, data is spread over several Provinces

    if full_table[full_table['Country/Region'] == country]['Province/State'].nunique() > 1:

        country_table = full_table[full_table['Country/Region'] == country]

        country_df = pd.DataFrame(pd.pivot_table(country_table, values = ['Confirmed'],

                              index='Date', aggfunc=sum).to_records())

        return country_df.set_index('Date')[['Confirmed']]

    df = full_table[(full_table['Country/Region'] == country) 

                & (full_table['Province/State'].isin(['', country]))]

    return df.set_index('Date')[['Confirmed']]
country = 'New Zealand'

df = get_time_series(country)

if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:

    df.drop(df.tail(1).index,inplace=True)

df.tail()
country = 'France'

df = get_time_series(country)

if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:

    df.drop(df.tail(1).index,inplace=True)

df.tail()
import math

def model_with_lag(N, a, alpha, lag, t):

    # we enforce N, a and alpha to be positive numbers using min and max functions

    lag = min(max(lag, -100), 100) # lag must be less than +/- 100 days 

    return max(N, 0) * (1 - math.e ** (min(-a, 0) * (t - lag))) ** max(alpha, 0)



def model(N, a, alpha, t):

    return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)


def model_loss(params):

#     N, a, alpha, lag = params

    N, a, alpha = params

    model_x = []

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t) - df.iloc[t, 0]) ** 2

#         r += (math.log(1 + model(N, a, alpha, t)) - math.log(1 + df.iloc[t, 0])) ** 2 

#         r += (model_with_lag(N, a, alpha, lag, t) - df.iloc[t, 0]) ** 2

#         print(model(N, a, alpha, t), df.iloc[t, 0])

    return math.sqrt(r) 
import numpy as np

from scipy.optimize import minimize

use_lag_model = False

if use_lag_model:

    opt = minimize(model_loss, x0=np.array([200000, 0.05, 15, 0]), method='Nelder-Mead', tol=1e-5).x

else:

    opt = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

opt
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline 



model_x = []

for t in range(len(df)):

    model_x.append([df.index[t], model(*opt, t)])

model_sim = pd.DataFrame(model_x, dtype=int)

model_sim.set_index(0, inplace=True)

model_sim.columns = ['model']

pd.concat([model_sim, df], axis=1).plot()

plt.show()
import datetime

start_date = df.index[0]

n_days = 210

extended_model_x = []

for t in range(n_days):

    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt, t)])

extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)

extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['model']

pd.concat([extended_model_sim, df], axis=1).plot()

plt.show()
df.tail()
pd.options.display.float_format = '{:20,.0f}'.format

concat_df = pd.concat([df, extended_model_sim], axis=1)

concat_df[concat_df.index.day % 7 == 0].astype({'model': 'int32'})
def display_fit(df, opt, ax):

    model_x = []

    for t in range(len(df)):

        model_x.append([df.index[t], model(*opt, t)])

    model_sim = pd.DataFrame(model_x, dtype=int)

    model_sim.set_index(0, inplace=True)

    model_sim.columns = ['model']

    return pd.concat([model_sim, df], axis=1).plot(ax=ax, figsize=(12, 8))



def display_extended_curve(df, opt, ax):

    start_date = df.index[0]

    n_days = 200

    extended_model_x = []

    for t in range(n_days):

        extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt, t)])

    extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)

    extended_model_sim.set_index(0, inplace=True)

    extended_model_sim.columns = ['model -> ' + str(int(opt[0]))]

    return pd.concat([extended_model_sim, df], axis=1).plot(ax=ax, figsize=(12, 8))



stats = []

for country in full_table['Country/Region'].unique():

# for country in ['Sweden']:

    df = get_time_series(country)

    # only consider countries with at least 5000 cases (plus Sweden)

    if len(df) == 0 or (max(df['Confirmed']) < 5000 and country != 'Sweden'): 

        continue

    df.columns = [df.columns[0] + ' ' + country]

    # if the last data point repeats the previous one, or is lower, drop it

    if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:

        df.drop(df.tail(1).index,inplace=True)

#     if country == 'France':

#         display(df.tail())

    opt = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

#     print(country, opt)

    if min(opt) > 0:

        stats.append([country, *opt])

        n_plot = len(stats)

        plt.figure(1)

        ax1 = plt.subplot(221)

        display_fit(df, opt, ax1)

        ax2 = plt.subplot(222)

        display_extended_curve(df, opt, ax2)

        plt.show()

stats_df = pd.DataFrame(stats)

# stats_df.columns = ['country', 'N', 'a', 'alpha', 'lag']

stats_df.columns = ['country', 'N', 'a', 'alpha']
pd.set_option('display.max_rows', 500)

pd.options.display.float_format = '{:20,.4f}'.format

stats_df.astype({'N': 'int'}).sort_values(by='N', ascending=False)
ax = stats_df.plot.scatter(x='alpha', y='a')

# ax.set_xlim([0, 100])

plt.show()