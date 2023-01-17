# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from datetime import datetime

import scipy.optimize as opt



warnings.filterwarnings('ignore')



sub0 = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', parse_dates=['Date'])

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', parse_dates=['Date'])

train.shape, test.shape
print (train.Date.min(), train.Date.max())

print (test.Date.min(), test.Date.max())
# Get the results we already have

know = test[test.Date <= train.Date.max()]

not_know = test[test.Date > train.Date.max()]



know = know.merge(train, on=['Date','Country_Region','Province_State'], how='left')

know.head()
# Add columns

train['days'] = (train['Date'] - train.Date.min()).dt.days

train['location'] = train['Country_Region'] + ' ' + train['Province_State'].fillna('')

train['location'] = train['location'].str.strip()



not_know['days'] = (not_know['Date'] - train.Date.min()).dt.days

not_know['location'] = not_know['Country_Region'] + ' ' + not_know['Province_State'].fillna('')

not_know['location']= not_know['location'].str.strip()
train.location.unique()
# Fit logistic curve

def log_curve(x, k, x_0, ymax):

    return ymax / (1 + np.exp(-k*(x-x_0))) 



# Plot hypothetical logistic curve

x_hat = np.arange(0, 115)

y_hat = log_curve(x_hat, 0.2, 60, 1000)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(x_hat, y_hat, '-')
# Plot cases by each country



months_fmt = mdates.DateFormatter('%b-%e')



def plot_cty(num, evo_col, title):

    ax[num].plot(evo_col, lw=3)

    ax[num].set_title(title)

    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))

    ax[num].xaxis.set_major_formatter(months_fmt)

    ax[num].grid(True)



def evo_cty(country):

    evo_cty = train[train.Country_Region==country].groupby('Date')[['ConfirmedCases','Fatalities']].sum()

    evo_cty['Death Rate'] = evo_cty['Fatalities'] / evo_cty['ConfirmedCases'] * 100

    plot_cty(0, evo_cty['ConfirmedCases'], 'Confirmed cases')

    plot_cty(1, evo_cty['Fatalities'], 'Death cases')

    plot_cty(2, evo_cty['Death Rate'], 'Death rate')

    fig.suptitle(country, fontsize=13)

    plt.show()

    

fig, ax = plt.subplots(1, 3, figsize=(17,5))

evo_cty('US')
# Get initial parameter estimates

def get_param(loc):

    _ = train[train.location == loc]

    _['diff'] = _.ConfirmedCases.diff()

    _['pct'] = _.ConfirmedCases.pct_change()

    initial_speed = _.loc[_.ConfirmedCases.diff().argmax(),'pct']

    initial_mid = _.loc[_.ConfirmedCases.diff().argmax(), 'days']

    initial_max = _.ConfirmedCases.max() * 2.1

    return initial_speed, initial_mid, initial_max



get_param('Finland')
# Forecast confirmed cases



loc_list = train.location.unique()

all_param = pd.DataFrame(index=loc_list, columns=['k','x_0','y_max'])



for loc in loc_list:

    _ = train[train.location == loc]

    nn = not_know[not_know.location == loc]

    initial_max = _.ConfirmedCases.max()*2

    x = _.days

    y1 = _.ConfirmedCases

    try:

        popt, pcov = opt.curve_fit(log_curve, x, y1, p0 = get_param(loc))

        popt[2] = max(popt[2],get_param(loc)[2]/2) # y_max must be at least the latest confirmed cases

    except RuntimeError:

        popt = get_param(loc)    

    # print(loc, round(popt[0],2), round(popt[1], 0), round(popt[2],0))

    y1_hat = log_curve(nn.days, *popt)

    not_know.loc[y1_hat.index, 'ConfirmedCases'] = y1_hat

    all_param.loc[loc, 'k'] = popt[0]

    all_param.loc[loc, 'x_0'] = popt[1]

    all_param.loc[loc, 'y_max'] = popt[2]

print('Done')
all_param.loc['Finland', 'y_max']
# Plot prediction vs actual data

def plot_log_curve(location):

    _ = train[train.location == location]

    nn = not_know[not_know.location == location]

    x = _.days

    y1 = _['ConfirmedCases']

    #(k, x_0, ymax), _a = opt.curve_fit(log_curve, x, y1)

    #print(k, x_0, ymax)

    x_hat = pd.concat([x, nn['days']])

    y1_hat = log_curve(x_hat, all_param.loc[location, 'k'], all_param.loc[location, 'x_0'], all_param.loc[location, 'y_max'])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.plot(x, y1, 'o', markersize=3)

    ax.plot(x_hat, y1_hat, '-', lw=2)

plot_log_curve('US New York')
plot_log_curve('Finland')
plot_log_curve('Sweden')
plot_log_curve('Germany')
plot_log_curve('Israel')
plot_log_curve('Korea, South')
plot_log_curve('Timor-Leste')
plot_log_curve('Bhutan')
plot_log_curve('Burma')
# Forecast deaths

latest = train[train.Date == train.Date.max()]

latest['DeathRate'] = latest['Fatalities'] / latest['ConfirmedCases']

not_know2 = not_know.merge(latest[['location','DeathRate']], on='location')

not_know2['Fatalities'] = not_know2['ConfirmedCases'] * not_know2['DeathRate']*1.1
# Plot prediction vs actual data - deaths

def plot_log_curve(location):

    _ = train[train.location == location]

    nn = not_know2[not_know2.location == location]

    x = _.days

    y = _['Fatalities']

    x_hat = nn.days

    y_hat = nn.Fatalities

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.plot(x, y, '-', lw=2)

    ax.plot(x_hat, y_hat, 'o', markersize=3)

plot_log_curve('US New York')
plot_log_curve('Finland')
plot_log_curve('Sweden')
plot_log_curve('Germany')
plot_log_curve('Israel')
plot_log_curve('Korea, South')
plot_log_curve('Timor-Leste')
plot_log_curve('Bhutan')
plot_log_curve('Burma')
# Rounding to integer

not_know2['ConfirmedCases'] = not_know2['ConfirmedCases'].round()

not_know2['Fatalities'] = not_know2['Fatalities'].round()
# Submission

sub1 = pd.concat([know[['ForecastId', 'ConfirmedCases','Fatalities']], 

                 not_know2[['ForecastId', 'ConfirmedCases','Fatalities']]])

sub1=sub1.sort_values('ForecastId').reset_index(drop=True)

sub1.to_csv('submission.csv', index=False)
sub1.head(50)
sub1.tail(50)