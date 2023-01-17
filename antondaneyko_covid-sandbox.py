import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from os import path
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["figure.figsize"] = (20, 6)
@dataclass(eq=True, frozen=True)
class CountryKey:
    '''For some countries the region is not applicable and is 
    going to be empty'''
    country: str
    sub: str


@dataclass(eq=True, frozen=True)
class CountryRecord:
    start_date: str
    observations: np.array


def parse_johns_hopkins_timeseries(filename: str) -> Dict[CountryKey, CountryRecord]:
    # pandas kata:
    # I could probably rotate the initial data, so I have the index 
    # on country-region-date (it helps to reason about the dataframe
    # as a relational table).
    csv = pd.read_csv(filename).transpose().drop(['Lat', 'Long'])
    csv.loc['Province/State'] = csv.loc['Province/State'].fillna('')
    ret = {}
    for col_name in csv.columns:
        region_data = csv[col_name]
        key = CountryKey(region_data.loc['Country/Region'],
                         region_data.loc['Province/State'],)
        region_data = region_data.drop(['Province/State', 'Country/Region'])
        region_data = region_data[region_data > 0]
        start_date = region_data.index[0] if not region_data.empty else None
        ret[key] = CountryRecord(start_date, region_data.to_numpy())
    return ret


DATA_ROOT = '/kaggle/input'
TIMESERIES_ROOT = path.join(
    DATA_ROOT,
    'covid-19-cssegisanddata',
    'csse_covid_19_data',
    'csse_covid_19_time_series',)
CONFIRMED = path.join(TIMESERIES_ROOT,
                      'time_series_19-covid-Confirmed.csv')
confirmed = parse_johns_hopkins_timeseries(CONFIRMED)
ger = confirmed[CountryKey('Germany', '')]

def plot_raw(x, y, title=''):
    fig, axs = plt.subplots(1, 2)
    if title:
        fig.suptitle(title)
    axs[0].plot(x, y, 'bo', fillstyle='none')
    axs[1].semilogy(x, y, 'bo', fillstyle='none')
    
    plt.show()    

days = np.array(range(len(ger.observations)))
plot_raw(days,
         ger.observations,
         'Number of cases vs days elapsed from the first case')
def infected(days, growth_factor, first_day):
    return first_day * (1 + growth_factor)**days

fit_start = 30
relevant_cases = ger.observations[fit_start:]
fit_days = days[fit_start:]
popt, pcov = curve_fit(infected, fit_days, relevant_cases)

plt.semilogy(fit_days, infected(fit_days, *popt), 'r-')
plt.semilogy(days, ger.observations, 'bo')
plt.show()
popt
