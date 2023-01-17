import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os
deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv').groupby(['Country/Region', 'ObservationDate']).agg({'Deaths':'sum'})['Deaths']

deaths
l0 = list(deaths.index.get_level_values(0))

l1 = deaths.index.get_level_values(1)

# cond = (l0 == "foo") | ((l0=="bar") & (l1=="two"))

# df[cond]

l0[0], l0[1] = 'Azerbaijan', 'St. Martin'

idx = pd.MultiIndex.from_tuples(list(zip(l0, pd.to_datetime(l1, format="%m/%d/%Y"))), names=['Country/Region', 'ObservationDate'])

deaths.index = idx

deaths = deaths.groupby(['Country/Region', 'ObservationDate']).sum()

deaths = deaths[deaths.index.get_level_values(1)>='2020-03-01']

deaths
coeff_us_france = 328 / 67

print(f'We need to multiply French figures by {round(coeff_us_france, 1)} to adjust them to US scale')

deaths.loc['France'] = (coeff_us_france * deaths.loc['France'].values).astype(int)
coeff_us_uk = 328 / 67

print(f'We need to multiply UK figures by {round(coeff_us_uk, 1)} to adjust them to US scale')

deaths.loc['UK'] = (coeff_us_uk * deaths.loc['UK'].values).astype(int)
# coeff_us_spain = 328 / 47

# print(f'We need to multiply Spanish figures by {round(coeff_us_spain, 1)} to adjust them to US scale')

# deaths.loc['Spain'] = (coeff_us_spain * deaths.loc['Spain'].values).astype(int)
fig=plt.figure(figsize=(25,5))

fig.show()

ax=fig.add_subplot(111)

ax.plot(deaths.loc['US'], label='US', c='r')

ax.plot(deaths.loc['France'], label='France adjusted for size',c='b', ls='--')

ax.plot(deaths.loc['UK'], label='UK adjusted for size',c='green', ls='--')

# ax.axhline(y=150000, label='150,000 deaths', ls=':', c='grey')

ax.set_title('Total deaths')

ax.legend(loc=2)

# ax.set_xticks(ax.get_xticks()[::10])

plt.draw()
from sklearn.linear_model import LinearRegression

dic = {}

for country in ['US', 'UK', 'France']:

    y = deaths.loc[country]

    y = y.loc[y.index>'07/01/2020']

    l = LinearRegression().fit(np.arange(len(y)).reshape(-1, 1), y)#.reshape(1, -1)

    dic[country] = [l.coef_, l.intercept_]
dic
from datetime import date

d0 = date(2020, 7, 1)

d1 = date(2020, 11, 3)

delta = d1 - d0

print(f"Regression over {delta.days} days")

for country, coeff in zip(['US', 'UK', 'France'], [1, coeff_us_uk, coeff_us_france]):

    print(f"By {d1} there could be {round((dic[country][0][0] * delta.days + dic[country][1])/coeff)} deaths in {country}")
coeff_pop = (100./328)*(10**-6)

France_pct = pd.Series(coeff_pop * deaths.loc['France'].values, index = deaths.loc['France'].index)

# Spain_pct = pd.Series(coeff_pop * deaths.loc['Spain'].values, index = deaths.loc['Spain'].index)

UK_pct = pd.Series(coeff_pop * deaths.loc['UK'].values, index = deaths.loc['UK'].index)

US_pct = pd.Series(coeff_pop * deaths.loc['US'].values, index = deaths.loc['US'].index)

# US_pct
fig=plt.figure(figsize=(25,5))

fig.show()

ax=fig.add_subplot(111)

ax.plot(US_pct, label='US', c='r')

ax.plot(France_pct, label='France adjusted for size',c='b', ls='--')

ax.plot(UK_pct, label='UK adjusted for size',c='green', ls='--')

ax.set_title('Total deaths as pct of population')

ax.legend(loc=2)

# ax.set_xticks(ax.get_xticks()[::10])

plt.draw()
smooth_days = 7
france_smoothed = pd.Series(data = np.convolve((deaths.loc['France']-deaths.loc['France'].shift(1)).dropna(), np.ones(smooth_days)/smooth_days, mode='valid').astype(int), index=deaths.loc['France'].index[smooth_days::])

# france_smoothed
# spain_smoothed = pd.Series(data = np.convolve((deaths.loc['Spain']-deaths.loc['Spain'].shift(1)).dropna(), np.ones(smooth_days)/smooth_days, mode='valid').astype(int), index=deaths.loc['Spain'].index[smooth_days::])

# spain_smoothed
uk_smoothed = pd.Series(data = np.convolve((deaths.loc['UK']-deaths.loc['UK'].shift(1)).dropna(), np.ones(smooth_days)/smooth_days, mode='valid').astype(int), index=deaths.loc['UK'].index[smooth_days::])

# uk_smoothed
us_smoothed = pd.Series(data = np.convolve((deaths.loc['US']-deaths.loc['US'].shift(1)).dropna(), np.ones(smooth_days)/smooth_days, mode='valid').astype(int), index=deaths.loc['US'].index[smooth_days::])

# us_smoothed
fig=plt.figure(figsize=(25,5))

fig.show()

ax=fig.add_subplot(111)

ax.plot(us_smoothed, label='US', c='r')

ax.plot(france_smoothed, label='France adjusted for size',c='b', ls='--')

ax.plot(uk_smoothed, label='UK adjusted for size',c='green', ls='--')

# ax.axhline(y=2000, label='2,000 deaths', ls=':', c='grey')

ax.set_title(f'Daily new deaths, smoothed over last {smooth_days} days')

ax.legend(loc=2)

ax.set_ylim(bottom=0)

# ax.set_xticks(ax.get_xticks()[::10])

plt.draw()