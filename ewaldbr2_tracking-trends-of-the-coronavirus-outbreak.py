import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
# print(df.info)
# print(df.ObservationDate.unique())
dates = df.ObservationDate.unique()
countries = df['Country/Region'].unique()
# print(countries)

import matplotlib.pyplot as plt

pre_dict = {}
pre_dict['country'] = []
pre_dict['date'] = []
pre_dict['Confirmed'] = []
pre_dict['Deaths'] = []
pre_dict['Recovered'] = []
Variable_Lists = ['Confirmed', 'Deaths', 'Recovered']

for date in dates:
    slice1 = df.loc[df.ObservationDate == date, :]
    for country in countries:
        slice2 = slice1.loc[slice1['Country/Region'] == country, :]
        pre_dict['country'].append(country)
        pre_dict['date'].append(date)
        for var in Variable_Lists:
            pre_dict[var].append(slice2[var].sum())
summed_df = pd.DataFrame.from_dict(pre_dict)
# print(summed_df.info)
import matplotlib.pyplot as plt

def plotcountries(df, countries):
    fig = plt.figure(figsize=(15,7))
    for country in countries:
        plt.plot(summed_df.loc[summed_df['country'] == country, 'date'], summed_df.loc[summed_df['country'] == country, 'Confirmed'], label=country)
    plt.yscale('log')
    plt.legend()
    plt.ylabel('Conformed Cases')
    plt.xlabel('Date')
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.grid()
    plt.show()

def plotcountryDailyNew(df, countries, logif=False):
    fig = plt.figure(figsize=(15,7))
    for country in countries:
        cdf = summed_df.loc[summed_df['country'] == country, :]
        plt.bar(cdf.date, cdf.Confirmed.diff(), label=country)
    if logif:
        plt.yscale('log')
        plt.ylabel('Daily New Cases (Log)')
        plt.title('Daily New Cases by Country (Logrithmic Scale)')
    else:
        plt.ylabel('Daily New Cases (Linear)')
        plt.title('Daily New Cases by Country (Linear Scale)')
    plt.legend()
    plt.ylabel('Daily New Cases')
    plt.xlabel('Date')
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.grid()
    ax2 = ax.twinx()
    plt.plot(cdf.date, cdf.Confirmed, '--r', linewidth=3)
    if logif:
        plt.yscale('log')
        plt.ylabel('Cumulative (Log)')
    else:
        plt.ylabel('Cumulative (Linear)')
    plt.show()
    
    
def plotcountrydeathrate(df, countries):
    ax = plt.figure(figsize=(15,7))
    for country in countries:
#         if country == 'Russia':
#             print(df.loc[summed_df['country'] == country, :])
        plt.plot(df.loc[summed_df['country'] == country, 'date'], df.loc[summed_df['country'] == country, 'Deaths']/df.loc[summed_df['country'] == country, 'Confirmed'], label=country)
    plt.legend()
    plt.ylabel('Mortality Rate')
    plt.xlabel('Date')
    plt.ylim([0,.15])
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.grid()
    plt.show()


def DateSumDF(df):
    pre_dict = {}
    pre_dict['date'] = []
    pre_dict['Confirmed'] = []
    pre_dict['Deaths'] = []
    pre_dict['Recovered'] = []
    Variable_Lists = ['Confirmed', 'Deaths', 'Recovered']

    for date in dates:
        slice1 = df.loc[df.ObservationDate == date, :]
        pre_dict['date'].append(date)
        for var in Variable_Lists:
            pre_dict[var].append(slice1[var].sum())
    summed_df = pd.DataFrame.from_dict(pre_dict)
    return summed_df


def plotActiveRecoveredDead(df):
    ax = plt.figure(figsize=(15,7))
    plt.fill_between(df.date, df.Confirmed, label='Recovered')
    plt.fill_between(df.date, df.Confirmed - df.Recovered, label='Active')
    plt.fill_between(df.date, df.Deaths, label='Deaths')
    plt.ylabel('Number of People (Linear Scale)')
    plt.xlabel('Date')
    plt.title('Stacked Case State Plot - Linear')
    #plt.yscale('log')
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.legend(loc=2)
    plt.grid()
    plt.show()
    
    ax = plt.figure(figsize=(15,7))
    plt.fill_between(df.date, df.Confirmed, label='Recovered')
    plt.fill_between(df.date, df.Confirmed - df.Recovered, label='Active')
    plt.fill_between(df.date, df.Deaths, label='Deaths')
    plt.ylabel('Number of People (Log Scale)')
    plt.xlabel('Date')
    plt.title('Stacked Case State Plot - Log')
    plt.yscale('log')
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.legend(loc=2)
    plt.grid()
    plt.show()
    
    ax = plt.figure(figsize=(15,7))
    plt.plot(df.date, df.Recovered, linewidth=3, label='Recovered')
    plt.plot(df.date, df.Confirmed - df.Recovered - df.Deaths, linewidth=3, label='Active')
    plt.plot(df.date, df.Deaths, linewidth=3, label='Deaths')
    plt.ylabel('Number of People (Linear Scale)')
    plt.xlabel('Date')
    plt.title('Case State Plot')
    #plt.yscale('log')
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.legend(loc=2)
    plt.grid()
    plt.show()

date_summed_df = DateSumDF(df)
plotActiveRecoveredDead(date_summed_df)

countries_to_look_at = ['US', 'Japan', 'Iran', 'South Korea', 'UK', 'Italy', 'Brazil', 'Mainland China', 'Germany', 'Sweden']
plotcountrydeathrate(summed_df, countries_to_look_at)
plotcountries(summed_df, countries_to_look_at)
# print(countries) 
plotcountryDailyNew(summed_df, ['US'], logif=True)
plotcountryDailyNew(summed_df, ['US'], logif=False)


def detectslope(df, countries, days):
    fig = plt.figure(figsize=(15,7))
    for country in countries:
        cdf = summed_df.loc[summed_df['country'] == country, :]
        daily_new = np.asarray(cdf.Confirmed.diff())
        active = np.asarray(cdf.Confirmed)
        cumulative = active
        dn = []
        total = []
        for i in range(days,cumulative.shape[0]):
            dn.append(daily_new[i-days:i].sum())
            total.append(cumulative[i])
        plt.plot(total, dn, '-o', label=country)
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().set_xlim(left=100)
    plt.gca().set_ylim(bottom=10)
    plt.xlabel('Confirmed Cases (Log)')
    plt.ylabel('New Cases in Previous {} Days (Log)'.format(str(days)))
    plt.legend()
    plt.grid()
    
    fig = plt.figure(figsize=(15,7))
    for country in countries:
        cdf = summed_df.loc[summed_df['country'] == country, :]
        daily_new = np.asarray(cdf.Confirmed.diff())
        active = np.asarray(cdf.Confirmed - cdf.Deaths - cdf.Recovered)
        cumulative = active
        dn = []
        total = []
        for i in range(days,cumulative.shape[0]):
            dn.append(daily_new[i-days:i].sum())
            total.append(cumulative[i])
        plt.plot(total, dn, '-o', label=country)
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().set_xlim(left=100)
    plt.gca().set_ylim(bottom=10)
    plt.xlabel('Active Cases (Log)')
    plt.ylabel('New Cases in Previous {} Days (Log)'.format(str(days)))
    plt.legend()
    plt.grid()
    
    fig = plt.figure(figsize=(15,7))
    for country in countries:
        cdf = summed_df.loc[summed_df['country'] == country, :]
        daily_new = np.asarray(cdf.Deaths.diff())
        cumulative = np.asarray(cdf.Deaths)
        dn = []
        total = []
        for i in range(days,cumulative.shape[0]):
            dn.append(daily_new[i-days:i].sum())
            total.append(cumulative[i])
        plt.plot(total, dn, '-o', label=country)
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().set_xlim(left=10)
    plt.gca().set_ylim(bottom=1)
    plt.xlabel('Total Deaths (Log)')
    plt.ylabel('New Deaths in Previous {} Days (Log)'.format(str(days)))
    plt.legend()
    plt.grid()


detectslope(summed_df, countries_to_look_at, 7)
def stateplot(df ,state):
    cdf = df.loc[df['Province/State'] == state, :]
    cdf = DateSumDF(cdf)
    offset_start = 0
    fig = plt.figure(figsize=(15, 7) )
    plt.bar(cdf.date[offset_start:], cdf.Confirmed.diff()[offset_start:])
    plt.ylabel('Daily New Cases')
    plt.yscale('log')
    ax = plt.gca()
    ax2 = ax.twinx()
    plt.plot(cdf.date[offset_start:], cdf.Confirmed[offset_start:], '-r', linewidth=3)
    plt.yscale('log')
    plt.ylabel('Confirmed cases (log)')
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::7]))
    for label in temp:
        label.set_visible(False)
    plt.title(state + ' Coronavirus Outbreak')
    plt.xlabel('Date')


counties = ['Michigan', 'New York']
for county in counties:
    stateplot(df, county)

plotcountryDailyNew(summed_df, ['Mainland China'], logif=True)
plotcountryDailyNew(summed_df, ['Italy'], logif=True)
plotcountryDailyNew(summed_df, ['Spain'], logif=True)
# ud_idx = df['Country/Region'] == 'US'
# usdf = df.loc[ud_idx, :] 
# print(usdf['Province/State'].unique())
# counties = ['Seattle, WA', 'San Diego County', 'New York City, NY']