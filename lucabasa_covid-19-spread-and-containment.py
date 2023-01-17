import numpy as np

import pandas as pd 

from scipy.optimize import curve_fit



from datetime import timedelta



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.lines import Line2D

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def min_date(data, num, text, by=None):

    

    if by is None:

        by = 'Country_Region'

    

    min_dates = data[data['n_cases'] > num].groupby(by, as_index=False).date.min()

    

    return min_dates.rename(columns={'date': text})





def increase(data):

    increase = data[['Country_Region', 'date', 'n_cases']].sort_values(['Country_Region', 'date'], ascending=True).copy()

    increase['increase'] = increase.n_cases.diff().fillna(0)

    increase.loc[increase.increase < 0, 'increase'] = 0

    increase['perc_increase'] = (increase.increase / (increase.n_cases - increase.increase) * 100).fillna(0)

    

    return increase[['Country_Region', 'date', 'increase', 'perc_increase']]





def load_series(path):

    ignore = ['Province/State', 'Lat', 'Long']

    data = pd.read_csv(path)

    

    data = data[[col for col in data if col not in ignore]].groupby('Country/Region', as_index=False).sum()

    data.rename(columns={'Country/Region': 'Country_Region'}, inplace=True)

    

    data = data.melt(id_vars='Country_Region')

    data['date'] = pd.to_datetime(data['variable'])

    del data['variable']

    data.rename(columns={'value': 'n_cases'}, inplace=True)

    

    first_case = min_date(data, 0, 'first_date')

    data = pd.merge(data, first_case, on='Country_Region', how='left')

    data['from_first'] = (data['date'] - data['first_date']).dt.days

    

    case_10 = min_date(data, 9, '10th_date')

    data = pd.merge(data, case_10, on='Country_Region', how='left')

    data['from_10th'] = (data['date'] - data['10th_date']).dt.days

    

    case_50 = min_date(data, 49, '50th_date')

    data = pd.merge(data, case_50, on='Country_Region', how='left')

    data['from_50th'] = (data['date'] - data['50th_date']).dt.days

    

    case_100 = min_date(data, 99, '100th_date')

    data = pd.merge(data, case_100, on='Country_Region', how='left')

    data['from_100th'] = (data['date'] - data['100th_date']).dt.days

    

    case_500 = min_date(data, 499, '500th_date')

    data = pd.merge(data, case_500, on='Country_Region', how='left')

    data['from_500th'] = (data['date'] - data['500th_date']).dt.days

    

    data['Country_Region'] = data['Country_Region'].map({'US': 'United States', 

                                                         'Korea, South': 'South Korea'}).fillna(data['Country_Region'])

    

    continents = pd.read_csv('/kaggle/input/country-to-continent/countryContinent.csv', encoding = 'ISO-8859-1')

    continents['country'] = continents['country'].map({'United States of America': 'United States', 

                                                       'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',

                                                       'Korea (Republic of)': 'South Korea', 

                                                       "Korea (Democratic People's Republic of)": 'North Korea'}).fillna(continents['country'])

    

    data = pd.merge(data, continents[['country', 'continent', 'sub_region']], left_on='Country_Region', right_on='country', how='left')

    del data['country']

    

    country_info = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

    data = pd.merge(data, country_info[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Med. Age']], 

                    left_on='Country_Region', right_on='Country (or dependency)', how='left')

    data.rename(columns={'Population (2020)': 'population', 'Density (P/Km²)': 'pop_density', 'Med. Age': 'median_age'}, inplace=True)

    data.loc[data.median_age == 'N.A.', 'median_age'] = np.nan

    data['median_age'] = pd.to_numeric(data.median_age)

    del data['Country (or dependency)']

    

    new_cases = increase(data)

    data = pd.merge(data, new_cases, on=['Country_Region', 'date'], how='left')

    

    return data





def start_from(cases, deaths, col_start, on_cases=False, n_top=10, true_count='x', make_rate=False, ch_date=True):

    if on_cases:

        tmp = pd.merge(cases[['Country_Region', 'date', 'continent', 'population', 'n_cases']+[col_start]], 

                      deaths[['Country_Region', 'date', 'n_cases']], on=['Country_Region', 'date'])

    else:

        tmp = pd.merge(cases[['Country_Region', 'date', 'continent', 'population', 'n_cases']], 

                      deaths[['Country_Region', 'date', 'n_cases']+[col_start]], on=['Country_Region', 'date'])

        

    top_countries = tmp.groupby('Country_Region').n_cases_x.max().sort_values(ascending=True).tail(n_top).index.tolist()

    tmp = tmp[tmp[col_start] > 0]

    tmp = tmp[tmp.Country_Region.isin(top_countries)]

    if ch_date:

        tmp['date'] = tmp[col_start]

    if make_rate:

        tmp['n_cases_y'] = (tmp['n_cases_y'] / tmp['n_cases_x'] * 100).fillna(0)

        true_count = 'y'

    

    tmp['n_cases'] = tmp[f'n_cases_{true_count}']

    

    return tmp





def country_cont(data, country):

    df = data[data.Country == country][['Date Start',

                                    'Description of measure implemented',

                                    'Exceptions', 

                                    'Keywords', 

                                    'Target region']].copy()

    df['region'] = df['Target region'].fillna('All')

    df['Keywords'] = df['Keywords'].fillna('-')

    df['date'] = pd.to_datetime(df['Date Start'])

    df.drop(['Date Start', 'Target region'], axis=1, inplace=True)

    df = df.sort_values(by='date').reset_index(drop=True)

    

    return df



def process_germany(data_loc, pop_loc):

    germany = pd.read_csv(data_loc)

    germ_pop = pd.read_csv(pop_loc)

    

    germany['date'] = pd.to_datetime(germany['date']).dt.date

    germany['gender'] = germany['gender'].map({'M': 'male', 'F': 'female'})

    germany = germany.groupby(['state', 'age_group', 'gender', 'date'], as_index=False).sum()

    

    germany = pd.merge(germany, germ_pop, on=['state', 'age_group', 'gender'], how='left').sort_values('date')

    

    germany.rename(columns={'cases': 'new_cases', 'deaths': 'new_deaths'}, inplace=True)

    

    germany['total_cases'] = germany.groupby(['state', 'age_group', 'gender', 'population'], as_index=False).new_cases.cumsum()

    germany['total_deaths'] = germany.groupby(['state', 'age_group', 'gender', 'population'], as_index=False).new_deaths.cumsum()

    

    return germany.reset_index(drop=True)
conf_cases = load_series('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered = load_series('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

deaths = load_series('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')



# containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID%2019%20Containment%20measures%202020-03-30.csv')

# tmp = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID-19%20containment%20measures/COVID 19 Containment measures data.csv')

# containment = pd.concat([containment, tmp], ignore_index=True)

# containment.loc[containment.Country.fillna('-').str.contains('US:'), 'Country'] = 'United States'

containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')

containment.loc[containment.Country.fillna('-').str.contains('US:'), 'Country'] = 'United States'



germany = process_germany('/kaggle/input/covid19-tracking-germany/covid_de.csv', '/kaggle/input/covid19-tracking-germany/demographics_de.csv')



conf_cases.head()
conf_cases.date.max()
def plot_time_count(data, title, groups_time=None, groups_count=None, legend=True, increase=False):    

    fig = plt.figure(figsize=(18, 8), facecolor='#f7f7f7') 

    fig.suptitle(title, fontsize=18)



    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[5, 2])



    ax0 = fig.add_subplot(spec[0])

    ax1 = fig.add_subplot(spec[1])

    

    if increase:

        use_col = 'increase'

    else:

        use_col = 'n_cases'

    

    if groups_time is None:

        groups_time = ['date']

    if groups_count is None:

        groups_count = 'Country_Region'

    

    tmp = data.groupby(groups_time,as_index=False)[use_col].sum().set_index('date')

    if len(groups_time) > 1:

        tmp.groupby([c for c in groups_time if c != 'date'])[use_col].plot(ax=ax0, legend=legend)

    else:   

        tmp[use_col].plot(ax=ax0, legend=legend)

    

    tmp = data[data.date == data.date.max()].groupby(groups_count)[use_col].sum().sort_values(ascending=True).tail(10)

    tmp.plot(ax=ax1, kind='barh')

    ax1.grid(axis='x')

    

    ax1.set_ylabel('')

    ax0.set_xlabel('')

    

    plt.show()

    

    

def plot_resample(data, column, ax, color, label, index=True, resample='3D'):

    if index:

        res = pd.Series(data[column], index=pd.to_datetime(data.index))

    else:

        res = pd.Series(data[column], index=pd.to_datetime(data['date']))

    

    res.resample(resample).mean().plot(ax=ax, color=color, label=label)

    

    return ax





def make_areas(ax, max_date, offset=7):

    ax.axvline(pd.to_datetime('2020-03-1'), color='y', linestyle='--')

    ax.axvline(pd.to_datetime('2020-03-9'), color='r', linestyle='--')

    ax.axvline(pd.to_datetime('2020-03-20'), color='m', linestyle='--')

    ax.axvspan(pd.to_datetime('2020-03-1') + timedelta(days=offset), max_date, facecolor='y', alpha=0.2)

    ax.axvspan(pd.to_datetime('2020-03-9') + timedelta(days=offset), max_date, facecolor='r', alpha=0.2)

    ax.axvspan(pd.to_datetime('2020-03-20') + timedelta(days=offset), max_date, facecolor='m', alpha=0.2)

    

    return ax





def get_avg_beforemeasure(data, col, ax):

    avg_beforeyellow = data[(data.index > pd.to_datetime('2020-03-1') - timedelta(days=7)) & 

                           (data.index < pd.to_datetime('2020-03-1'))][col].mean()

    avg_beforered = data[(data.index > pd.to_datetime('2020-03-9') - timedelta(days=7)) & 

                           (data.index < pd.to_datetime('2020-03-9'))][col].mean()

    avg_beforepurple = data[(data.index > pd.to_datetime('2020-03-20') - timedelta(days=7)) & 

                           (data.index < pd.to_datetime('2020-03-20'))][col].mean()

    

    ax[2].axhline(avg_beforeyellow, color='y', linestyle='--', alpha=0.5)

    ax[2].axhline(avg_beforered, color='r', linestyle='--', alpha=0.5)

    ax[2].axhline(avg_beforepurple, color='m', linestyle='--', alpha=0.5)

    

    return ax





def expon(x, x0, k, i):

    y = x0 * np.exp(k*x) + i

    return y





def plot_fits(data, col, ax, offset=7):

    df = data.reset_index()

    

    #df[col] = df[col] - df[col].min()

    

    dates = {'2020-03-1': ['y', 'No Yellow Zone'],

             '2020-03-9': ['r', 'No Red Zone'], 

             '2020-03-20': ['m', 'No Purple Zone']}

    

    for day in dates.keys():

        tmp_df = df[df.Date < (pd.to_datetime(day) + timedelta(days=offset))]

        n_days = (pd.to_datetime(day) - pd.to_datetime(df.Date.min())).days + 1

        xdata = np.array(tmp_df.index)

        ydata = np.array(tmp_df[col])

        popt, pcov = curve_fit(expon, xdata, ydata)

        x = np.linspace(-1, df.index.max(), 100)

        y = expon(x, *popt)



        ax.plot(xdata, ydata, 'o', color='k')

        ax.plot(x, y, label=dates[day][1], color=dates[day][0])

        ax.axvline(n_days, color=dates[day][0], linestyle='--')

    

    return ax





def plot_country(data, country, col, ax, color, cases=False):

    tmp = data[data.Country_Region==country]

    if cases:

        ax.scatter(tmp[col], tmp.n_cases/tmp.population*1000, s=tmp.population/1000000, color=color)

    else:

        ax.scatter(tmp[col], tmp.n_cases, s=tmp.population/1000000, color=color)

    return ax
plot_time_count(conf_cases, 'Total Confirmed Cases', ['date'], legend=False)
plot_time_count(conf_cases, 'Total Confirmed Cases by Continent', ['date', 'continent'], ['continent'], legend=True)
plot_time_count(conf_cases, 'New Confirmed Cases by Continent', ['date', 'continent'], ['continent'], legend=True, increase=True)
tmp = conf_cases[conf_cases.continent == 'Europe']

top_countries = tmp.groupby('Country_Region').n_cases.max().sort_values(ascending=True).tail(10).index.tolist()

tmp = tmp[tmp.Country_Region.isin(top_countries)]



plot_time_count(tmp, 'Total Confirmed Cases in Europe', ['date', 'Country_Region'], legend=True)
tmp = conf_cases.copy()

tmp['n_cases'] = tmp['n_cases'] / tmp['population'] * 1000

tmp = tmp[(tmp.continent == 'Europe') & (tmp.population > 1000000)]

top_countries = tmp.groupby('Country_Region').n_cases.max().sort_values(ascending=True).tail(10).index.tolist()

tmp = tmp[tmp.Country_Region.isin(top_countries)]



plot_time_count(tmp, 'Total Confirmed Cases per 1000 inhabitants in Europe', ['date', 'Country_Region'], legend=True)
tmp = conf_cases[conf_cases.continent == 'Africa']

top_countries = tmp.groupby('Country_Region').n_cases.max().sort_values(ascending=True).tail(10).index.tolist()

tmp = tmp[tmp.Country_Region.isin(top_countries)]



plot_time_count(tmp, 'Total Confirmed Cases in Africa', ['date', 'Country_Region'], legend=True)
tmp = conf_cases.copy()

tmp['n_cases'] = tmp['n_cases'] / tmp['population'] * 1000

tmp = tmp[(tmp.continent == 'Americas') & (tmp.population > 1000000)]

top_countries = tmp.groupby('Country_Region').n_cases.max().sort_values(ascending=True).tail(10).index.tolist()

tmp = tmp[tmp.Country_Region.isin(top_countries)]



plot_time_count(tmp, 'Total Confirmed Cases per 1000 inhabitants in the Americas', ['date', 'Country_Region'], legend=True)
tmp = conf_cases.copy()

tmp['n_cases'] = tmp['n_cases'] / tmp['population'] * 1000

tmp = tmp[(tmp.continent == 'Asia') & (tmp.population > 1000000)]

top_countries = tmp.groupby('Country_Region').n_cases.max().sort_values(ascending=True).tail(10).index.tolist()

tmp = tmp[tmp.Country_Region.isin(top_countries)]



plot_time_count(tmp, 'Total Confirmed Cases per 1000 inhabitants in Asia', ['date', 'Country_Region'], legend=True)
tmp = start_from(conf_cases, deaths, 'from_10th', on_cases=False, n_top=12, true_count='x')



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7') 

fig.suptitle('Confirmed cases since the 10th deceased patient', fontsize=18)



tmp = tmp.groupby(['date', 'Country_Region'],as_index=False).n_cases.sum().set_index('date')

tmp.groupby([c for c in ['date', 'Country_Region'] if c != 'date']).n_cases.plot(ax=ax, legend=True)



ax.annotate('USA',

            xy=(tmp[tmp.Country_Region=='United States'].reset_index().date.max(), tmp[tmp.Country_Region=='United States'].n_cases.max()), 

            xycoords='data', xytext=(-55, 0), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Italy',

            xy=(tmp[tmp.Country_Region=='Italy'].reset_index().date.max(), tmp[tmp.Country_Region=='Italy'].n_cases.max()), 

            xycoords='data', xytext=(20, 20), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Spain',

            xy=(tmp[tmp.Country_Region=='Spain'].reset_index().date.max(), tmp[tmp.Country_Region=='Spain'].n_cases.max()), 

            xycoords='data', xytext=(-25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

# ax.annotate('S. Korea',

#             xy=(tmp[tmp.Country_Region=='South Korea'].reset_index().date.max(), tmp[tmp.Country_Region=='South Korea'].n_cases.max()), 

#             xycoords='data', xytext=(-25, 25), textcoords='offset points',

#             arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Germany',

            xy=(tmp[tmp.Country_Region=='Germany'].reset_index().date.max(), tmp[tmp.Country_Region=='Germany'].n_cases.max()), 

            xycoords='data', xytext=(-25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Netherlands',

            xy=(tmp[tmp.Country_Region=='Netherlands'].reset_index().date.max(), tmp[tmp.Country_Region=='Netherlands'].n_cases.max()), 

            xycoords='data', xytext=(25, -25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('UK',

            xy=(tmp[tmp.Country_Region=='United Kingdom'].reset_index().date.max(), tmp[tmp.Country_Region=='United Kingdom'].n_cases.max()), 

            xycoords='data', xytext=(25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('China',

            xy=(40, tmp[tmp.Country_Region=='China'].n_cases.max()- 1000), 

            xycoords='data', xytext=(-25, -25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('France',

            xy=(tmp[tmp.Country_Region=='France'].reset_index().date.max(), tmp[tmp.Country_Region=='France'].n_cases.max()), 

            xycoords='data', xytext=(25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))



ax.set_xlabel('Days after 10th deceased patient', fontsize=14)

ax.set_xlim(1,55)



plt.show()
plot_time_count(recovered, 'Total Recovered Patients', ['date'], legend=False)
tmp = start_from(conf_cases, recovered, 'from_100th', 10, make_rate=True, ch_date=False)



plot_time_count(tmp, 'Recovery rate after 100th confirmed case', ['date', 'Country_Region'], legend=True)
tmp = start_from(conf_cases, recovered, 'from_100th', 10, make_rate=True)



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7') 

fig.suptitle('Recovery rate from 100th patient', fontsize=18)



tmp = tmp.groupby(['date', 'Country_Region'],as_index=False).n_cases.sum().set_index('date')

tmp.groupby([c for c in ['date', 'Country_Region'] if c != 'date']).n_cases.plot(ax=ax, legend=True)



ax.set_xlabel('Days after 100th confirmed case', fontsize=14)



plt.show()
tmp = pd.merge(conf_cases[['Country_Region', 'date', 'continent', 'n_cases', 'first_date', '10th_date', '50th_date', '100th_date']], 

               recovered[['Country_Region', 'date', 'continent', 'n_cases', 'first_date', '10th_date', '50th_date', '100th_date']], 

               on=['Country_Region', 'date', 'continent'])



top_countries = tmp.groupby('Country_Region').n_cases_x.max().sort_values(ascending=True).tail(12).index.tolist()



tmp = tmp[(tmp.Country_Region.isin(top_countries)) & (tmp.Country_Region != 'China')]



tmp['1st recovered/confirmed'] = (tmp['first_date_y'] - tmp['first_date_x']).dt.days

tmp['10th recovered/confirmed'] = (tmp['10th_date_y'] - tmp['10th_date_x']).dt.days

tmp['50th recovered/confirmed'] = (tmp['50th_date_y'] - tmp['50th_date_x']).dt.days

tmp['100th recovered/confirmed'] = (tmp['100th_date_y'] - tmp['100th_date_x']).dt.days



fig, ax = plt.subplots(1,1, figsize=(10, 18), facecolor='#f7f7f7')

fig.suptitle('Days between confirmed case and recovery', fontsize=18)

fig.subplots_adjust(top=0.95)



tmp.groupby('Country_Region')[['1st recovered/confirmed', '10th recovered/confirmed', 

                               '50th recovered/confirmed', '100th recovered/confirmed']].min().plot(kind='barh', ax=ax)



ax.axvline(tmp.groupby('Country_Region')[['1st recovered/confirmed', 

                                          '10th recovered/confirmed', 

                                          '50th recovered/confirmed', 

                                          '100th recovered/confirmed']].min().mean().mean(), linestyle='--', color='k')



ax.set_xlabel('Days from case to recovey', fontsize=14)



plt.show()
plot_time_count(deaths, 'Total Deceased Patients', ['date'], legend=False)
plot_time_count(deaths, 'Total Deceased Patients by Continent', ['date', 'continent'], ['continent'], legend=True)
tmp = start_from(conf_cases, deaths, 'from_500th', n_top=10, on_cases=True, make_rate=True, ch_date=False)



plot_time_count(tmp, 'Death rate after 500th confirmed case', ['date', 'Country_Region'], legend=True)
tmp = start_from(conf_cases, deaths, 'from_500th', n_top=10, on_cases=True, make_rate=True, ch_date=True)



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7') 

fig.suptitle('Death rate from 500th patient', fontsize=18)



tmp = tmp.groupby(['date', 'Country_Region'],as_index=False).n_cases.sum().set_index('date')

tmp.groupby([c for c in ['date', 'Country_Region'] if c != 'date']).n_cases.plot(ax=ax, legend=True)



ax.set_xlabel('Days after 500th confirmed case', fontsize=14)

ax.set_xlim(1,45)



plt.show()
tmp = pd.merge(conf_cases[['Country_Region', 'date', 'continent', 'n_cases', 'from_500th', 'from_first', 'from_10th', 'population']], 

               deaths[['Country_Region', 'date', 'continent', 'n_cases']], 

               on=['Country_Region', 'date', 'continent'])



tmp['n_cases'] = (tmp['n_cases_y'] / tmp['n_cases_x'] * 100).fillna(0)

tmp_2 = tmp.copy()

tmp_2['n_cases_x'] = tmp_2['n_cases_x'] / tmp_2['population'] * 100

top_countries = tmp_2[tmp_2.population > 1000000].groupby('Country_Region').n_cases_x.max().sort_values(ascending=True).tail(10).index.tolist()

tmp = tmp[(tmp.from_500th > 0)]

tmp = tmp[tmp.Country_Region.isin(top_countries)]

tmp['date'] = tmp['from_500th']



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7') 

fig.suptitle('Death rate from 500th patient', fontsize=18)



tmp = tmp.groupby(['date', 'Country_Region'],as_index=False).n_cases.sum().set_index('date')

tmp.groupby([c for c in ['date', 'Country_Region'] if c != 'date']).n_cases.plot(ax=ax, legend=True)



ax.set_xlabel('Days after 500th confirmed case', fontsize=14)



plt.show()
tmp = pd.merge(conf_cases[['Country_Region', 'date', 'continent', 'n_cases', 'first_date', '10th_date', '50th_date', '100th_date']], 

               deaths[['Country_Region', 'date', 'continent', 'n_cases', 'first_date', '10th_date', '50th_date', '100th_date']], 

               on=['Country_Region', 'date', 'continent'])



top_countries = tmp.groupby('Country_Region').n_cases_x.max().sort_values(ascending=True).tail(12).index.tolist()



tmp = tmp[(tmp.Country_Region.isin(top_countries))]



tmp['1st deceased/confirmed'] = (tmp['first_date_y'] - tmp['first_date_x']).dt.days

tmp['10th deceased/confirmed'] = (tmp['10th_date_y'] - tmp['10th_date_x']).dt.days

tmp['50th deceased/confirmed'] = (tmp['50th_date_y'] - tmp['50th_date_x']).dt.days

tmp['100th deceased/confirmed'] = (tmp['100th_date_y'] - tmp['100th_date_x']).dt.days



fig, ax = plt.subplots(1,1, figsize=(10, 18), facecolor='#f7f7f7')

fig.suptitle('Days between confirmed case and death', fontsize=18)

fig.subplots_adjust(top=0.95)



tmp.groupby('Country_Region')[['1st deceased/confirmed', '10th deceased/confirmed', 

                               '50th deceased/confirmed', '100th deceased/confirmed']].min().plot(kind='barh', ax=ax)



ax.axvline(tmp.groupby('Country_Region')[['1st deceased/confirmed', 

                                          '10th deceased/confirmed', 

                                          '50th deceased/confirmed', 

                                          '100th deceased/confirmed']].min().mean().mean(), linestyle='--', color='k')



ax.set_xlabel('Days from case to decease', fontsize=14)



plt.show()
df = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')



ignore = ['Country', 'RegionCode', 'Latitude', 'Longitude', 'SNo']

df = df[[col for col in df if col not in ignore]].copy()



df['Date'] = pd.to_datetime(df['Date']).dt.date



df.head()
tmp = df.groupby('Date').sum()



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7')

fig.suptitle('Total Cases in Italy', fontsize=18)



tmp.CurrentPositiveCases.plot(label='Current Cases', color='k')

tmp.Deaths.plot(label='Deceased patients', color='r')

tmp.Recovered.plot(label='Recovered patients', color='g')



ax.legend()

plt.xticks(rotation=45)



plt.show()
tmp['Positivity rate'] = tmp.TotalPositiveCases / tmp.TestsPerformed * 100

tmp['Death rate'] = tmp.Deaths / tmp.TotalPositiveCases * 100

tmp['Recovery rate'] = tmp.Recovered / tmp.TotalPositiveCases * 100



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7')

fig.suptitle('Positivy, Recovery, Death rates in Italy', fontsize=18)



tmp['Positivity rate'].plot(color='k')

tmp['Death rate'].plot(color='r')

tmp['Recovery rate'].plot(color='g')



ax.legend()

plt.xticks(rotation=45)



plt.show()
tmp['New Tests'] = tmp.TestsPerformed.diff().fillna(0)



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7')

fig.suptitle('New tests and positive cases per day in Italy', fontsize=18)



ax = plot_resample(tmp, 'New Tests', ax, 'k', 'New Tests')

ax2 = ax.twinx()

ax2 = plot_resample(tmp, 'NewPositiveCases', ax2, 'r', 'New Positive Cases')



custom_lines = [Line2D([0], [0], color='k', lw=4),

                Line2D([0], [0], color='r', lw=4)]

ax.legend(custom_lines, ['New Tests', 'New Positive Cases'])



ax.set_xlabel('Date', fontsize=14)

ax.set_ylabel('Number of tests', fontsize=14)

ax2.set_ylabel('Number of Positive Cases', fontsize=14, color='r')



plt.show()
tmp['NewDeaths'] = tmp.Deaths.diff().fillna(0)

tmp['NewRecovered'] = tmp.Recovered.diff().fillna(0)

tmp['DeathIncreaseRatio'] = tmp.NewDeaths / (tmp.Deaths - tmp.NewDeaths) * 100

tmp['RecoveredIncreaseRatio'] = tmp.NewRecovered / (tmp.Recovered - tmp.NewRecovered) * 100

tmp['CasesIncreaseRatio'] = tmp.NewPositiveCases / (tmp.TotalPositiveCases - tmp.NewPositiveCases) * 100

tmp.loc[tmp.CasesIncreaseRatio > 200, 'CasesIncreaseRatio'] = np.nan

tmp.loc[tmp.RecoveredIncreaseRatio > 200, 'RecoveredIncreaseRatio'] = np.nan



fig, ax = plt.subplots(2,1, figsize=(18, 18), facecolor='#f7f7f7')

fig.subplots_adjust(top=0.95)

fig.suptitle('Daily increase in Italy', fontsize=18)



ax[0] = plot_resample(tmp, 'NewPositiveCases', ax[0], 'k', 'New Positive Cases')

ax[0] = plot_resample(tmp, 'NewDeaths', ax[0], 'r', 'New Victims')

ax[0] = plot_resample(tmp, 'NewRecovered', ax[0], 'g', 'New Recovered')

ax[1] = plot_resample(tmp, 'CasesIncreaseRatio', ax[1], 'k', 'New Positive Cases')

ax[1] = plot_resample(tmp, 'DeathIncreaseRatio', ax[1], 'r', 'New Victims')

ax[1] = plot_resample(tmp, 'RecoveredIncreaseRatio', ax[1], 'g', 'New Recovered')



ax[0].legend()

ax[1].legend()

ax[0].set_xlabel('')

ax[1].set_xlabel('')

ax[0].set_title('Total New Cases', fontsize=14)

ax[1].set_title('Increase Ratio', fontsize=14)



plt.show()
tmp['HospitalizedRatio'] = tmp['TotalHospitalizedPatients'] / tmp['CurrentPositiveCases'] * 100

tmp['HomeConfinementRatio'] = tmp['HomeConfinement'] / tmp['CurrentPositiveCases'] * 100

tmp['ICRatio'] = tmp['IntensiveCarePatients'] / tmp['TotalHospitalizedPatients'] * 100



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7')

fig.suptitle('Hospitalization Ratios', fontsize=18)



tmp['HospitalizedRatio'].plot(color='k')

tmp['ICRatio'].plot(color='r')

tmp['HomeConfinementRatio'].plot(color='g')



ax.legend()

plt.xticks(rotation=45)



plt.show()
tmp = df.groupby(['Date', 'RegionName'], as_index=False).sum().set_index('Date')



fig, ax = plt.subplots(3,1, figsize=(18, 24), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.95)

fig.suptitle('Total Cases in Italy by Region', fontsize=18)



tmp.groupby('RegionName').CurrentPositiveCases.plot(legend=True, ax=ax[0], title='Positive Cases')

tmp.groupby('RegionName').Deaths.plot(legend=True, ax=ax[1], title='Deceased Patients')

tmp.groupby('RegionName').Recovered.plot(legend=True, ax=ax[2], title='Recovered Patients')



plt.xticks(rotation=45)



plt.show()
macro_regions = {'Abruzzo': 'South', 

                 'Basilicata': 'South', 

                 'Calabria': 'South', 

                 'Campania': 'South', 

                 'Emilia-Romagna': 'North-East', 

                 'Friuli Venezia Giulia': 'North-East', 

                 'Lazio': 'Center', 

                 'Liguria': 'North-West', 

                 'Lombardia': 'North-West', 

                 'Marche': 'Center', 

                 'Molise': 'South', 

                 'Piemonte': 'North-West', 

                 'P.A. Bolzano': 'North-East', 

                 'P.A. Trento': 'North-East', 

                 'Puglia': 'South', 

                 'Sardegna': 'Islands', 

                 'Sicilia': 'Islands', 

                 'Toscana': 'Center', 

                 'Trentino Alto Adige / Südtirol': 'North-East', 

                 'Umbria': 'Center', 

                 "Valle d'Aosta": 'North-West', 

                 'Veneto': 'North-East'}



df['macro_region'] = df.RegionName.map(macro_regions)

tmp = df.groupby(['Date', 'macro_region'], as_index=False).sum().set_index('Date')



fig, ax = plt.subplots(3,1, figsize=(18, 24), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.95)

fig.suptitle('Total Cases in Italy by Macro-Region', fontsize=18)



tmp.groupby('macro_region').CurrentPositiveCases.plot(legend=True, ax=ax[0], title='Positive Cases')

tmp.groupby('macro_region').Deaths.plot(legend=True, ax=ax[1], title='Deceased Patients')

tmp.groupby('macro_region').Recovered.plot(legend=True, ax=ax[2], title='Recovered Patients')



plt.xticks(rotation=45)



plt.show()
props = tmp[tmp.index == tmp.index.max()][['macro_region', 'CurrentPositiveCases', 'Deaths', 'TestsPerformed']].copy()

props = props.pivot_table(columns='macro_region')#.reset_index()

props['total'] = props.sum(axis=1)

macro_reg = [c for c in props if 'total' not in c]

for col in macro_reg:

    props[col] = props[col] / props['total']

    

props = props.reset_index()

fig, ax = plt.subplots(1,3, figsize=(15, 5), facecolor='#f7f7f7')

#fig.subplots_adjust(top=0.96)

fig.suptitle(f'Proportion of cases in the macro regions on {tmp.index.max()}', fontsize=18)

i=0

for axes in ax:

    axes.pie(props[macro_reg].values[i], labels=macro_reg, autopct='%.0f%%')

    i += 1

    

ax[0].set_title('Current Positive Cases', fontsize=14)

ax[1].set_title('Deceased Patients', fontsize=14)

ax[2].set_title('Test Performed', fontsize=14)

    

plt.show()
tmp = df[df.RegionName.isin(['Lombardia', 'Emilia-Romagna', 'Veneto', 'Piemonte'])].set_index('Date')

tmp['Positivity rate'] = tmp.TotalPositiveCases / tmp.TestsPerformed * 100

tmp['Death rate'] = tmp.Deaths / tmp.TotalPositiveCases * 100

tmp['Recovery rate'] = tmp.Recovered / tmp.TotalPositiveCases * 100



fig, ax = plt.subplots(3,1, figsize=(18, 26), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.95)

fig.suptitle('Positivy, Recovery, Death rates in Lombardia, Piemonte, Veneto, and Emilia Romagna', fontsize=18)



tmp.groupby('RegionName')['Positivity rate'].plot(legend=True, ax=ax[0], title='Positivity Rate')

tmp.groupby('RegionName')['Death rate'].plot(legend=True, ax=ax[1], title='Death Rate')

tmp.groupby('RegionName')['Recovery rate'].plot(legend=True, ax=ax[2], title='Recovery Rate')



plt.xticks(rotation=45)



plt.show()
pops = pd.DataFrame({'RegionName': ['Lombardia', 'Emilia-Romagna', 'Veneto', 'Piemonte'], 

                     'population': [10040, 4453, 4905, 4376]})

tmp = df.copy()

tmp = pd.merge(tmp, pops, on='RegionName')

tmp = tmp.set_index('Date')

tmp['Tests_per1000'] = tmp['TestsPerformed'] / tmp['population']



fig, ax = plt.subplots(2,1, figsize=(18, 16), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.93)

fig.suptitle('Tests performed in Lombardia, Piemonte, Veneto, and Emilia Romagna', fontsize=18)



tmp.groupby('RegionName')['TestsPerformed'].plot(legend=True, ax=ax[0], title='Total Tests')

tmp.groupby('RegionName')['Tests_per1000'].plot(legend=True, ax=ax[1], title='Tests per 1000 inhabitants')



plt.xticks(rotation=45)



plt.show()
test = df[df.RegionName.isin(['Lombardia', 'Emilia-Romagna', 'Veneto', 'Piemonte'])].sort_values(by=['RegionName', 'Date']).set_index('Date')

test['NewTests'] = test.TestsPerformed.diff().fillna(0)

test.loc[test.NewTests < 0, 'NewTests'] = 0



fig, ax = plt.subplots(4,1, figsize=(18, 33), facecolor='#f7f7f7')

fig.subplots_adjust(top=0.95)

fig.suptitle('New tests and positive cases per day in Lombardia, Emilia Romagna, Veneto, and Piemonte', fontsize=18)



i = 0

for region in ['Lombardia', 'Emilia-Romagna', 'Veneto', 'Piemonte']:

    ax[i] = plot_resample(test[test.RegionName==region], 'NewTests', ax[i], 'k', 'New Tests')

    ax2 = ax[i].twinx()

    ax2 = plot_resample(test[test.RegionName==region], 'NewPositiveCases', ax2, 'r', 'New Positive Cases')

    ax[i].set_xlabel('Date', fontsize=14)

    ax[i].set_ylabel('Number of tests', fontsize=14)

    ax2.set_ylabel('Number of Positive Cases', fontsize=14, color='r')

    custom_lines = [Line2D([0], [0], color='k', lw=4),

                Line2D([0], [0], color='r', lw=4)]

    ax[i].legend(custom_lines, ['New Tests', 'New Positive Cases'])

    ax[i].set_title(region, fontsize=14)

    i += 1



plt.show()
tmp = df[df.RegionName.isin(['Lombardia', 'Emilia-Romagna', 'Veneto', 'Piemonte'])].sort_values(by=['RegionName', 'Date']).set_index('Date')

tmp['NewDeaths'] = tmp.Deaths.diff().fillna(0)

tmp.loc[tmp.NewDeaths < 0, 'NewDeaths'] = 0

tmp['NewRecovered'] = tmp.Recovered.diff().fillna(0)

tmp.loc[tmp.NewRecovered < 0, 'NewRecovered'] = 0

tmp['DeathIncreaseRatio'] = tmp.NewDeaths / (tmp.Deaths - tmp.NewDeaths) * 100

tmp['RecoveredIncreaseRatio'] = tmp.NewRecovered / (tmp.Recovered - tmp.NewRecovered) * 100

tmp['CasesIncreaseRatio'] = tmp.NewPositiveCases / (tmp.TotalPositiveCases - tmp.NewPositiveCases) * 100

tmp.loc[tmp.CasesIncreaseRatio > 200, 'CasesIncreaseRatio'] = np.nan

tmp.loc[tmp.RecoveredIncreaseRatio > 200, 'RecoveredIncreaseRatio'] = np.nan



fig, ax = plt.subplots(4,2, figsize=(18, 20), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.93)

fig.suptitle('Daily increase in Lombardia, Emilia Romagna, Veneto, and Piemonte', fontsize=18)

i= 0



for region in ['Lombardia', 'Emilia-Romagna', 'Veneto', 'Piemonte']:

    ax[i][0] = plot_resample(tmp[tmp.RegionName == region], 'NewPositiveCases', ax[i][0], 'k', 'New Positive Cases', resample='5D')

    ax[i][1] = plot_resample(tmp[tmp.RegionName == region], 'CasesIncreaseRatio', ax[i][1], 'k', 'New Positive Cases', resample='5D')

    ax[i][0] = plot_resample(tmp[tmp.RegionName == region], 'NewDeaths', ax[i][0], 'r', 'New Victims', resample='5D')

    ax[i][1] = plot_resample(tmp[tmp.RegionName == region], 'DeathIncreaseRatio', ax[i][1], 'r', 'New Victims', resample='5D')

    ax[i][0] = plot_resample(tmp[tmp.RegionName == region], 'NewRecovered', ax[i][0], 'g', 'New Recovered', resample='5D')

    ax[i][1] = plot_resample(tmp[tmp.RegionName == region], 'RecoveredIncreaseRatio', ax[i][1], 'g', 'New Recovered', resample='5D')



    ax[i][0].legend()

    ax[i][1].legend()

    ax[i][0].set_xlabel('')

    ax[i][1].set_xlabel('')

    ax[i][0].set_title(f'{region} - Total New Cases', fontsize=14)

    ax[i][1].set_title(f'{region} - Increase Ratio', fontsize=14)

    

    i += 1



plt.show()
tmp['HospitalizedRatio'] = tmp['TotalHospitalizedPatients'] / tmp['CurrentPositiveCases'] * 100

tmp['HomeConfinementRatio'] = tmp['HomeConfinement'] / tmp['CurrentPositiveCases'] * 100

tmp['ICRatio'] = tmp['IntensiveCarePatients'] / tmp['TotalHospitalizedPatients'] * 100



fig, ax = plt.subplots(2,2, figsize=(18, 10), facecolor='#f7f7f7', sharex=True)

fig.suptitle('Hospitalization Ratios', fontsize=18)



i = 0

j = 0

for region in ['Lombardia', 'Emilia-Romagna', 'Veneto', 'Piemonte']:

    tmp[tmp.RegionName == region]['HospitalizedRatio'].plot(color='k', ax=ax[i][j])

    tmp[tmp.RegionName == region]['ICRatio'].plot(color='r', ax=ax[i][j])

    tmp[tmp.RegionName == region]['HomeConfinementRatio'].plot(color='g', ax=ax[i][j])

    ax[i][j].legend()

    ax[i][j].set_title(region, fontsize=14)

    if j == 1:

        i += 1

        j = 0

    elif j == 0:

        j += 1

    

plt.xticks(rotation=45)



plt.show()
italy_cont = country_cont(containment, 'Italy')

italy_cont.head()
tmp = df.groupby('Date').sum()

tmp['NewCasesPerc'] = tmp['NewPositiveCases'] / (tmp.TotalPositiveCases - tmp.NewPositiveCases) * 100

tmp.loc[tmp.index < pd.to_datetime('2020-02-25'), 'NewCasesPerc'] = 0



avg_beforeyellow = tmp[(tmp.index < pd.to_datetime('2020-03-1') - timedelta(days=7)) & (tmp.index < pd.to_datetime('2020-03-1'))].NewCasesPerc.mean()

avg_beforeyellow = tmp[(tmp.index < pd.to_datetime('2020-03-1') - timedelta(days=7)) & (tmp.index < pd.to_datetime('2020-03-1'))].NewCasesPerc.mean()



fig, ax = plt.subplots(3,1, figsize=(15, 15), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.93)

fig.suptitle('Total Cases in Italy', fontsize=18)



ax[0] = plot_resample(tmp, 'TotalPositiveCases', ax[0], 'k', '', index=True, resample='3D')

ax[1] = plot_resample(tmp, 'NewPositiveCases', ax[1], 'k', '', index=True, resample='3D')

ax[2] = plot_resample(tmp, 'NewCasesPerc', ax[2], 'k', '', index=True, resample='3D')

ax[0].set_title('Total Cases', fontsize=13)

ax[1].set_title('New Cases', fontsize=13)

ax[2].set_title('New Cases %', fontsize=13)



max_date = tmp.index.max()



for axes in ax:

    axes = make_areas(axes, max_date)

    

ax = get_avg_beforemeasure(tmp, 'NewCasesPerc', ax)



plt.show()
tmp['NewDeaths'] = tmp.Deaths.diff().fillna(0)

tmp['DeathIncreaseRatio'] = tmp.NewDeaths / (tmp.Deaths - tmp.NewDeaths) * 100



fig, ax = plt.subplots(3,1, figsize=(15, 15), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.93)

fig.suptitle('Total Deceased in Italy', fontsize=18)



ax[0] = plot_resample(tmp, 'Deaths', ax[0], 'k', '', index=True, resample='3D')

ax[1] = plot_resample(tmp, 'NewDeaths', ax[1], 'k', '', index=True, resample='3D')

ax[2] = plot_resample(tmp, 'DeathIncreaseRatio', ax[2], 'k', '', index=True, resample='3D')

ax[0].set_title('Total Deceased', fontsize=13)

ax[1].set_title('New Deceased', fontsize=13)

ax[2].set_title('New Deceased %', fontsize=13)



max_date = tmp.index.max()



for axes in ax:

    axes = make_areas(axes, max_date, offset=14)

    

ax = get_avg_beforemeasure(tmp, 'DeathIncreaseRatio', ax)



plt.show()
tmp['NewHospitalized'] = tmp.TotalHospitalizedPatients.diff().fillna(0)

tmp['NewHospPerc'] = tmp.NewHospitalized / (tmp.TotalHospitalizedPatients - tmp.NewHospitalized) * 100



fig, ax = plt.subplots(3,1, figsize=(15, 15), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.93)

fig.suptitle('Total Hospitalized in Italy', fontsize=18)



ax[0] = plot_resample(tmp, 'TotalHospitalizedPatients', ax[0], 'k', '', index=True, resample='3D')

ax[1] = plot_resample(tmp, 'NewHospitalized', ax[1], 'k', '', index=True, resample='3D')

ax[2] = plot_resample(tmp, 'NewHospPerc', ax[2], 'k', '', index=True, resample='3D')

ax[0].set_title('Total Hospitalized', fontsize=13)

ax[1].set_title('New Hospitalized', fontsize=13)

ax[2].set_title('New Hospitalized %', fontsize=13)



max_date = tmp.index.max()



for axes in ax:

    axes = make_areas(axes, max_date, offset=7)

    

ax = get_avg_beforemeasure(tmp, 'NewHospPerc', ax)



plt.show()
tmp['NewIC'] = tmp.IntensiveCarePatients.diff().fillna(0)

tmp['NewICPerc'] = tmp.NewIC / (tmp.IntensiveCarePatients - tmp.NewIC) * 100



fig, ax = plt.subplots(3,1, figsize=(15, 15), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.93)

fig.suptitle('IC Patients in Italy', fontsize=18)



ax[0] = plot_resample(tmp, 'IntensiveCarePatients', ax[0], 'k', '', index=True, resample='3D')

ax[1] = plot_resample(tmp, 'NewIC', ax[1], 'k', '', index=True, resample='3D')

ax[2] = plot_resample(tmp, 'NewICPerc', ax[2], 'k', '', index=True, resample='3D')

ax[0].set_title('Current IC Patients', fontsize=13)

ax[1].set_title('New IC Patients', fontsize=13)

ax[2].set_title('New IC Patients %', fontsize=13)



max_date = tmp.index.max()



for axes in ax:

    axes = make_areas(axes, max_date, offset=7)

    

ax = get_avg_beforemeasure(tmp, 'NewICPerc', ax)



plt.show()
fig, ax = plt.subplots(3,1, figsize=(15, 15), facecolor='#f7f7f7', sharex=True)

fig.subplots_adjust(top=0.92)

fig.suptitle('Total Expected case without containment', fontsize=18)



ax[0] = plot_fits(tmp, 'TotalPositiveCases', ax[0], offset=10)

ax[1] = plot_fits(tmp, 'Deaths', ax[1], offset=14)

ax[2] = plot_fits(tmp, 'TotalHospitalizedPatients', ax[2], offset=10)

ax[0].set_title('Confirmed Cases', fontsize=14)

ax[1].set_title('Deceased Patients', fontsize=14)

ax[2].set_title('Hospitalized Patients', fontsize=14)

ax[0].legend()

ax[1].legend()

ax[2].legend()



plt.show()
fig, ax = plt.subplots(1,2, figsize=(15, 6), facecolor='#f7f7f7')



tmp = conf_cases[conf_cases.pop_density < 1000]

tmp.pop_density.hist(bins=50, ax=ax[0], color='peru', alpha=0.7)

tmp.median_age.hist(bins=10, ax=ax[1], color='seagreen', alpha=0.7)

ax[0].grid(False)

ax[1].grid(False)

ax[0].annotate('Italy',

            xy=(tmp[tmp.Country_Region=='Italy'].pop_density.max(), 200), 

            xycoords='data', xytext=(-25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[0].annotate('Netherlands',

            xy=(tmp[tmp.Country_Region=='Netherlands'].pop_density.max(), 100), 

            xycoords='data', xytext=(-25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[0].annotate('UK',

            xy=(tmp[tmp.Country_Region=='United Kingdom'].pop_density.max(), 100), 

            xycoords='data', xytext=(25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[0].annotate('China',

            xy=(tmp[tmp.Country_Region=='China'].pop_density.max(), 100), 

            xycoords='data', xytext=(-35, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[0].annotate('Spain',

            xy=(tmp[tmp.Country_Region=='Spain'].pop_density.max(), 600), 

            xycoords='data', xytext=(35, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[0].annotate('South Africa',

            xy=(tmp[tmp.Country_Region=='South Africa'].pop_density.max(), 820), 

            xycoords='data', xytext=(35, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[0].annotate('Germany',

            xy=(tmp[tmp.Country_Region=='Germany'].pop_density.max(), 220), 

            xycoords='data', xytext=(35, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))



ax[1].annotate('Italy',

            xy=(tmp[tmp.Country_Region=='Italy'].median_age.max(), 620), 

            xycoords='data', xytext=(-25, -35), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate('Netherlands',

            xy=(tmp[tmp.Country_Region=='Netherlands'].median_age.max(), 1100), 

            xycoords='data', xytext=(-25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate('UK',

            xy=(tmp[tmp.Country_Region=='United Kingdom'].median_age.max(), 900), 

            xycoords='data', xytext=(-25, -25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate('China',

            xy=(tmp[tmp.Country_Region=='China'].median_age.max(), 1000), 

            xycoords='data', xytext=(-25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate('Spain',

            xy=(tmp[tmp.Country_Region=='Spain'].median_age.max(), 630), 

            xycoords='data', xytext=(15, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate('South Africa',

            xy=(tmp[tmp.Country_Region=='South Africa'].median_age.max(), 1200), 

            xycoords='data', xytext=(15, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax[1].annotate('Germany',

            xy=(tmp[tmp.Country_Region=='Germany'].median_age.max(), 200), 

            xycoords='data', xytext=(-15, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))





ax[0].set_title('Population Density', fontsize=16)

ax[1].set_title('Median Age', fontsize=16)

ax[0].set_xlabel('Pop/Km2', fontsize=14)

ax[1].set_xlabel('Age', fontsize=14)



plt.show()
fig, ax = plt.subplots(2,2, figsize=(15, 15), facecolor='#f7f7f7')



cases = conf_cases[conf_cases.pop_density < 1000]

dea = deaths[deaths.pop_density < 1000]

tmp = pd.merge(cases[['Country_Region', 'date', 'n_cases', 'population', 'pop_density', 'median_age']], 

               dea[['Country_Region', 'date', 'n_cases', 'population', 'pop_density', 'median_age']], 

               on=['Country_Region', 'date', 'population', 'pop_density', 'median_age'])

tmp['n_cases'] = tmp['n_cases_y'] / tmp['n_cases_x'] * 100

cases = cases[cases.date == cases.date.max()][['Country_Region', 'n_cases', 'population', 'pop_density', 'median_age']].drop_duplicates()

cases = cases[cases.n_cases > 1000]

# dea = dea[dea.date == dea.date.max()][['Country_Region', 'n_cases', 'population', 'pop_density', 'median_age']].drop_duplicates()

# dea = dea[dea.n_cases > 10]

tmp = tmp[tmp.n_cases_y > 10]

tmp = tmp[tmp.date == tmp.date.max()][['Country_Region', 'n_cases', 'population', 'pop_density', 'median_age']].drop_duplicates()





ax[0][0].scatter(cases.pop_density, cases.n_cases/cases.population*1000, s=cases.population/1000000, color='lightsteelblue', alpha=0.9)

ax[0][1].scatter(tmp.pop_density, tmp.n_cases, s=tmp.population/1000000, color='lightsteelblue', alpha=0.9)

ax[1][0].scatter(cases.median_age, cases.n_cases/cases.population*1000, s=cases.population/1000000, color='lightsteelblue', alpha=0.9)

ax[1][1].scatter(tmp.median_age, tmp.n_cases, s=tmp.population/1000000, color='lightsteelblue', alpha=0.9)



countries = {'Italy': 'r', 'Netherlands': 'darkorange', 'Spain': 'y', 'United Kingdom': 'dodgerblue', 'South Africa': 'g', 'India': 'pink', 'Germany': 'k'}



for country in countries:

    ax[0][0] = plot_country(cases, country, 'pop_density', ax[0][0], countries[country], cases=True)

    ax[0][1] = plot_country(tmp, country, 'pop_density', ax[0][1], countries[country])

    ax[1][0] = plot_country(cases, country, 'median_age', ax[1][0], countries[country], cases=True)

    ax[1][1] = plot_country(tmp, country, 'median_age', ax[1][1], countries[country])



ax[0][0].set_xlabel('Population Density', fontsize=14)

ax[0][1].set_xlabel('Population Density', fontsize=14)

ax[1][0].set_xlabel('Median Age', fontsize=14)

ax[1][1].set_xlabel('Median Age', fontsize=14)

ax[0][0].set_ylabel('Confirmed Cases x 1000', fontsize=14)

ax[0][1].set_ylabel('Mortality Rate (%)', fontsize=14)

ax[1][0].set_ylabel('Confirmed Cases x 1000', fontsize=14)

ax[1][1].set_ylabel('Mortality Rate (%)', fontsize=14)



plt.show()
tmp = start_from(conf_cases, deaths, 'from_10th', on_cases=False, n_top=1000, true_count='x', make_rate=False, ch_date=True)



tmp['n_cases'] = tmp['n_cases'] / tmp['population'] * 1000

tmp = tmp[(tmp.population > 1000000)]



top_countries = ['Italy', 'Netherlands', 'Germany', 'United Kingdom', 'Spain', 'France', 'United States', 'China', 'South Africa']

tmp = tmp[tmp.Country_Region.isin(top_countries)]



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7') 

fig.suptitle('Confirmed cases per 1000 inhabitants since the 10th deceased patient', fontsize=18)



tmp = tmp.groupby(['date', 'Country_Region'],as_index=False).n_cases.sum().set_index('date')

tmp.groupby([c for c in ['date', 'Country_Region'] if c != 'date']).n_cases.plot(ax=ax, legend=True)



ax.annotate('USA',

            xy=(tmp[tmp.Country_Region=='United States'].reset_index().date.max(), tmp[tmp.Country_Region=='United States'].n_cases.max()), 

            xycoords='data', xytext=(55, 0), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Italy',

            xy=(tmp[tmp.Country_Region=='Italy'].reset_index().date.max(), tmp[tmp.Country_Region=='Italy'].n_cases.max()), 

            xycoords='data', xytext=(-35, -35), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Spain',

            xy=(tmp[tmp.Country_Region=='Spain'].reset_index().date.max(), tmp[tmp.Country_Region=='Spain'].n_cases.max()), 

            xycoords='data', xytext=(45, 5), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Germany',

            xy=(tmp[tmp.Country_Region=='Germany'].reset_index().date.max(), tmp[tmp.Country_Region=='Germany'].n_cases.max()), 

            xycoords='data', xytext=(-25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Netherlands',

            xy=(tmp[tmp.Country_Region=='Netherlands'].reset_index().date.max(), tmp[tmp.Country_Region=='Netherlands'].n_cases.max()), 

            xycoords='data', xytext=(15, 15), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('UK',

            xy=(tmp[tmp.Country_Region=='United Kingdom'].reset_index().date.max(), tmp[tmp.Country_Region=='United Kingdom'].n_cases.max()), 

            xycoords='data', xytext=(25, -25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('France',

            xy=(tmp[tmp.Country_Region=='France'].reset_index().date.max(), tmp[tmp.Country_Region=='France'].n_cases.max()), 

            xycoords='data', xytext=(25, 15), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))



ax.set_yscale('log')

ax.set_xlim((1,55))

ax.grid(axis='y')



plt.show()
tmp = start_from(conf_cases, deaths, 'from_500th', on_cases=True, n_top=1000, true_count='y', make_rate=False, ch_date=True)



top_countries = ['Italy', 'Netherlands', 'Germany', 'United Kingdom', 'Spain', 'France', 'United States', 'China', 'South Africa']

tmp = tmp[tmp.Country_Region.isin(top_countries)]



fig, ax = plt.subplots(1,1, figsize=(18, 8), facecolor='#f7f7f7') 

fig.suptitle('Victims since the 500th confirmed cases', fontsize=18)



tmp = tmp.groupby(['date', 'Country_Region'],as_index=False).n_cases.sum().set_index('date')

tmp.groupby([c for c in ['date', 'Country_Region'] if c != 'date']).n_cases.plot(ax=ax, legend=True)



ax.annotate('USA',

            xy=(tmp[tmp.Country_Region=='United States'].reset_index().date.max(), tmp[tmp.Country_Region=='United States'].n_cases.max()), 

            xycoords='data', xytext=(-35, 5), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Italy',

            xy=(tmp[tmp.Country_Region=='Italy'].reset_index().date.max(), tmp[tmp.Country_Region=='Italy'].n_cases.max()), 

            xycoords='data', xytext=(-35, -35), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Spain',

            xy=(tmp[tmp.Country_Region=='Spain'].reset_index().date.max(), tmp[tmp.Country_Region=='Spain'].n_cases.max()), 

            xycoords='data', xytext=(55, 10), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Germany',

            xy=(tmp[tmp.Country_Region=='Germany'].reset_index().date.max(), tmp[tmp.Country_Region=='Germany'].n_cases.max()), 

            xycoords='data', xytext=(25, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('Netherlands',

            xy=(tmp[tmp.Country_Region=='Netherlands'].reset_index().date.max(), tmp[tmp.Country_Region=='Netherlands'].n_cases.max()), 

            xycoords='data', xytext=(45, 15), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('UK',

            xy=(tmp[tmp.Country_Region=='United Kingdom'].reset_index().date.max(), tmp[tmp.Country_Region=='United Kingdom'].n_cases.max()), 

            xycoords='data', xytext=(25, -35), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('France',

            xy=(tmp[tmp.Country_Region=='France'].reset_index().date.max(), tmp[tmp.Country_Region=='France'].n_cases.max()), 

            xycoords='data', xytext=(25, -10), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('South Africa',

            xy=(tmp[tmp.Country_Region=='South Africa'].reset_index().date.max(), tmp[tmp.Country_Region=='South Africa'].n_cases.max()), 

            xycoords='data', xytext=(35, 5), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05))



ax.set_yscale('log')

ax.set_xlim((1,52))

ax.grid(axis='y')



plt.show()
tmp = germany.groupby('age_group', as_index=False)[['new_cases', 'new_deaths']].sum()

germ_pop = pd.read_csv('/kaggle/input/covid19-tracking-germany/demographics_de.csv')

germ_pop = germ_pop.groupby('age_group').population.sum()

tmp = pd.merge(tmp, germ_pop, on='age_group')

tmp['positivy_rate'] = tmp['new_cases'] / tmp['population']

tmp['death_rate'] = tmp['new_deaths'] / tmp['new_cases']

tmp['prop_positives'] = tmp['new_cases'] / tmp['new_cases'].sum()

tmp['prop_deaths'] = tmp['new_deaths'] / tmp['new_deaths'].sum()



fig, ax = plt.subplots(3,2, figsize=(15, 15), facecolor='#f7f7f7')

fig.subplots_adjust(top=0.92)

fig.suptitle('Summary of the situation in Germany', fontsize=18)



tmp.set_index('age_group').new_cases.plot(kind='bar', ax=ax[0][0], color='gold')

tmp.set_index('age_group').new_deaths.plot(kind='bar', ax=ax[0][1], color='red')

tmp.set_index('age_group').positivy_rate.plot(kind='bar', ax=ax[1][0], color='gold')

tmp.set_index('age_group').death_rate.plot(kind='bar', ax=ax[1][1], color='red')



ax[2][0].pie(tmp.prop_positives.values, labels=tmp.age_group, autopct='%.0f%%')

ax[2][1].pie(tmp.prop_deaths.values, labels=tmp.age_group, autopct='%.0f%%')



ax[0][0].set_title('Total Cases', fontsize=14)

ax[0][1].set_title('Total Victims', fontsize=14)

ax[1][0].set_title('Positivity Rate (%)', fontsize=14)

ax[1][1].set_title('Death Rate (%)', fontsize=14)

ax[2][0].set_title('Proportion of Positives', fontsize=14)

ax[2][1].set_title('Proportion of Victims', fontsize=14)



for axes in ax[0]:

    axes.set_xlabel('')

    axes.set_xticklabels(axes.get_xticklabels(), rotation=0)

    axes.grid(axis='y')

for axes in ax[1]:

    axes.set_xlabel('')

    axes.set_xticklabels(axes.get_xticklabels(), rotation=0)

    axes.grid(axis='y')

    axes.set_yticklabels(['{:,.2%}'.format(x) for x in axes.get_yticks()])



plt.show()
to_use = ['Country_Region', 'n_cases', 'increase', 'perc_increase', 'date', 'first_date', 'from_first', 

          '10th_date', 'from_10th', '50th_date', 'from_50th', '100th_date', 'from_100th', '500th_date', 'from_500th']



full_data = pd.merge(conf_cases[to_use].rename(columns={'first_date': 'first_case_date', 

                                                        'from_first': 'from_first_case', 

                                                        '10th_date': '10th_case_date', 

                                                        'from_10th': 'from_10th_case', 

                                                        '50th_date': '10th_case_date', 

                                                        'from_50th': 'from_50th_date',

                                                        '100th_date': '100th_case_date', 

                                                        'from_100th': 'from_100th_case', 

                                                        '500th_date': '500th_case_date',

                                                        'from_500th': 'from_500th_case', 

                                                        'increase': 'new_cases', 

                                                        'perc_increase': 'new_cases_perc'}), 

                     deaths[to_use].rename(columns={'first_date': 'first_victim_date', 

                                                        'from_first': 'from_first_victim', 

                                                        '10th_date': '10th_victim_date', 

                                                        'from_10th': 'from_10th_victim', 

                                                        '50th_date': '50th_victim_date', 

                                                        'from_50th': 'from_50th_victim',

                                                        '100th_date': '100th_victim_date', 

                                                        'from_100th': 'from_100th_victim', 

                                                        '500th_date': '500th_victim_date',

                                                        'from_500th': 'from_500th_victim', 

                                                    'n_cases': 'n_victims', 

                                                        'increase': 'new_victims', 

                                                        'perc_increase': 'new_victims_perc'}), 

                    on=['Country_Region', 'date'])





full_data.head()
cont_all = []



for country in ['Italy', 'Netherlands', 'Germany', 'United Kingdom', 'Spain', 'France', 'United States']:

    tmp = country_cont(containment, country)

    tmp['Country_Region'] = country

    tmp = tmp[tmp.Keywords.str.contains('lockdown|business suspension|school closure|travel ban|social distancing')]

    cont_all.append(tmp[['Country_Region', 'date', 'Keywords', 'region']])

    

cont_all = pd.concat(cont_all, ignore_index=False)

cont_all = cont_all[(cont_all.date >= pd.to_datetime('2020-02-01'))]  # a school ban that I can't find proof of



cont_all.head()
tmp = cont_all[cont_all.Keywords.str.contains('school')].groupby('Country_Region', as_index=False).date.min()



tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])

tmp[['Country_Region', 'date', 'n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
tmp = cont_all[cont_all.Keywords.str.contains('business')].groupby('Country_Region', as_index=False).date.min()



tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])

tmp[['Country_Region', 'date', 'n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
tmp = cont_all[cont_all.Keywords.str.contains('social dista')].groupby('Country_Region', as_index=False).date.min()



tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])

tmp[['Country_Region', 'date', 'n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
tmp = cont_all[cont_all.Keywords.str.contains('travel ban')].groupby('Country_Region', as_index=False).date.min()



tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])

tmp[['Country_Region', 'date', 'n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
cont_all[cont_all.Keywords.str.contains('lockdown')].groupby('Country_Region', as_index=False).date.min()
def get_case(data, col):

    df = data.copy().reset_index()

    date = df[col].max()

    if 'case' in col:

        y_coord = df[df.date == date]['new_cases'].max()

    elif 'victim' in col:

        y_coord = df[df.date == date]['new_victims'].max()

    return date, y_coord





def line_containment(data, measure, ax, color, name=None, legend_lines=None, legend_names=None):

    x_coord = data[data.Keywords.str.contains(measure)].date.min()

    

    try:

        ax.axvline(x_coord, linestyle='--', color=color, alpha=0.5)     

        legend_lines.append(Line2D([0], [0], color=color, lw=2, linestyle='--', alpha=0.5))

        legend_names.append(name)

    except ValueError:

        pass

    

    return ax, legend_lines, legend_names





def country_containment(containment, data, ax):

    tmp = data[data.Country_Region == country].set_index('date')

    tmp_c = containment[containment.Country_Region == country].copy()

    

    tmp.new_cases.plot(color='b', alpha=0.3, ax=ax, label='New Cases')

    ax2 = ax.twinx()

    tmp.new_victims.plot(color='r', alpha=0.3, ax=ax2, label='New Victims')

    

    custom_lines = [Line2D([0], [0], color='b', lw=2, alpha=0.3),

                Line2D([0], [0], color='r', lw=2, alpha=0.3)]

    legend_names = ['New Cases', 'New Victims']

    

    events = ['100th_case_date', '500th_case_date', 'first_victim_date', '10th_victim_date', '100th_victim_date']

    for event in events:

        x_coord, y_coord = get_case(tmp.copy(), event)

        if y_coord is np.nan:

            y_coord = 1

        

        if 'case' in event:

            ax.annotate(event.split('_date')[0].replace('_', ' '),

                xy=(x_coord, y_coord), 

                xycoords='data', xytext=(-55, 22), textcoords='offset points',

                arrowprops=dict(facecolor='black', shrink=0.05))

        else:

            ax2.annotate(event.split('_date')[0].replace('_', ' '),

                xy=(x_coord, y_coord), 

                xycoords='data', xytext=(-55, 42), textcoords='offset points',

                arrowprops=dict(facecolor='black', shrink=0.05))

    

    measures = {'school': ['School cls.', 'k'], 

                'social': ['Soc. Dist.', 'b'], 

                'business': ['Business limit', 'g'],

                'lockdown': ['Lockdown', 'r']}

    for measure in measures:

        ax, custom_lines, legend_names = line_containment(tmp_c, measure, ax, measures[measure][1], measures[measure][0], custom_lines, legend_names)

    

    ax.set_title(country, fontsize=14)

    ax.set_ylabel('New Cases', color='b')

    ax2.set_ylabel('New Victims', color='r')

    

    ax.legend(custom_lines, legend_names)

    

    return ax
top_countries = ['Italy', 'Netherlands', 'Germany', 'France', 'Spain', 'United Kingdom', 'United States']



full_data['date'] = full_data['date'].dt.date



full_data = full_data[full_data.date > pd.to_datetime('2020-02-15')].copy()



fig, ax = plt.subplots(len(top_countries),1, figsize=(18, 3*len(top_countries)), facecolor='#f7f7f7', sharex=True) 

fig.suptitle('Containmente measures and daily increase', fontsize=18)

fig.subplots_adjust(top=0.95)



i = 0

for country in top_countries:

    ax[i] = country_containment(cont_all, full_data, ax[i])

    i += 1



plt.xticks(rotation=45)

plt.show()