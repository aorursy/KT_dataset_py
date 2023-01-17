import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

import seaborn as sns

import matplotlib

import matplotlib.dates as mdates

import matplotlib.pyplot as plt

import time

import matplotlib.ticker as ticker

from matplotlib.ticker import MultipleLocator

from datetime import timedelta

from datetime import datetime



warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 15})

pd.options.display.max_rows = 100

# pd.options.display.float_format = '{:.2f}'.format



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Dataset: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset/data



input_data_path = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"

df_raw = pd.read_csv(input_data_path)

print(f'Last observation: {max(df_raw.ObservationDate)}')

print(f'Number of rows: {len(df_raw)}')

print('Columns: ' + ', '.join([f'{k} ({v})' for k, v in df_raw.dtypes.to_dict().items()]))

df_raw.head()
df_timna = pd.read_csv("../input/timna-april-15/corona_lab_tests_ver002_april_15.csv")

df_timna['corona_result'] = df_timna['corona_result'].map({

    'שלילי': 'Negative',

    'חיובי': 'Positive',

    'לא בוצע': 'Not done',

    'בעבודה': 'In progress',

    'לא ודאי': 'Unknown'

})



df_timna = pd.crosstab(df_timna['result_date'], df_timna['corona_result']).reset_index(inplace=False)

df_timna['result_date'] = pd.to_datetime(df_timna['result_date'], dayfirst=True).dt.date

df_timna.sort_values('result_date', ascending=False, inplace=True)

df_timna.set_index('result_date', inplace=True)

df_timna['total_tests'] = df_timna.sum(axis=1)

df_timna['Confirmed'] = df_timna['Positive']

df_timna['TPR'] = 100. * df_timna['Confirmed'] / df_timna['total_tests']

df_timna[1:]


israel_daily_tests = {

'26/01/2020': 3,

'27/01/2020': 4,

'28/01/2020': 4,

'29/01/2020': 7,

'30/01/2020': 11,

'31/01/2020': 11,

'01/02/2020': 11,

'02/02/2020': 17,

'03/02/2020': 23,

'04/02/2020': 29,

'05/02/2020': 29,

'06/02/2020': 29,

'07/02/2020': 29,

'08/02/2020': 29,

'09/02/2020': 68,

'10/02/2020': 97,

'11/02/2020': 172,

'12/02/2020': 210,

'13/02/2020': 247,

'14/02/2020': 275,

'15/02/2020': 277,

'16/02/2020': 305,

'17/02/2020': 348,

'18/02/2020': 393,

'19/02/2020': 429,

'20/02/2020': 457,

'21/02/2020': 472,

'22/02/2020': 485,

'23/02/2020': 560,

'24/02/2020': 618,

'25/02/2020': 722,

'26/02/2020': 845,

'27/02/2020': 979,

'28/02/2020': 1094,

'29/02/2020': 1252,

'01/03/2020': 1412,

'02/03/2020': 1585,

'03/03/2020': 1811,

'04/03/2020': 1934,

'05/03/2020': 2111,

'06/03/2020': 2413,

'07/03/2020': 2800,

'08/03/2020': 3275,

'09/03/2020': 3863,

'10/03/2020': 4381,

'11/03/2020': 4892,

'12/03/2020': 5614,

'13/03/2020': 6399,

'14/03/2020': 7044,

'15/03/2020': 8186,

'16/03/2020': 9464,

'17/03/2020': 11599,

'18/03/2020': 14255,

'19/03/2020': 16758,

'20/03/2020': 19401,

'21/03/2020': 21375,

'22/03/2020': 24734,

'23/03/2020': 28599,

'24/03/2020': 33666,

'25/03/2020': 38906,

'26/03/2020': 44461,

'27/03/2020': 49475,

'28/03/2020': 54436,

'29/03/2020': 60925,

'30/03/2020': 66606,

'31/03/2020': 74457,

'01/04/2020': 82670,

'02/04/2020': 91752,

'03/04/2020': 101655,

'04/04/2020': 108302,

'05/04/2020': 117581,

'06/04/2020': 124831,

'07/04/2020': 131423,

'08/04/2020': 136993,

'09/04/2020': 142514,

'10/04/2020': 148494,

'11/04/2020': 154579,

'12/04/2020': 163072,

'13/04/2020': 172063,

'14/04/2020': 187250,

'15/04/2020': 198002,

'16/04/2020': 207509,

'17/04/2020': 221572,



}





df_israel_tests = pd.DataFrame(list(israel_daily_tests.items()), columns=['ObservationDate', 'Tested'])

df_israel_tests['Country/Region'] = 'Israel'

df_israel_tests['ObservationDate'] = pd.to_datetime(df_israel_tests['ObservationDate'], dayfirst=True).dt.date

df_israel_tests = df_israel_tests.loc[df_israel_tests['ObservationDate'] > pd.to_datetime('2020, 2, 20').date()]
def add_death_rate_col(df):

    df['Death Rate'] = 100 * df['Deaths'] / (df['Confirmed'] + df['Recovered'])

    df[['Death Rate']] = df[['Death Rate']].fillna(0.)

    return df



def add_sick_col(df):

    df['Sick'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

    return df



def add_location_col(df):

    df['Location'] = df['Country/Region'] + ' ' + df['Province/State']

    df['Location'] = df['Location'].str.strip()

    return df



df = df_raw.copy(deep=True)

df = add_sick_col(df)

df = add_death_rate_col(df)

df = add_location_col(df)

df[['Province/State']] = df[['Province/State']].fillna('')

df['ObservationDate'] = pd.to_datetime(df['ObservationDate']).dt.date

df = pd.merge(df, df_israel_tests, on=['ObservationDate', 'Country/Region'], how='outer')



df.head()
df_latest_date = df[df['ObservationDate'] == max(df.loc[df['Country/Region'] != 'Israel', 'ObservationDate'])]

df_latest_date = add_location_col(df_latest_date)

df_latest_date = df_latest_date.loc[:, ['ObservationDate', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Sick', 'Tested', 'Death Rate']]

# print(f'Number of rows: {len(df_latest_date)}')



df_summary = df_latest_date.groupby(['ObservationDate'])[['Confirmed', 'Deaths', 'Recovered', 'Sick']].sum()

df_latest_date_per_country = df_latest_date.groupby(['Country/Region'])[['Confirmed', 'Deaths', 'Recovered', 'Sick', 'Tested']].sum()

df_latest_date_per_country = add_death_rate_col(df_latest_date_per_country).sort_values(by='Confirmed', ascending=False)
df_summary_p = df_summary.copy(deep=True)

for col in df_summary_p:

    df_summary_p[col] = df_summary_p.apply(lambda x: "{:,.0f}".format(x[col]), axis=1)

    

display(df_summary_p.style.background_gradient(cmap='Reds'))
display(df_latest_date_per_country[df_latest_date_per_country.Confirmed > 100].drop('Tested', axis=1).style.background_gradient(cmap='Reds'))
display(df_latest_date_per_country[df_latest_date_per_country.Recovered <= 0][df_latest_date_per_country.Confirmed > 10].sort_values(by='Confirmed', ascending=False).style.background_gradient(cmap='Reds'))
df_daily_summary = df.groupby('ObservationDate')[['Recovered', 'Sick', 'Deaths']].sum().reset_index()

df_daily_summary['ObservationDate'] = pd.to_datetime(df_daily_summary['ObservationDate'])

df_daily_summary.set_index('ObservationDate', inplace=True)

ax = df_daily_summary.plot(kind='bar', stacked=True, figsize=(20,7), grid=True, color=['seagreen', 'cornflowerblue', 'red'], title="COVID-19 sick, recovered and dead people over the world")

ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

plt.xticks(rotation=60, ha='right');

allowed_remainder = max(df['ObservationDate']).day % 2

labels = [item.get_text().replace(' 00:00:00', '') if i % 2 == allowed_remainder else '' for i, item in enumerate(ax.get_xticklabels())]

ax.set_xticklabels(labels);

avg_window=5

pos_rate_col = 'True Positive Rate (TPR)'

smooth_growth_col_name = f'TPR Simple Moving Average ({avg_window}d)'

exp_smooth_growth_col_name = f'TPR Exponential Moving Average ({avg_window}d)'



df_isr_tests_all = df_timna[1:-1]

df_isr_tests_all = df_isr_tests_all.iloc[::-1]

df_isr_tests_all[pos_rate_col] = df_isr_tests_all['TPR']

df_isr_tests_all[smooth_growth_col_name] = df_isr_tests_all[pos_rate_col].rolling(window=avg_window, min_periods=avg_window).mean()

df_isr_tests_all[exp_smooth_growth_col_name] = df_isr_tests_all[pos_rate_col].ewm(span=avg_window, min_periods=avg_window).mean()

# df_isr_tests_all = df_isr_tests_all.reset_index()



df_annotations = df_isr_tests_all.loc[:, ['total_tests']].dropna().to_dict()['total_tests']

df_isr_tests = df_isr_tests_all[['In progress', 'Not done', 'Unknown', 'Positive', 'Negative']]

fig, ax = plt.subplots(figsize=(18,8))

df_isr_tests.plot(ax=ax, kind='bar', stacked=True, grid=True, color=['orange', 'cornflowerblue', 'blue', 'red', 'mediumseagreen' ], title="https://data.gov.il/dataset/covid-19/")

ax2 = ax.twinx()

ax2.spines['right'].set_position(('axes', 1.0))

df_isr_tests_all = df_isr_tests_all.reset_index()

df_isr_tests_all.loc[:, ['TPR']].plot(ax=ax2, alpha=0.6, color='gray', linestyle='--')

# df_isr_tests_all.loc[:, [smooth_growth_col_name]].plot(ax=ax2, alpha=0.99, color='blue', linestyle='--')

df_isr_tests_all.loc[:, [exp_smooth_growth_col_name]].plot(ax=ax2, alpha=0.99, color='navy', linestyle=':')

ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

lines, labels = ax.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax.set_ylabel(f'Number of Tests')

ax2.set_ylabel('True Positive Rate (TPR) [%]')

ax.set_xlabel(f'Test Result Date')

ax.legend(loc='upper left')

plt.suptitle(f'    Daily Corona Tests Results in Israel',fontsize=24, y=0.98)

style = dict(size=11, color='black')

annotations = {}

annotations.update(df_annotations)

x_coord = 0

for k, v in annotations.items():

    annotate_val = f'{v/1000000.:,.2f}M' if v >= 1000000 else (f'{v/1000.:,.2f}K' if v >= 1000 else f'{v:,.0f}')

#     print(f'{annotate_val}: ({k}, {v})')

    ax.text(x_coord, v, annotate_val, ha='center', va='bottom', **style)

    x_coord += 1

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor");

def y_fmt(tick_val, pos):

    if tick_val >= 1000:

        val = int(tick_val) / 1000

        return '{:.0f}k'.format(val)

    else:

        return int(tick_val)



nrow=9

ncol=5

col = 'Sick'

num_countries = nrow * ncol

selected_countries = dict(enumerate(df.groupby(['Country/Region'])[['Confirmed']].sum().reset_index().sort_values('Confirmed', ascending=False)['Country/Region'].values[:num_countries+1], start=0))

df_selected_countries = df[df['Country/Region'].isin(selected_countries.values())]

df_selected_countries = df_selected_countries.groupby(['ObservationDate', 'Country/Region'])[['Recovered', 'Deaths', 'Sick', 'Confirmed']].sum().reset_index().sort_values('Confirmed', ascending=False)



fig, axes = plt.subplots(nrow, ncol,figsize=(20,14))

fig.suptitle(f'{col} Curve in each Country ({max(df["ObservationDate"])})', y=1.025, fontsize=30)

for r in range(nrow):

    for c in range(ncol):

        country = selected_countries[r * ncol + c]

        country_df = df_selected_countries[df_selected_countries['Country/Region'] == country].set_index('ObservationDate')[col].sort_index(ascending=False)

        ax = country_df.plot(ax=axes[r,c], title=f'{country}', color='r')

        ax.set_xticks([]) 

        ax.set_xticklabels([])

        ax.set_xlabel('')

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

        ax.fill_between(country_df.index, country_df.values, facecolor='red', alpha=0.2)

        ax.axis('off')



fig.tight_layout()

country_min_dates_support = 1

df_daily_per_country = df.groupby(['Country/Region', 'ObservationDate'])[['Confirmed', 'Deaths', 'Recovered', 'Sick', 'Tested']].sum().reset_index()

df_daily_per_country = df_daily_per_country[df_daily_per_country['Country/Region'].map(df_daily_per_country['Country/Region'].value_counts()) >= country_min_dates_support]

df_daily_per_country = df_daily_per_country.sort_values(['Country/Region', 'ObservationDate'], ascending=[True, True])
def add_new_cases_col(df, col):

    new_cases_col_name = f'New {col}'

    df[new_cases_col_name] = df[col] - df[col].shift(1)

    return df, new_cases_col_name



def plot_new_per_total(data, col, countries=None, figsize=(17,7), log_scale=True, xticks_interval=2, days_to_sum=5):

    if countries is not None:

        data = data[data['Country/Region'].isin(countries)]

    

    data = data[data['Confirmed'] >= 30]

    

    fig, ax = plt.subplots(figsize = (20,10))

    country_to_df = {}

    cmap = plt.get_cmap('jet_r')

    for i, country in enumerate(countries, start=0):

        color = cmap(float(i)/len(countries))

        country_to_df[country] = data.loc[data['Country/Region'] == country, [col]]

        country_to_df[country], new_cases_col_name = add_new_cases_col(country_to_df[country], col)

        new_cases_sum_col_name = f'{new_cases_col_name} (last {days_to_sum} days)'

        country_to_df[country][new_cases_sum_col_name] = country_to_df[country][new_cases_col_name].rolling(window=days_to_sum).sum()

        country_to_df[country].plot(x=col, y=new_cases_sum_col_name, ax=ax, grid=True, linewidth=0.45, marker='o', markersize=4, label=country)#, c=color)

    

    for line, name in zip(ax.lines, countries):

        y = line.get_ydata()[-1]

        x = line.get_xdata()[-1]

        x_offset = x / 20 if name is not 'US' else x_offset

        ax.annotate(name, xy=(x+x_offset,y), xytext=(x+x_offset,y), color=line.get_color(), size=14, va="top", ha='left')

    

    ax.set_ylabel(f'{new_cases_col_name} (log scale)')

    ax.set_xlabel(f'Accumulated {col} (log scale)')

    ax.legend(loc='upper left')

    ax.set_yscale('log')

    ax.set_xscale('log')

    plt.suptitle(f'COVID-19 New Confirmed Cases against Accumulated Confirmed Cases on Log-Scale',fontsize=24, y=0.95)

    plt.title(f'Johns Hopkins University Data ({max(df["ObservationDate"])})')

    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))



num_countries = 20

selected_countries = df_latest_date_per_country.reset_index().sort_values('Confirmed', ascending=False)['Country/Region'].values[:num_countries+1]

selected_countries = [c for c in selected_countries if c != 'Others']

countries_to_plot = ['Mainland China', 'South Korea', 'Japan', 'US', 'France', 'Germany', 'Italy','Spain', 'Israel', 'Greece', 'UK', 'Iran', 'Russia', 'India']#, 'Brazil', 'Russia']

plot_new_per_total(df_daily_per_country, col='Confirmed', countries=countries_to_plot, log_scale=False, xticks_interval=2)
def plot_col_per_country(data, col, countries=None, figsize=(15,7), log_scale=True, xticks_interval=2, alignment_val=None):

    if countries is not None:

        data = data[data['Country/Region'].isin(countries)]

        

    if alignment_val is not None:

        data = data[data['Confirmed'] >= alignment_val]

        

    pivot_df = data.pivot_table(index='ObservationDate', columns='Country/Region', values=col)

    

    if alignment_val is not None:

        pivot_df = pd.concat([pivot_df[x].dropna().reset_index(drop=True) for x in pivot_df], axis=1)

        pivot_df.index.name = f"Days Since {alignment_val} {col} Cases"

    

    ax = pivot_df.plot(xticks=pivot_df.index, figsize=figsize, grid=True, ls='--' if alignment_val else '-')

    ax.set_ylabel(col)

    

    if alignment_val is None:

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=xticks_interval))

    

    ax.legend(loc='upper left')

    

    if log_scale:

        ax.set_yscale('log')

        

    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    alignment_text = f' since {alignment_val} {col} Cases' if alignment_val else ""

    plt.title(f'Number of {col} Cases ({"Log scale" if log_scale else "Linear scale"}){alignment_text}')

    

    if alignment_val is None:

        plt.xticks(rotation=60, ha='right');



countries_to_plot = ['South Korea', 'Japan', 'US', 'France', 'Germany', 'Italy','Spain', 'Israel', 'UK', 'Greece', 'Russia']

# countries_to_plot = ['US', 'Italy', 'Israel']

plot_col_per_country(df_daily_per_country, col='Confirmed', countries=countries_to_plot, log_scale=False, xticks_interval=2, alignment_val=100)
countries = ['South Korea', 'Japan', 'Thailand', 'Mainland China', 'US', 'France', 'Germany', 'Italy','Spain', 'Israel']
plot_col_per_country(df_daily_per_country, col='Confirmed', countries=countries)
plot_col_per_country(df_daily_per_country, col='Confirmed', countries=['US', 'Germany', 'Spain', 'France', 'Israel'], log_scale=False)
plot_col_per_country(df_daily_per_country, col='Recovered', countries=countries)
plot_col_per_country(df_daily_per_country, col='Deaths', countries=countries)
def plot_daily_new_cases(data, country, col='Confirmed', xticks_interval = None, yticks_interval=None, avg_window=7, max_ylim=None, x_offset=0, y_offset=0, num_days_to_skip=0):

    df_g = data.loc[data['Country/Region'] == country]

    df_g, new_cases_col_name = add_new_cases_col(df_g, col)

    smooth_growth_col_name = f'Simple moving average ({avg_window}d)'

    exp_smooth_growth_col_name = f'Exponential moving average ({avg_window}d)'

    df_g[smooth_growth_col_name] = df_g[new_cases_col_name].rolling(window=avg_window, min_periods=avg_window).mean()

    df_g[exp_smooth_growth_col_name] = df_g[new_cases_col_name].ewm(span=avg_window, min_periods=avg_window).mean()

    df_g = df_g.reset_index().sort_values('ObservationDate', ascending=False)

    df_g = df_g.drop(df_g.tail(num_days_to_skip).index, inplace=False)

    df_g['ObservationDate'] = df_g['ObservationDate'].astype(str)

    df_g = df_g.set_index('ObservationDate')[::-1]

    

    fig, ax = plt.subplots(figsize = (15,7))

    df_g.plot(y=new_cases_col_name, kind = 'bar', color='lightcoral', ax = ax, grid=True)

    df_g.plot(y=smooth_growth_col_name, kind = 'line', linewidth=2.0, marker='o', color='black', ax=ax, grid=True)

    df_g.plot(y=exp_smooth_growth_col_name, kind = 'line', linewidth=2.0, marker='o', color='red', ax=ax, grid=True)

    ax.set_ylabel(new_cases_col_name)

    ax.legend(loc='upper left')

    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    

    if max_ylim is not None and max_ylim < df_g[new_cases_col_name].max():

        ax.set_ylim([0.9, max_ylim])

    

    style = dict(size=14, color='dimgray')

    for p in ax.patches:

        b = p.get_bbox()

        annotate_val = f'{b.y1/1000.:,.1f}K' if b.y1 >= 1000 else f'{b.y1:,.0f}'

        ax.annotate(annotate_val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), ha='center', va='bottom', **style)

    

    plt.title(f'{new_cases_col_name} Cases in {country}')

    plt.xticks(rotation=60, ha='right');

    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 1);

    



# plot_daily_new_cases(df_daily_per_country, 'US', num_days_to_skip=30)

# plot_daily_new_cases(df_daily_per_country, 'Greece', col='Confirmed', num_days_to_skip=0)
plot_daily_new_cases(df_daily_per_country, 'Israel', col='Confirmed', num_days_to_skip=10)
plot_daily_new_cases(df_daily_per_country, 'US', num_days_to_skip=30)
plot_daily_new_cases(df_daily_per_country, 'Italy', num_days_to_skip=18)
plot_daily_new_cases(df_daily_per_country, 'Greece', col='Confirmed', num_days_to_skip=0)
def plot_daily_pos_rate(data, country, xticks_interval = None, yticks_interval=None, avg_window=5, max_ylim=None, x_offset=0, y_offset=0, num_days_to_skip=0):

    pos_rate_col = 'True Positive Rate (TPR)'

    df_g = data.loc[data['Country/Region'] == country]

    df_g, new_tested_col_name = add_new_cases_col(df_g, 'Tested')

    df_g, new_confirmed_col_name = add_new_cases_col(df_g, 'Confirmed')

    df_g[pos_rate_col] = 100. * df_g[new_confirmed_col_name] / df_g[new_tested_col_name].astype(float)

    smooth_growth_col_name = f'TPR Simple Moving Average ({avg_window}d)'

    exp_smooth_growth_col_name = f'TPR Exponential Moving Average ({avg_window}d)'

    df_g[smooth_growth_col_name] = df_g[pos_rate_col].rolling(window=avg_window, min_periods=avg_window).mean()

    df_g[exp_smooth_growth_col_name] = df_g[pos_rate_col].ewm(span=avg_window, min_periods=avg_window).mean()

    df_g = df_g.reset_index().sort_values('ObservationDate', ascending=False)

    df_g = df_g.drop(df_g.tail(num_days_to_skip).index, inplace=False)

    df_g = df_g[df_g['ObservationDate'] <= max([pd.to_datetime(d, dayfirst=True).date() for d in israel_daily_tests.keys()])]

    df_g['ObservationDate'] = df_g['ObservationDate'].astype(str)

    df_g = df_g.set_index('ObservationDate')[::-1]

    fig, ax = plt.subplots(figsize=(20,10))

    title = 'Daily True Positive Rate of Corona Tests in Israel (with simple and exponential averages)'

#     df_g[new_tested_col_name] -= df_g[new_confirmed_col_name]

#     df_g[new_tested_col_name] -= df_g[new_confirmed_col_name]

    ax = df_g.loc[:, [new_confirmed_col_name, new_tested_col_name]].plot(ax=ax, kind='bar', stacked=True, grid=True, color=['Blue', 'lightsteelblue'], title=title)

    ax2 = ax.twinx()

    ax2.spines['right'].set_position(('axes', 1.0))

    df_g.loc[:, [pos_rate_col]].plot(ax=ax2, alpha=0.6, color='gray', linestyle='--')

    df_g.loc[:, [smooth_growth_col_name]].plot(ax=ax2, alpha=0.99, color='blue', linestyle='--')

    df_g.loc[:, [exp_smooth_growth_col_name]].plot(ax=ax2, alpha=0.99, color='navy', linestyle=':')

    

    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    labels = [item.get_text().replace(' 00:00:00', '') for i, item in enumerate(ax.get_xticklabels())]

    ax.set_xticklabels(labels);

    

    lines, labels = ax.get_legend_handles_labels()

    lines2, labels2 = ax2.get_legend_handles_labels()

#     ax.legend()

    ax2.legend(lines + lines2, labels + labels2, loc='upper left')



    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 1)

    ax.set_ylabel('Number of Tests and Confirmed Cases')

    ax2.set_ylabel('True Positive Rate (TPR) [%]')

    

    colors = ['blue','royalblue']

    for i, p in enumerate(ax.patches):

        style = dict(size=14, color=colors[int(i/(len(ax.patches)/2))])

        b = p.get_bbox()

        annotate_val = f'{b.y1/1000.:,.2f}K' if b.y1 >= 1000 else f'{b.y1:,.0f}'

        ax.annotate(annotate_val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), ha='center', va='bottom', **style)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     display(df_g[['New Confirmed', 'New Tested','Confirmed', 'Tested', 'True Positive Rate (TPR)']].tail(n=20))

    

plot_daily_pos_rate(df_daily_per_country, 'Israel', num_days_to_skip=4)

# print("Not available yet, the data is not reliable.")
def add_growth_col(df, col):

    growth_col_name = f'{col} Growth'

    df[growth_col_name] = df[col] / df[col].shift(1)

    return df, growth_col_name



def plot_growth_factor(data, country, xticks_interval = 2, yticks_interval=None, avg_window=7, max_ylim=None, num_days_to_skip=0):

    df_g = data.loc[data['Country/Region'] == country]

    df_g, growth_col_name = add_growth_col(df_g, 'Confirmed')

    smooth_growth_col_name = f'Simple moving average ({avg_window}d)'

    exp_smooth_growth_col_name = f'Exponential moving average ({avg_window}d)'

    df_g[smooth_growth_col_name] = df_g[growth_col_name].rolling(window=avg_window, min_periods=avg_window).mean()

    df_g[exp_smooth_growth_col_name] = df_g[growth_col_name].ewm(span=avg_window, min_periods=avg_window).mean()

    df_g = df_g.reset_index().sort_values('ObservationDate', ascending=False)

    df_g = df_g.drop(df_g.tail(num_days_to_skip).index, inplace=False)

    df_g = df_g.set_index('ObservationDate')

    ax = df_g[[growth_col_name]].plot(linewidth=1.5, figsize=(15,7), grid=True, style = 'r--')

    df_g[[smooth_growth_col_name]].plot(linewidth=1.0, ax=ax, grid=True)

    df_g[[exp_smooth_growth_col_name]].plot(linewidth=1.0, ax=ax, grid=True)

    ax.set_ylabel(growth_col_name)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.2020'))

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=xticks_interval))

    ax.legend(loc='upper left')

    

    if max_ylim is not None and max_ylim < df_g[growth_col_name].max():

        ax.set_ylim([0.9, max_ylim])

    

    if yticks_interval is not None:

        ax.yaxis.set_major_locator(MultipleLocator(yticks_interval))

        

    plt.title(f'{growth_col_name} Factor in {country}')

    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    plt.xticks(rotation=60, ha='right');





# plot_growth_factor(df_daily_per_country, 'US', num_days_to_skip=6)
plot_growth_factor(df_daily_per_country, 'Israel')
plot_growth_factor(df_daily_per_country, 'US', max_ylim=2., xticks_interval=1, num_days_to_skip=50)
plot_growth_factor(df_daily_per_country, 'Italy', max_ylim=2.)
plot_growth_factor(df_daily_per_country, 'Greece', max_ylim=2.)
def plot_daily_new_cases(data, country, xticks_interval = None, yticks_interval=None, avg_window=7, max_ylim=None, x_offset=0, y_offset=0, num_days_to_skip=0):

    df_g = data.loc[data['Country/Region'] == country]

    df_g, new_cases_col_name = add_new_cases_col(df_g, 'Confirmed')

    smooth_growth_col_name = f'Simple moving average ({avg_window}d)'

    exp_smooth_growth_col_name = f'Exponential moving average ({avg_window}d)'

    df_g[smooth_growth_col_name] = df_g[new_cases_col_name].rolling(window=avg_window, min_periods=avg_window).mean()

    df_g[exp_smooth_growth_col_name] = df_g[new_cases_col_name].ewm(span=avg_window, min_periods=avg_window).mean()

    df_g = df_g.reset_index().sort_values('ObservationDate', ascending=False)

    df_g = df_g.drop(df_g.tail(num_days_to_skip).index, inplace=False)



def pred_with_avg_growth_factor(data, country, days_for_avg = 2, days_for_prediction = 7, xticks_interval = 2, factor_penalty=0.0, 

                                yticks_interval=None, print_pred_vals=False, num_days_to_skip=0, lockdown_date=None):

    df_c = data.loc[data['Country/Region'] == country] 

    df_c, growth_col_name = add_growth_col(df_c, 'Confirmed') 

    avg_growth_7d = df_c.sort_values('ObservationDate', ascending=False)[growth_col_name][:days_for_avg].mean() 

    avg_growth_7d -= factor_penalty 

    df_c = df_c.sort_values('ObservationDate', ascending=False) 

    df_c = df_c.drop(df_c.tail(num_days_to_skip).index, inplace=False) 

    df_c = df_c.set_index('ObservationDate') 

    

    max_date = max(df_c.index) 

    first_pred_day = max_date + timedelta(days=1) 

    df_c.loc[max_date, 'Prediction'] = df_c.loc[max_date, 'Confirmed']



    for new_date in pd.date_range(start=first_pred_day, periods=days_for_prediction, freq='D'):

        df_c.loc[new_date, 'Prediction'] = df_c.loc[max_date, 'Prediction'] * avg_growth_7d

        max_date = new_date

        

    df_c = df_c.reset_index().sort_values('ObservationDate', ascending=False)

    df_c = df_c.set_index('ObservationDate')

    ax = df_c[['Confirmed', 'Prediction']].plot(style=['r-','ko--'], linewidth=2.0, figsize=(15,7), grid=True)

    ax.set_ylabel('Confirmed')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.2020'))

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=xticks_interval))

    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    ax.legend(loc='upper left')



    if yticks_interval is not None:

        ax.yaxis.set_major_locator(MultipleLocator(yticks_interval))



    ax.set_ylim(0, int(max(df_c['Prediction']) * 1.1))

    plt.title(f'Predicted number of confirmed cases (factor {avg_growth_7d:.2f}) in {country}')

    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    plt.xticks(rotation=60, ha='right');



    style = dict(size=11, color='dimgray')

    annotations = {}

    annotations.update(df_c.loc[:, ['Prediction']].dropna().to_dict()['Prediction'])

#     annotations.update(df_c.loc[:, ['Confirmed']].dropna().to_dict()['Confirmed'])

#     print(annotations)

    for k, v in annotations.items():

        annotate_val = f'{v/1000000.:,.2f}M' if v >= 1000000 else (f'{v/1000.:,.2f}K' if v >= 1000 else f'{v:,.0f}')

        ax.text(k, v, annotate_val, ha='right', va='bottom', **style)

    

    if lockdown_date is not None:

        arrowprops=dict(arrowstyle='->', color='red', linewidth=3, mutation_scale=50, ls='-')

        lockdown_yval = df_c.loc[lockdown_date, 'Confirmed'].values[0]

        time_since_lockdown = (max(df_c.loc[:, ['Confirmed']].dropna().index) - datetime.strptime(lockdown_date, "%Y-%m-%d")).days

        lockdown_x = datetime.strptime(lockdown_date, "%Y-%m-%d") - timedelta(days=5)

        lockdown_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.4

        arrow_text = f"{country} Lockdown ({lockdown_date})\n{time_since_lockdown} days ago"

        plt.annotate(arrow_text, xy=(lockdown_date, lockdown_yval), xytext=(lockdown_x, lockdown_y), arrowprops=arrowprops, va='top', ha='center')

    

    if print_pred_vals:

        display(df_c.loc[:, ['Prediction']].dropna().astype(int))

        

#     display(df_c)

        

country = 'US'

pred_with_avg_growth_factor(df_daily_per_country, country=country, days_for_prediction=1, num_days_to_skip=45, factor_penalty=0.01, xticks_interval=1, lockdown_date='2020-03-20')
country = 'Israel'

pred_with_avg_growth_factor(df_daily_per_country, country=country, factor_penalty=0.03, xticks_interval=1, lockdown_date='2020-03-16', days_for_prediction=1)
country = 'US'

pred_with_avg_growth_factor(df_daily_per_country, country=country, days_for_prediction=1, num_days_to_skip=45, xticks_interval=1, factor_penalty=0.015, lockdown_date='2020-03-20')
country = 'Italy'

pred_with_avg_growth_factor(df_daily_per_country, country=country, days_for_prediction=1, lockdown_date='2020-03-09')
country = 'Greece'

pred_with_avg_growth_factor(df_daily_per_country, country=country, days_for_prediction=1, lockdown_date=None, num_days_to_skip=10)