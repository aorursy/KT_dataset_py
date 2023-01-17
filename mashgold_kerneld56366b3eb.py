import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport

import os

from matplotlib import pyplot as plt





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
columns = [c.split('/')[0].lower() for c in df.columns]

df.columns = columns



df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')

#df = df.set_index('date')



print(df.head(), '\n\n', df.tail())
last_date = max(df['date'])

print('last_date: ', last_date)



df_last = (

    df

    .loc[df['date']==last_date]

    .groupby('country')

    .agg({'confirmed': 'sum',

          'deaths': 'sum',

          'recovered': 'sum'})

    .rename(columns={'confirmed': 'sum_confirmed',

                     'deaths': 'sum_deaths',

                     'recovered': 'sum_recovered'})    

    .sort_values(by='sum_confirmed', ascending=False))



df_last['death_rate'] = df_last['sum_deaths'] / df_last['sum_confirmed']

df_last['recover_rate'] = df_last['sum_recovered'] / df_last['sum_confirmed']
# top 20 countries for confirmed infection



(df_last

 .query("sum_confirmed >= 10")

 .sort_values(by='sum_confirmed', ascending=False)

 .head(20))
# top 20 countries for death rate

# limited to countries which number of cumulative confirmed infection over 10



(df_last

 .query("sum_confirmed >= 10")

 .sort_values(by='death_rate', ascending=False)

 .head(20))
# top 20 countries for recover rate

# limited to countries which number of cumulative confirmed infection over 10



(df_last

 .query("sum_confirmed >= 10")

 .sort_values(by='recover_rate', ascending=False)

 .head(30))
df['death_rate'] = df['deaths'] / df['confirmed']

df['recover_rate'] = df['recovered'] / df['confirmed']
# print country list to filter

sorted(df.country.unique())
# Iran is an important one but excluded because of first day's radical increasement

# It's a kind of outlier



country_list = ['China', 'Korea, South', 'Japan', 'US', 

                'France', 'Italy', 'Germany', 'UK', 'Spain', 'Vietnam', 'France', 'Switzerland', 'United Kingdom'

                #, 'Iran'

               ]
df_sdate_country = (

    df

    .query("confirmed > 0")

    .groupby('country')

    .agg({'date': 'min'})

    .rename(columns={'date': 'sdate'}))



df_sdate_country.sort_values(by='sdate')
df_trend = (

    df

    .merge(df_sdate_country, on=['country'])

    .query("date >= sdate")

    .groupby(['country', 'date'])

    .agg({'confirmed': 'sum',

          'deaths': 'sum',

          'recovered': 'sum'})

    .reset_index())



df_trend['death_rate'] = df_trend['deaths'] / df_trend['confirmed']

df_trend['recover_rate'] = df_trend['recovered'] / df_trend['confirmed']



#print(df_trend.query("country == 'Vietnam'").head(20))

#print(df_trend.query("country == 'South Korea'").head(20))
df_trend['daycnt_from_start'] = df_trend.groupby('country')['date'].rank(method='first')



df_cum_confirmed = (

    df_trend

    .groupby('country')

    .agg({'confirmed':'max'})

    .rename(columns={'confirmed': 'confirmed_last'}))



df_trend = df_trend.merge(df_cum_confirmed, on='country')

df_trend['confirmed_pct_by_cum'] = df_trend['confirmed'] / df_trend['confirmed_last']



select_columns = ['country', 'date', 'daycnt_from_start', 'confirmed',

                  'death_rate', 'recover_rate', 'confirmed_pct_by_cum']

df_trend = df_trend.loc[:, select_columns]



df_trend.query("country == 'South Korea'")
# Confirmed Infection Trends

# Since First Infection Day of Each Country



(df_trend

 .loc[df_trend['country'].isin(country_list)]

 .pivot(index='date', columns='country', values='confirmed')

 .plot(figsize=(16, 10))

 .legend(loc=2, prop={'size': 18}))



plt.suptitle('Daily Confirmed Infection Trends', fontsize=30)
# Cumulative Confirmed Infection Trend



(df_trend

 .loc[df_trend['country'].isin(country_list)]

 .pivot(index='date', columns='country', values='confirmed_pct_by_cum')

 .plot(figsize=(16, 10))

 .legend(loc=2, prop={'size': 18}))



plt.suptitle('Cumulative Confirmed Infection Trends', fontsize=30)
# Death Rate Trends

# Since First Infection Day of Each Country



(df_trend

 .loc[df_trend['country'].isin(country_list)]

 .pivot(index='daycnt_from_start', columns='country', values='death_rate')

 .plot(figsize=(16, 10))

 .legend(loc=2, prop={'size': 18}))



plt.suptitle('Daily Death Rates Trends Since First Confirmed Infection Day', fontsize=30)
# print country list to filter

#sorted(df.country.unique())
# Recover Rate Trends

# Since First Infection Day of Each Country



(df_trend

 .loc[df_trend['country'].isin(country_list)]

 .pivot(index='daycnt_from_start', columns='country', values='recover_rate')

 .plot(figsize=(16, 10))

 .legend(loc=2, prop={'size': 18}))



plt.suptitle('Daily Recover Rate Trends Since First Confirmed Infection Day', fontsize=30)
# Cumulative Confirmed Infection Trend

# Since First Infection Day of Each Country



"""

(df_trend

 .loc[df_trend['country'].isin(country_list)]

 .pivot(index='daycnt_from_start', columns='country', values='confirmed_pct_by_cum')

 .plot(figsize=(16, 10))

 .legend(loc=2, prop={'size': 18}))



plt.suptitle('Confirmed Infection Trends Since First Confirmed Infection Day', fontsize=30)

"""
#df_trend.head()

# df_trend.query("date=='2020-03-05' and country=='Mainland China'")