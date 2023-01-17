#hide

print('''

 Tracking the spread of the Coronavirus



''')
#hide

import numpy as np

import pandas as pd

from jinja2 import Template

from IPython.display import HTML
#hide



# FETCH

import getpass

base_url = 'https://raw.githubusercontent.com/pratapvardhan/notebooks/master/covid19/'

base_url = '' if (getpass.getuser() == 'Pratap Vardhan') else base_url

paths = {

    'mapping': base_url + 'mapping_countries.csv',

    'overview': base_url + 'overview.tpl'

}



def get_mappings(url):

    df = pd.read_csv(url)

    return {

        'df': df,

        'replace.country': dict(df.dropna(subset=['Name']).set_index('Country')['Name']),

        'map.continent': dict(df.set_index('Name')['Continent'])

    }



mapping = get_mappings(paths['mapping'])



def get_template(path):

    from urllib.parse import urlparse

    if bool(urlparse(path).netloc):

        from urllib.request import urlopen

        return urlopen(path).read().decode('utf8')

    return open(path).read()



def get_frame(name):

    url = (

        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'

        f'csse_covid_19_time_series/time_series_19-covid-{name}.csv')

    df = pd.read_csv(url)

    # rename countries

    df['Country/Region'] = df['Country/Region'].replace(mapping['replace.country'])

    return df



def get_dates(df):

    dt_cols = df.columns[~df.columns.isin(['Province/State', 'Country/Region', 'Lat', 'Long'])]

    LAST_DATE_I = -1

    # sometimes last column may be empty, then go backwards

    for i in range(-1, -len(dt_cols), -1):

        if not df[dt_cols[i]].fillna(0).eq(0).all():

            LAST_DATE_I = i

            break

    return LAST_DATE_I, dt_cols
#hide

COL_REGION = 'Country/Region'

# Confirmed, Recovered, Deaths

df = get_frame('Confirmed')

# dft_: timeseries, dfc_: today country agg

dft_cases = df

dft_deaths = get_frame('Deaths')

dft_recovered = get_frame('Recovered')

LAST_DATE_I, dt_cols = get_dates(df)



dt_today = dt_cols[LAST_DATE_I]

dt_5ago = dt_cols[LAST_DATE_I-5]



dfc_cases = dft_cases.groupby(COL_REGION)[dt_today].sum()

dfc_deaths = dft_deaths.groupby(COL_REGION)[dt_today].sum()

dfc_recovered = dft_recovered.groupby(COL_REGION)[dt_today].sum()

dfp_cases = dft_cases.groupby(COL_REGION)[dt_5ago].sum()

dfp_deaths = dft_deaths.groupby(COL_REGION)[dt_5ago].sum()

dfp_recovered = dft_recovered.groupby(COL_REGION)[dt_5ago].sum()
#hide

df_table = (pd.DataFrame(dict(

    Cases=dfc_cases, Deaths=dfc_deaths, Recovered=dfc_recovered,

    PCases=dfp_cases, PDeaths=dfp_deaths, PRecovered=dfp_recovered))

             .sort_values(by=['Cases', 'Deaths'], ascending=[False, False])

             .reset_index())

for c in 'Cases, Deaths, Recovered'.split(', '):

    df_table[f'{c} (+)'] = (df_table[c] - df_table[f'P{c}']).clip(0)  # DATA BUG

df_table['Fatality Rate'] = (100 * df_table['Deaths'] / df_table['Cases']).round(1)

df_table['Continent'] = df_table['Country/Region'].map(mapping['map.continent'])

df_table.head(15)
#hide

# world, china, europe, us

metrics = ['Cases', 'Deaths', 'Recovered', 'Cases (+)', 'Deaths (+)', 'Recovered (+)']

s_china = df_table[df_table['Country/Region'].eq('China')][metrics].sum().add_prefix('China ')

s_us = df_table[df_table['Country/Region'].eq('US')][metrics].sum().add_prefix('US ')

s_eu = df_table[df_table['Continent'].eq('Europe')][metrics].sum().add_prefix('EU ')

summary = {'updated': pd.to_datetime(dt_today), 'since': pd.to_datetime(dt_5ago)}

summary = {**summary, **df_table[metrics].sum(), **s_china, **s_us, **s_eu}

summary
#hide

dft_ct_cases = dft_cases.groupby(COL_REGION)[dt_cols].sum()

dft_ct_new_cases = dft_ct_cases.diff(axis=1).fillna(0).astype(int)

dft_ct_new_cases.head()
#hide_input

template = Template(get_template(paths['overview']))

html = template.render(

    D=summary, table=df_table,  # REMOVE .head(20) to see all values

    newcases=dft_ct_new_cases.loc[:, dt_cols[LAST_DATE_I-50]:dt_cols[LAST_DATE_I]],

    np=np, pd=pd, enumerate=enumerate)

HTML(f'<div>{html}</div>')