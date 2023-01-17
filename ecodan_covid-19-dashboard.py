%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as matplotlib

from matplotlib import pyplot as plt

import matplotlib.gridspec as gridspec

from datetime import datetime

import os

import gc

import json

from scipy.optimize import curve_fit

import datetime

from pathlib import Path



import plotly.express as px

from urllib.request import urlopen
pd.options.display.float_format = '{:,.3f}'.format

pd.set_option('display.max_columns', 50)
df = pd.read_csv('https://query.data.world/s/keax53lpqwffhayvcjmowjiydtevwo', parse_dates=['REPORT_DATE']).copy()
df.head()
print("date range: {0} to {1}".format(df['REPORT_DATE'].min(), df['REPORT_DATE'].max()))
df_us = df[df['COUNTRY_ALPHA_2_CODE'] == 'US']
df_us['PROVINCE_STATE_NAME'].unique()
df_usp = df_us.groupby(['REPORT_DATE','PROVINCE_STATE_NAME']).sum()[

    [

        'PEOPLE_POSITIVE_CASES_COUNT', 

        'PEOPLE_POSITIVE_NEW_CASES_COUNT', 

        'PEOPLE_DEATH_COUNT', 

        'PEOPLE_DEATH_NEW_COUNT'

    ]

]

df_usp['MORTALITY_RATIO'] = df_usp['PEOPLE_DEATH_COUNT']/df_usp['PEOPLE_POSITIVE_CASES_COUNT']

df_usp = df_usp.unstack().copy().drop('District of Columbia', level=1, axis=1)
top_10 = df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[-1].sort_values(ascending=False)[0:10].index.values

top_25 = df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[-1].sort_values(ascending=False)[0:25].index.values
print("Total deaths to date:\n{0}".format(df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[-1][top_25]))
df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[30::][top_10].plot.line(

    figsize=(12,9),

    title="Top 10 US States with the most commulative COVID-19 fatalities"

);
df_usp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[30::].rolling(window=5).mean().diff().rolling(3).mean().plot(

    subplots=True, 

#     ylim=(-10,25), 

    layout=(10,5), 

    figsize=(18,24),

    grid=True, 

    title='New confirmed COVID-19 cases (US / daily rolling average)',

);
df_usp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[30::].rolling(window=5).mean().diff().rolling(3).mean().plot(

    subplots=True, 

#     ylim=(-10,25), 

    layout=(10,5), 

    figsize=(18,24),

    grid=True, 

    title='New COVID-19 fatalities (US / daily rolling average)',

);
df_usp.xs('MORTALITY_RATIO', axis=1, level=0).iloc[30::].rolling(window=5).mean().plot(

    subplots=True, 

#     ylim=(-10,25), 

    layout=(10,5), 

    figsize=(18,24),

    grid=True, 

    title='Mortality Ratio by state over time',

);
df_usp.xs('MORTALITY_RATIO', axis=1, level=0).iloc[30::].rolling(window=5).mean().iloc[-1].plot.box(

    title="Range of US State Mortalities (median: {0:0.1f}%)".format(np.median(df_usp.xs('MORTALITY_RATIO', axis=1, level=0).iloc[30::].rolling(window=5).mean().iloc[-1].fillna(0).values)*100)

)
df_cp = df.groupby(['REPORT_DATE','COUNTRY_SHORT_NAME']).sum()[

    [

        'PEOPLE_POSITIVE_CASES_COUNT', 

        'PEOPLE_POSITIVE_NEW_CASES_COUNT', 

        'PEOPLE_DEATH_COUNT', 

        'PEOPLE_DEATH_NEW_COUNT'

    ]

]

df_cp['MORTALITY_RATIO'] = df_cp['PEOPLE_DEATH_COUNT']/df_cp['PEOPLE_POSITIVE_CASES_COUNT']

df_cp = df_cp.unstack().copy()
top_10c = df_cp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[-5:-1].max().sort_values(ascending=False)[0:10].index.values

top_25c = df_cp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[-5:-1].max().sort_values(ascending=False)[0:25].index.values
df_cp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0)[top_10c].plot.line(

    figsize=(12,9),

    title="Top 10 Countries with the most commulative COVID-19 fatalities"

);
df_cp.xs('PEOPLE_POSITIVE_CASES_COUNT', axis=1, level=0).iloc[30::][top_25c].rolling(window=5).mean().diff().rolling(3).mean().plot(

    subplots=True, 

#     ylim=(-25,100), 

    grid=True, 

    layout=(5,5), 

    figsize=(18,12), 

#     cmap='tab20',

    title='New confirmed COVID-19 cases (global / daily rolling average)'

);
df_cp.xs('PEOPLE_DEATH_COUNT', axis=1, level=0).iloc[30::][top_25c].rolling(window=5).mean().diff().rolling(3).mean().plot(

    subplots=True, 

#     ylim=(-25,100), 

    grid=True, 

    layout=(5,5), 

    figsize=(18,12), 

    cmap='tab20',

    title='New COVID-19 fatalities (global / daily rolling average)'

);
df_cp.xs('MORTALITY_RATIO', axis=1, level=0).iloc[30::][top_25c].rolling(window=5).mean().plot(

    subplots=True, 

#     ylim=(-10,25), 

    layout=(10,5), 

    figsize=(18,24),

    grid=True, 

    title='Mortality ratio by country over time',

);
df_cp.xs('MORTALITY_RATIO', axis=1, level=0).iloc[30::].rolling(window=5).mean().iloc[-1].plot.box(

    title="Range of Country Mortalities (median: {0:0.1f}%)".format(np.median(df_cp.xs('MORTALITY_RATIO', axis=1, level=0).iloc[30::].rolling(window=5).mean().iloc[-1].fillna(0).values)*100)

)
df_statepop = pd.read_csv('../input/world-and-us-population-data/nst-est2019-alldata.csv').iloc[5::]

df_countrypop = pd.read_csv('../input/world-and-us-population-data/world_pop_2020.csv')

df_usppop = df_usp.iloc[-1].swaplevel(0,1).unstack().merge(df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')
df_cppop = df_cp.iloc[-5:-1].max().swaplevel(0,1).unstack().merge(df_countrypop[['country_code','population', 'country']], left_index=True, right_on='country').set_index('country')

df_cppop_lg = df_cppop[df_cppop['population'] > 10000000]
ax = df_cppop_lg[df_cppop_lg.columns[0:4]].div(df_cppop_lg['population'], axis=0)[['PEOPLE_POSITIVE_CASES_COUNT','PEOPLE_DEATH_COUNT']].sort_values(ascending=False, by='PEOPLE_POSITIVE_CASES_COUNT')[0:50].plot.bar(

    figsize=(20,8), 

    title="% of population infected with or killed by COVID-19 (by country >10M pop)",

#     stacked=True,

#     logy=True

#     icons='child', 

#     icon_size=18, 

#     icon_legend=True,

);

vals = ax.get_yticks();

ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals]);
ax = df_usppop[df_usppop.columns[0:4]].div(df_usppop['POPESTIMATE2019'], axis=0)[['PEOPLE_POSITIVE_CASES_COUNT','PEOPLE_DEATH_COUNT']].sort_values(ascending=False, by='PEOPLE_POSITIVE_CASES_COUNT')[0:50].plot.bar(

    figsize=(20,8), 

    title="% of population infected with or killed by COVID-19 (by US state)",

#     stacked=True,

#     logy=True

#     icons='child', 

#     icon_size=18, 

#     icon_legend=True,

);

vals = ax.get_yticks();

ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals]);
ax = (df_cppop_lg['PEOPLE_DEATH_COUNT']/df_cppop_lg['PEOPLE_POSITIVE_CASES_COUNT']).sort_values(ascending=False)[0:50].plot.bar(

    figsize=(15,8), 

    title="Mortality (fatalities per infections / by country >10M pop)"

);

vals = ax.get_yticks();

ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals]);
ax = (df_usppop['PEOPLE_DEATH_COUNT']/df_usppop['PEOPLE_POSITIVE_CASES_COUNT']).sort_values(ascending=False).plot.bar(

    figsize=(15,8), 

    title="Mortality (fatalities per infections / by US state)"

);

vals = ax.get_yticks();

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals]);
per_x = 100000
df_per = df_usp.iloc[-14::].sum().swaplevel(0,1).unstack().merge(df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')

(df_per['PEOPLE_POSITIVE_NEW_CASES_COUNT']/df_per['POPESTIMATE2019']*10000).sort_values(ascending=False).plot.bar(

    figsize=(15,8), 

    title="Active infections per 10,000 people (based on 14 day infection period)"

);
df_rinf = df_usp.xs('PEOPLE_POSITIVE_NEW_CASES_COUNT', axis=1, level=0).rolling(window=14).sum().iloc[-180::].T.merge(

    df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')

df_rinf_popadj = (df_rinf[df_rinf.columns[0:-2]].div(df_rinf[df_rinf.columns[-1]], axis=0)*per_x)

df_rinf_popadj.T.plot(

    subplots=True, 

#     ylim=(0.01,50),

#     logy=True,

    grid=True, 

    layout=(12,5), 

    figsize=(18,24), 

#     cmap='tab20',

    title='Active infections per 10,000 people (based on 14 infection period)'

);
(df_rinf[df_rinf.columns[0:-2]].div(df_rinf[df_rinf.columns[-1]], axis=0)*10000).T.iloc[-1].plot.box(

    title="Range of State Infection Rate (median: {0:0.1f}%)".format(np.median((df_rinf[df_rinf.columns[0:-2]].div(df_rinf[df_rinf.columns[-1]], axis=0)*10000).T.iloc[-1].values))

)
df_states = pd.read_csv('../input/us-states/us_states.csv')
df_rinf_popadj_ansi = df_rinf_popadj.merge(df_states[['Subdivision name', 'Ansi']], left_index=True, right_on="Subdivision name" )
df_rinf_popadj_ansi.head()
df_rinf_popadj_ansi.columns[-3]
fig = px.choropleth(

    df_rinf_popadj_ansi,

    locations="Ansi", 

    locationmode="USA-states", 

    color=df_rinf_popadj_ansi[df_rinf_popadj_ansi.columns[-3]].values,       

    color_continuous_scale="YlOrRd",

    range_color=(0, 500),    

    scope="usa",

    labels={df_rinf_popadj_ansi.columns[-3]:'Current cases per {0:,} people'.format(per_x)},

)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
# df_trc = df.groupby(['Date','Country_Region','Case_Type']).agg({'Cases':sum,'Population_Count':sum})



df_rinfc = df_cp.xs('PEOPLE_POSITIVE_NEW_CASES_COUNT', axis=1, level=0)[top_25c].rolling(window=14).sum().iloc[-180::].T.merge(

    df_countrypop[['country_code','population', 'country']], left_index=True, right_on='country').set_index('country')



(df_rinfc[df_rinfc.columns[0:-2]].div(df_rinfc[df_rinfc.columns[-1]], axis=0)*10000).T.plot(

    subplots=True, 

#     ylim=(0.01,50),

#     logy=True,

    grid=True, 

    layout=(5,5), 

    figsize=(18,12), 

    cmap='tab20',

    title='Active infections per 10,000 people (based on 14 infection period)'

);
def make_fips(st, ct):

    return "{0:02d}{1:03d}".format(st, ct)

df_countypop = pd.read_csv('../input/us-pop/co-est2019-alldata.csv', engine='python');

df_countypop['fips'] = df_countypop.apply(lambda x: make_fips(x['STATE'],x['COUNTY']), axis=1);

# extract NYC population since COVID dataset bundles all NYC counties as one location

nyc_pop = df_countypop[df_countypop['CTYNAME'].isin(['New York County','Kings County','Bronx County','Richmond County','Queens County'])]['POPESTIMATE2019'].sum();
def clean_county_name(name):   

    toks = name.split()

    if toks[-1] in ['County', 'Area', 'Municipality','Parish']:

        return " ".join(toks[0:-1])

    else:

        return name
df_countypop['CTYNAME_SHORT'] = df_countypop['CTYNAME'].apply(lambda x: clean_county_name(x));

df_countypop['STATE-COUNTY'] = df_countypop.apply(lambda x: "{0}, {1}".format(x['CTYNAME_SHORT'],x['STNAME']), axis=1);

df_us['STATE-COUNTY'] = df_us.apply(lambda x: "{0}, {1}".format(x['COUNTY_NAME'],x['PROVINCE_STATE_NAME']), axis=1)

df_uscp = df_us.groupby(['REPORT_DATE','STATE-COUNTY']).sum()[

    [

        'PEOPLE_POSITIVE_CASES_COUNT', 

        'PEOPLE_POSITIVE_NEW_CASES_COUNT', 

        'PEOPLE_DEATH_COUNT', 

        'PEOPLE_DEATH_NEW_COUNT'

    ]

].unstack().copy();



df_uscppop = df_uscp.iloc[-14::].sum().swaplevel(0,1).unstack().merge(df_countypop[['STATE-COUNTY','REGION','DIVISION','STNAME','CTYNAME_SHORT','fips','POPESTIMATE2019']], left_index=True, right_on='STATE-COUNTY').set_index('STATE-COUNTY');



df_uscppop['14D_CASES_POP_ADJ'] = (df_uscppop['PEOPLE_POSITIVE_NEW_CASES_COUNT']/df_uscppop['POPESTIMATE2019']*per_x);

df_uscppop['14D_DEATHS_POP_ADJ'] = (df_uscppop['PEOPLE_DEATH_NEW_COUNT']/df_uscppop['POPESTIMATE2019']*per_x);



df_uscppop['label'] = df_uscppop.apply(lambda x: "{0} (pop: {1:,})".format(x.CTYNAME_SHORT,x.POPESTIMATE2019), axis=1);



from urllib.request import urlopen

import json

import plotly.express as px



with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)



fig = px.choropleth(

    df_uscppop, 

    geojson=counties, 

    locations='fips', 

    color='14D_CASES_POP_ADJ',       

    color_continuous_scale="YlOrRd",

#     color_continuous_scale="matter",

    range_color=(0, 1000),

#     color_continuous_midpoint=(250),

    scope="usa",

    labels={

        '14D_CASES_POP_ADJ':'Current cases per {0:,} people'.format(per_x), 

#         "POPESTIMATE2019":"Polulation",

    },

    hover_name='label',

    width=1200, 

    height=800,

)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
