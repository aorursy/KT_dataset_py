# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
# reference values 

PER_X = 100000

INFECTION_PERIOD = 10



# display options

pd.options.display.float_format = '{:,.3f}'.format

pd.set_option('display.max_columns', 50)



PROJECT_DIR = '/'



# DATA_DIR_COMMON = '/Users/dcripe/dev/code/notebooks/data_common'

DATA_DIR_COMMON = os.path.join(PROJECT_DIR, '/kaggle/input/')



DATA_DIR = os.path.join(PROJECT_DIR, 'data')

OUT_DIR = os.path.join(PROJECT_DIR, 'out')
df = pd.read_csv('https://query.data.world/s/keax53lpqwffhayvcjmowjiydtevwo', parse_dates=['REPORT_DATE']).copy()
print("date range: {0} to {1}".format(df['REPORT_DATE'].min(), df['REPORT_DATE'].max()))
# limit selection to US

df_us = df[df['COUNTRY_ALPHA_2_CODE'] == 'US'].copy();



# pivot by data and state

df_usp = df_us.groupby(['REPORT_DATE','PROVINCE_STATE_NAME']).sum()[

    [

        'PEOPLE_POSITIVE_CASES_COUNT', 

        'PEOPLE_POSITIVE_NEW_CASES_COUNT', 

        'PEOPLE_DEATH_COUNT', 

        'PEOPLE_DEATH_NEW_COUNT'

    ]

];



# compute simple mortality (CFR) per day based on cummulative totals

df_usp['MORTALITY_RATIO'] = df_usp['PEOPLE_DEATH_COUNT']/df_usp['PEOPLE_POSITIVE_CASES_COUNT'];

df_usp = df_usp.unstack().copy().drop('District of Columbia', level=1, axis=1);



df_statepop = pd.read_csv(os.path.join(DATA_DIR_COMMON,'world-and-us-population-data/nst-est2019-alldata.csv')).iloc[5::]



df_usppop = df_usp.iloc[-1].swaplevel(0,1).unstack().merge(df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')



df_states = pd.read_csv(os.path.join(DATA_DIR_COMMON,'us-states/us_states.csv'))



df_rinf = df_usp.xs('PEOPLE_POSITIVE_NEW_CASES_COUNT', axis=1, level=0).rolling(window=INFECTION_PERIOD).sum().T.merge(

    df_statepop[['NAME','POPESTIMATE2019']], left_index=True, right_on='NAME').set_index('NAME')

df_rinf_popadj = (df_rinf[df_rinf.columns[0:-2]].div(df_rinf[df_rinf.columns[-1]], axis=0)*PER_X)



df_rinf_popadj_ansi = df_rinf_popadj.merge(df_states[['Subdivision name', 'Ansi']], left_index=True, right_on="Subdivision name" )
def clean_county_name(name):   

    toks = name.split()

    if toks[-1] in ['County', 'Area', 'Municipality','Parish']:

        return " ".join(toks[0:-1])

    else:

        return name

    

def make_fips(st, ct):

    return "{0:02d}{1:03d}".format(st, ct)
# calculate county level data



df_countypop = pd.read_csv(os.path.join(DATA_DIR_COMMON,'world-and-us-population-data/co-est2019-alldata.csv'), engine='python')



df_countypop['fips'] = df_countypop.apply(lambda x: make_fips(x['STATE'],x['COUNTY']), axis=1)



# extract NYC population since COVID dataset bundles all NYC counties as one location

nyc_pop = df_countypop[df_countypop['CTYNAME'].isin(['New York County','Kings County','Bronx County','Richmond County','Queens County'])]['POPESTIMATE2019'].sum()



df_countypop['CTYNAME_SHORT'] = df_countypop['CTYNAME'].apply(lambda x: clean_county_name(x))

df_countypop['STATE-COUNTY'] = df_countypop.apply(lambda x: "{0}, {1}".format(x['CTYNAME_SHORT'],x['STNAME']), axis=1)



df_us['STATE-COUNTY'] = df_us.apply(lambda x: "{0}, {1}".format(x['COUNTY_NAME'],x['PROVINCE_STATE_NAME']), axis=1)



df_uscp = df_us.groupby(['REPORT_DATE','STATE-COUNTY']).sum()[

    [

        'PEOPLE_POSITIVE_CASES_COUNT', 

        'PEOPLE_POSITIVE_NEW_CASES_COUNT', 

        'PEOPLE_DEATH_COUNT', 

        'PEOPLE_DEATH_NEW_COUNT'

    ]

].unstack().copy()



df_rinfc = df_uscp.xs('PEOPLE_POSITIVE_NEW_CASES_COUNT', axis=1, level=0).rolling(window=INFECTION_PERIOD).sum().T.merge(

    df_countypop[['STATE-COUNTY','POPESTIMATE2019']], left_index=True, right_on='STATE-COUNTY').set_index('STATE-COUNTY')

df_rinfc_popadj = (df_rinfc[df_rinfc.columns[0:-2]].div(df_rinfc[df_rinfc.columns[-1]], axis=0)*PER_X)

df_rinfc_popadj_fips = df_rinfc_popadj.merge(df_countypop[['STATE-COUNTY', 'fips']], left_index=True, right_on="STATE-COUNTY" )

# shape the data for the animation

df_rinf_popadj_ansi2 = df_rinf_popadj_ansi.set_index('Ansi').drop('Subdivision name', axis=1).stack()

df_rinf_popadj_ansi2.index = df_rinf_popadj_ansi2.index.swaplevel(0,1)

df_rinf_popadj_ansi2 = df_rinf_popadj_ansi2.unstack()

df_rinf_popadj_ansi2.index = pd.to_datetime(df_rinf_popadj_ansi2.index)

df_rinf_popadj_ansi2 = df_rinf_popadj_ansi2.resample('W').mean()

df_rinf_popadj_ansi2 = df_rinf_popadj_ansi2.stack().reset_index()

df_rinf_popadj_ansi2['date'] = df_rinf_popadj_ansi2['level_0'].astype('str')

df_rinf_popadj_ansi2.columns=['ts','state','cases_per_{0}'.format(PER_X),'date']



fig = px.choropleth(

    df_rinf_popadj_ansi2,

    locations="state", 

    locationmode="USA-states", 

    color='cases_per_{0}'.format(PER_X),       

    color_continuous_scale="YlOrRd",

    range_color=(0, 500),    

    scope="usa",

    labels={df_rinf_popadj_ansi.columns[-3]:'Current cases per {0:,} people'.format(PER_X)},

    animation_frame="date",

    animation_group="state",

    hover_name="state",

    title="Cases per {0:,} (state level)".format(PER_X),

)

# fig.suptitle('COVID Cases per {0:,} (time lapsed)'.format(PER_X))

# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df_rinfc_popadj_fips2 = df_rinfc_popadj_fips.set_index('fips').drop('STATE-COUNTY', axis=1).stack()

df_rinfc_popadj_fips2.index = df_rinfc_popadj_fips2.index.swaplevel(0,1)

df_rinfc_popadj_fips2 = pd.pivot_table(df_rinfc_popadj_fips2.reset_index(), index='level_0', columns='fips', aggfunc=np.mean)

df_rinfc_popadj_fips2.index = pd.to_datetime(df_rinfc_popadj_fips2.index)

df_rinfc_popadj_fips2 = df_rinfc_popadj_fips2.resample('M').mean()

df_rinfc_popadj_fips2 = df_rinfc_popadj_fips2.stack().reset_index()

df_rinfc_popadj_fips2['date_str'] = df_rinfc_popadj_fips2['level_0'].astype('str')

df_rinfc_popadj_fips2.columns = ['ts','fips','cases_per_{0}'.format(PER_X),'date']

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)

    

fig = px.choropleth(

    df_rinfc_popadj_fips2, 

    geojson=counties, 

    locations='fips', 

    color='cases_per_{0}'.format(PER_X),       

    color_continuous_scale="YlOrRd",

    range_color=(0, 500),

    scope="usa",

#     labels={

#         '14D_CASES_POP_ADJ':'Current cases per {0:,} people'.format(PER_X), 

# #         "POPESTIMATE2019":"Polulation",

#     },

    hover_name='fips',

    animation_frame="date",

    animation_group="fips",

    title="Cases per {0:,} (county level)".format(PER_X),

)

# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# fig['layout']['geo']['subunitcolor']='rgba(0,0,0,0)'

fig.update_traces(marker_line_width=0.1)

fig.show()


# with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

#     counties = json.load(response)



# fig = px.choropleth(

#     df_rinfc_popadj_fips2[df_rinfc_popadj_fips2['date'] == df_rinfc_popadj_fips2.iloc[-1]['date']], 

#     geojson=counties, 

#     locations='fips', 

#     color='cases_per_{0}'.format(PER_X),       

#     color_continuous_scale="YlOrRd",

#     range_color=(0, 500),

#     scope="usa",

# #     labels={

# #         '14D_CASES_POP_ADJ':'Current cases per {0:,} people'.format(PER_X), 

# # #         "POPESTIMATE2019":"Polulation",

# #     },

#     hover_name='fips',

#     title="Cases per {0:,} (county level)".format(PER_X),

# )

# # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# # fig['layout']['geo']['subunitcolor']='rgba(0,0,0,0)'

# fig.update_traces(marker_line_width=0.1)

# fig.show()