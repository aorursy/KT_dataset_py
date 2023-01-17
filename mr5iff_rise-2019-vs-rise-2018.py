# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install iso3166
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



from IPython.core.display import display, HTML



import iso3166
DIR = '../input'
df_19 = pd.read_csv(f'{DIR}/raw_data_19.csv')

df_18 = pd.read_csv(f'{DIR}/raw_data_18.csv')
df_19['rise_year'] = 2019

df_18['rise_year'] = 2018
df = pd.concat([df_19, df_18], axis=0)
df
df = df.replace(['Korea (Republic of)', 'Korea, Republic of'], 'South Korea')
df.columns
df['tracks_rise19'].unique()
df['tracks_rise18'].unique()
df.groupby('rise_year').size()
STARTUP_TRACKS = ['GROWTH', 'BETA', 'ALPHA','Featured']

PARTNER_TRACKS = ['Exhibitor', 'Media Partner'] 
df_startup = df[df['tracks_rise19'].isin(STARTUP_TRACKS) | df['tracks_rise18'].isin(STARTUP_TRACKS)]

df_partner = df[df['tracks_rise19'].isin(PARTNER_TRACKS) | df['tracks_rise18'].isin(PARTNER_TRACKS)]
df_startup
df_partner
df_startup_country_count = df_startup.pivot_table(columns=['rise_year'], aggfunc='size').to_frame('count')
df_startup_country_count
df_startup_country_count = df_startup.pivot_table(index=['country'], columns=['rise_year'], aggfunc='size').fillna(0)

df_startup_country_count['change'] = df_startup_country_count[2019] - df_startup_country_count[2018]



df_startup_country_count.style.format('{:.0}')

df_startup_country_count['change_pct'] = df_startup_country_count['change'] / df_startup_country_count[2018]

df_startup_country_count['change_pct_of_2019total'] = df_startup_country_count['change'] / df_startup_country_count[2019].sum()



df_startup_country_count.sort_values('change', ascending=False).style.format({'change_pct': '{:.1%}', 'change_pct_of_2019total': '{:.1%}' })
df_startup_country_count.sort_values('change', ascending=True).head(5).style.format({'change_pct': '{:.1%}', 'change_pct_of_2019total': '{:.1%}'})
df_startup_country_count[(df_startup_country_count.loc[:, 2018] + df_startup_country_count.loc[:, 2019]) > 10]
countries = pd.DataFrame(iso3166.countries_by_name).T[[0, 2]]
countries.columns = ['country', 'country_code']
df_country = df_startup_country_count.merge(countries, on='country', how='left')

df_country[df_country['country_code'].isnull()]



df_country.loc[df_country['country'] == 'United Kingdom', 'country_code'] = 'GBR'

df_country.loc[df_country['country'] == 'South Korea', 'country_code'] = 'KOR'

df_country.loc[df_country['country'] == 'Taiwan', 'country_code'] = 'TWN'
data = [go.Choropleth(

    locations = df_country['country_code'],

    z = df_country['change'],

    text = df_country['country'],

    colorscale = 'Blackbody',

    autocolorscale = False,

#     autocolorscale = True,

#     reversescale = True,

    marker = go.choropleth.Marker(

        line = go.choropleth.marker.Line(

            color = 'rgb(180,180,180)',

            width = 0.5

        )),

    colorbar = go.choropleth.ColorBar(

        title = 'Company #'),

)]



layout = go.Layout(

#     title = go.layout.Title(

#         text = 'RISE 2019 vs 2018 change by country'

#     ),

    geo = go.layout.Geo(

        showframe = False,

        showcoastlines = True,

        projection = go.layout.geo.Projection(

            type = 'equirectangular'

        )

    ),

)



fig = go.Figure(data = data, layout = layout)

iplot(fig)
df_startup['funding_tier'].unique()
df_startup['funding_tier'] = pd.Categorical(df_startup['funding_tier'], ["Not specified", "USD 0 - USD 250k", "USD 250k - USD 500k", 'USD 500k - USD 1m', 'USD 1m+'])
df_startup['funding_tier'] = df_startup['funding_tier'].fillna("Not specified")
df_funding = df_startup.pivot_table(index='funding_tier', columns='rise_year', aggfunc='size')

df_funding
data = [

    go.Bar(

        x = df_funding.index,

        y = df_funding[2018],

#         base = [-500,-600,-700],

        marker = dict(

          color = 'grey'

        ),

        name = '2018'

    ),

    go.Bar(

        x = df_funding.index,

        y = df_funding[2019],

#         base = 0,

        marker = dict(

          color = 'green'

        ),

        name = '2019'

    )

]





fig = go.Figure(data=data)

iplot(fig)
df_startup[df_startup['tracks_rise17'].notnull() & df_startup['tracks_rise18'].notnull() & df_startup['tracks_rise19'].notnull()]
def retention_by(df, by='country'):

    df_retent2019 = df[df['tracks_rise18'].notnull() & df['tracks_rise19'].notnull()].groupby(by).size().to_frame(2019)

    df_retent2018 = df[df['tracks_rise17'].notnull() & df['tracks_rise18'].notnull()].groupby(by).size().to_frame(2018)

    df = df_retent2018.join(df_retent2019, how='outer').fillna(0)

    return df
def count_by(df, by='country'):

    df_count = df.pivot_table(index=[by], columns=['rise_year'], aggfunc='size').fillna(0)

    df_count['change'] = df_count[2019] - df_count[2018]

    df_count['change_pct'] = df_count['change'] / df_count[2018]

    df_count['change_pct_of_2019total'] = df_count['change'] / df_count[2019].sum()

    return df_count
def retent_stat(df, by='country'):

    df_retent = retention_by(df_startup, by=by).join(count_by(df_startup, by=by), how='left', lsuffix='_retent')

    df_retent['2018_retent%'] = df_retent['2018_retent'] / df_retent['2018']

    df_retent['2019_retent%'] = df_retent['2019_retent'] / df_retent['2019']

    return df_retent[[col for col in df_retent.columns if 'retent' in col]].sort_values('2019_retent', ascending=False).style.format({

        '2018_retent': '{:.0f}',

        '2019_retent': '{:.0f}',

        '2018_retent%': '{:.1%}',

        '2019_retent%': '{:.1%}',

    })
retent_stat(df_startup, by='country')
df_startup.loc[df_startup['industries_rise18'].notnull(), 'industries'] = df_startup.loc[df_startup['industries_rise18'].notnull(), 'industries_rise18']

df_startup.loc[df_startup['industries_rise19'].notnull(), 'industries'] = df_startup.loc[df_startup['industries_rise19'].notnull(), 'industries_rise19']



retent_stat(df_startup, by='industries')
count_by(df_startup, by='industries').sort_values('change', ascending=False).style.format({

        'change_pct': '{:.1%}',

        'change_pct_of_2019total': '{:.1%}',

    })