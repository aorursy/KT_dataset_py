import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import altair as alt

#alt.renderers.enable('notebook')
df_athlete = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv', index_col='ID')

df_noc = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv', index_col = 'NOC')
df_athlete.info()
df_athlete.head()
df_athlete['Sport'] = df_athlete['Sport'].astype('category')

df_athlete['Sex'] = df_athlete['Sex'].astype('category')

df_athlete['Team'] = df_athlete['Team'].astype('category')

df_athlete['NOC'] = df_athlete['NOC'].astype('category')

df_athlete['Season'] = df_athlete['Season'].astype('category')

df_athlete['Sport'] = df_athlete['Sport'].astype('category')

df_athlete['Event'] = df_athlete['Event'].astype('category')

df_athlete['Medal'].fillna('None', inplace=True)

df_athlete['Medal'] = df_athlete['Medal'].astype('category')
fig, axs = plt.subplots(2,2, figsize=(12,10))

axs = axs.flatten()

#axs[0].hist2d(y=df_athlete['Height'], x=df_athlete['Year'])

axs[0].hist(df_athlete.query('Year > 0')['Year'],bins=120)

axs[1].hist(df_athlete.query('Height > 0')['Height'],bins=100,range=(130,230))

axs[2].hist(df_athlete.query('Age > 0')['Age'],bins=100, range=(0,100))

axs[3].hist(df_athlete.query('Weight > 0')['Weight'],bins=100, range=(25,225))

axs[0].set_xlabel('Year')

axs[1].set_xlabel('Height')

axs[2].set_xlabel('Age')

axs[3].set_xlabel('Weight')

fig.tight_layout()
h1 = df_athlete.query('Season == "Summer"')['Year'].hist(bins = 40, range=(1980, 2020))

h2 = df_athlete.query('Season == "Winter"')['Year'].hist(bins = 40, range=(1980, 2020))
fig, axs = plt.subplots(2,2, figsize=(12,20), gridspec_kw={'height_ratios': [0.5, 5]})

axs = axs.flatten()

sns.countplot(x='Sex', data=df_athlete, ax=axs[0])

sns.countplot(x='Medal', data=df_athlete, ax=axs[1])

sns.countplot(y='Sport', data=df_athlete,  order = df_athlete['Sport'].value_counts().index, ax=axs[2])

noc_df = df_athlete[df_athlete['NOC'].isin(df_athlete.groupby('NOC').count().query('Name > 500').index)]

sns.countplot(y='NOC', data=noc_df, 

              order = noc_df['NOC'].astype('object').value_counts().index, ax=axs[3])

fig.tight_layout()
h = sns.pairplot(df_athlete.query('Weight > 0 and Height > 0 and Age > 0'), kind="reg", hue="Sex", aspect=1.3)
import csv

import json

import re



from collections import Counter, OrderedDict

from IPython.display import HTML

from  altair.vega import v4



##-----------------------------------------------------------

# This whole section 

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v4.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

    "This code block sets up embedded rendering in HTML output and<br/>",

    "provides the function `render(chart, id='vega-chart')` for use below."

)))



df_minmaxyear = df_athlete[['Sport','Year']].groupby('Sport').min()

df_minmaxyear.rename(columns={'Year':'StartYear'},inplace=True)

df_minmaxyear['EndYear'] = df_athlete[['Sport','Year']].groupby('Sport').max()

df_minmaxyear.reset_index(inplace=True)

bars = alt.Chart(df_minmaxyear).mark_bar().encode(

    x= alt.X('StartYear',

        scale=alt.Scale(zero=False)

    ),

    x2='EndYear',

    y='Sport',

    tooltip=['Sport', 'StartYear', 'EndYear']

)



circles1 = alt.Chart(df_minmaxyear).mark_circle().encode(

    x= 'EndYear',

    y='Sport',

    tooltip=['Sport', 'StartYear', 'EndYear']

)#.configure_mark(radius=10)

circles2 = alt.Chart(df_minmaxyear).mark_circle().encode(

    x= 'StartYear',

    y='Sport',

    tooltip=['Sport', 'StartYear', 'EndYear']

)#.configure_mark(radius=10)

render((bars+circles1+circles2).interactive(bind_x=False))



grouped_popular_sports = df_athlete[['Sport','Name']].groupby(['Sport']).count().sort_values(by='Name', ascending=False).head(10)

df_athlete[df_athlete['Sport'].isin(grouped_popular_sports.index)][['Sport','Event']].astype('object').groupby(['Sport','Event']).count().reset_index().groupby('Sport').count().sort_values(by='Event',ascending=False).rename(columns={'Event':'Number of Events'})
fig,ax = plt.subplots(figsize=(10,25))

h = sns.countplot(y='Sport',data=df_athlete, hue='Sex')
year_df = df_athlete.groupby(['Year','Name']).count().groupby('Year').sum()



year_df['AgeNaNRate'] = (year_df['City'] - year_df['Age']) / year_df['City']

year_df['WeightNaNRate'] = (year_df['City'] - year_df['Weight']) / year_df['City']

year_df['HeightNaNRate'] = (year_df['City'] - year_df['Height']) / year_df['City']

h = year_df[['AgeNaNRate', 'HeightNaNRate', 'WeightNaNRate']].plot(figsize=(15,5))
df_diff = df_athlete.groupby('Name').var().sort_values(by='Height',ascending=False)

print('Collision rate: %.2f%%'%(len(df_diff.query('Height > 0'))/len(df_diff)*100))

print('Histogram of Height variances: ')

h = df_diff.query('Height > 0')['Year'].hist(bins = 50, figsize=(5,5))
df_athlete['Height'].hist(range=(100, 220), bins=120, figsize=(15,10))

height_peak_cut = 'Height == 150 or Height == 160 or Height == 165 or Height ==168 or Height == 180 or Height ==170 or Height == 175 or Height == 190 or Height == 178'

df_athlete.query(height_peak_cut)['Height'].hist(range=(100, 220), bins=120, figsize=(15,10))
fig,ax = plt.subplots(figsize=(10,35))

not_nan = df_athlete.query('Height > 0')

not_nan['_peaks'] = not_nan['Height'].isin([150, 160, 165, 168, 170, 178, 180,190])

sns.countplot(data=not_nan, y='NOC', hue='_peaks', ax=ax)
fig,ax = plt.subplots(1,2,figsize=(10,5))

h = df_athlete.query('NOC == "USA"')['Height'].hist(range=(120,220),bins=100, ax=ax[1])

h = df_athlete.query('NOC == "JPN"')['Height'].hist(range=(120,220),bins=100, ax=ax[0])
fig, ax = plt.subplots(1,2,figsize=(12,5))

toymc_ft = np.round( np.random.normal(5.9,0.4, 10000)*48)/48

toymc_cm = np.round(toymc_ft*30.48)

h = ax[0].hist(toymc_ft, bins=48, range=(4,8))

h = ax[1].hist(toymc_cm, bins=100, range=(130,230))