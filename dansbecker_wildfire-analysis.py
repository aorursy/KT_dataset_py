import numpy as np

import pandas as pd

import sqlite3

import altair as alt



pd.set_option('display.float_format', lambda x: '%.2f' % x)

conn = sqlite3.connect('../input/188-million-us-wildfires/FPA_FOD_20170508.sqlite')

fires = pd.read_sql("""SELECT * FROM fires""", con=conn)



# table_list = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con=conn)

# table_list
FIRE_SIZE_THRESHOLD = 500



big_fires = fires[fires.FIRE_SIZE>FIRE_SIZE_THRESHOLD]

big_fire_count = pd.DataFrame(big_fires.groupby(['STAT_CAUSE_DESCR']).OBJECTID.count().sort_values(ascending=False)).reset_index()

big_fire_count.rename(columns={'OBJECTID': 'num_fires'}, inplace=True)

alt.Chart(big_fire_count).mark_bar().encode(

    alt.X('STAT_CAUSE_DESCR', sort='-y', title=None),

    alt.Y('num_fires', title='Number of Fires'),

).properties(title=f'Count of Fires Larger than {FIRE_SIZE_THRESHOLD} Acres',

    width=600,

    height=300

).configure_axis(

    labelFontSize=14,

    titleFontSize=14

)

acres_by_cause = pd.DataFrame(fires.groupby(['STAT_CAUSE_DESCR']).FIRE_SIZE.sum().sort_values(ascending=False)).reset_index()

acres_by_cause.rename(columns={'FIRE_SIZE': 'Acres'}, inplace=True)

alt.Chart(acres_by_cause).mark_bar().encode(

    alt.X('STAT_CAUSE_DESCR', sort='-y', title=None),

    alt.Y('Acres'),

).properties(

    title=f'Acres Burned by Cause',

    width=600,

    height=300

).configure_axis(

    labelFontSize=14,

    titleFontSize=14

)

acres_by_year = pd.DataFrame(fires.groupby(['FIRE_YEAR']).FIRE_SIZE.sum().sort_values(ascending=False)).reset_index()

acres_by_year.rename(columns={'FIRE_SIZE': 'Acres'}, inplace=True)



alt.Chart(acres_by_year).mark_line().encode(

    alt.X('FIRE_YEAR:N', title=None),

    alt.Y('Acres'),

).properties(

    title=f'Acres Burned by Year',

    width=600,

    height=300,

).configure_axis(

    labelFontSize=14,

    titleFontSize=14

)

frac_by_lightning = (fires.query('STAT_CAUSE_DESCR=="Lightning"').groupby(['FIRE_YEAR']).FIRE_SIZE.sum() / acres_by_year.set_index('FIRE_YEAR').Acres)

frac_by_lightning.name = 'frac_acres_lightning'

frac_by_lightning = pd.DataFrame(frac_by_lightning).reset_index()



alt.Chart(frac_by_lightning).mark_line().encode(

    alt.X('FIRE_YEAR:N', title=None),

    alt.Y('frac_acres_lightning', title=None),

).properties(

    title=f'Fraction of Burned Acres Caused By Lightning Induced Fires',

    width=600,

    height=300

).configure_axis(

    labelFontSize=14,

    titleFontSize=14

)
