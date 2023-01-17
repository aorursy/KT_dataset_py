import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
from  pandas import json_normalize

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/hicp-inflation-rate/Eurostat_Table_HICPv2.csv')
df
df.info()
value_vars = df.columns.tolist()[1:]
value_vars
df = pd.melt(frame=df, id_vars='geo', value_vars=df.columns.tolist()[1:], var_name='years', value_name='inflation_rate')
df
df.geo.unique()
filter1 = (df.geo != 'European Union (changing composition)')&(df.geo != 'EU (27 countries - from 2020)')\
            &(df.geo != 'EU (28 countries)')&(df.geo != 'Euro area (changing composition)')\
            &(df.geo != 'EU (28 countries)')&(df.geo != 'Euro area (changing composition)')\
            &(df.geo != 'Euro area - 19 countries  (from 2015)')&(df.geo != 'Euro area - 18 countries (2014)')
fig = px.line(df[filter1], x="years", y="inflation_rate", color="geo",title= 'Inflation Rates Comparison')
fig.show()
df_geo = json_normalize(pd.read_json('/kaggle/input/covid19-stream-data/json')['records'])[['countriesAndTerritories'
                                                                                            , 'countryterritoryCode']].drop_duplicates().reset_index(drop=True)
df_geo
df.loc[df.geo == 'United Kingdom','geo']='United_Kingdom'
df.loc[df.geo == 'United States','geo']='United_States'
df.loc[df.geo == 'North Macedonia','geo']='North_Macedonia'
df_geo = pd.merge(df[filter1].reset_index(drop=True), df_geo, left_on='geo', right_on='countriesAndTerritories').drop(columns=['countriesAndTerritories'])
df_geo
fig = px.choropleth(df_geo, locations="countryterritoryCode",
                    color="inflation_rate",
                    hover_name="geo",
                    animation_frame="years",
                    title = "Yearly Inflation Rates",
                    color_continuous_scale="Sunsetdark",
                    projection = 'equirectangular')

fig.update_geos(fitbounds="locations")
fig.update_layout(margin={'r':0,'t':50,'l':0,'b':0})
fig.show()
df_past = pd.read_csv('../input/hicp-inflation-rate/prc_hicp_aind/prc_hicp_aind_1_Data.csv')
df_past