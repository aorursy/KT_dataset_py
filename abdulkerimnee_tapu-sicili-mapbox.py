import pandas as pd

import geopandas as gpd

import numpy as np

import plotly 

import plotly.express as px
df=pd.read_excel('../input/registering-property/registering_property.xlsx')

df.head()
df.loc[(df.Tapu_Sicili_Puanı== -9999),'Tapu_Sicili_Puanı']='NaN'

df.loc[(df.İşlem_sayısı==-9999),'İşlem_sayısı']='NaN'

df.loc[(df.İşlem_Süresi==-9999),'İşlem_Süresi']='NaN'

df.loc[(df.İşlem_Süresi==-9999),'İşlem_Masrafı']='NaN'

df.loc[(df.Arazi_Yönetimi_Kalitesi_Endeksi==-9999),'Arazi_Yönetimi_Kalitesi_Endeksi']='NaN'
df["Arazi_Yönetimi_Kalitesi_Endeksi"] = df["Arazi_Yönetimi_Kalitesi_Endeksi"].astype(str).astype(float)

df["Tapu_Sicili_Puanı"] = df["Tapu_Sicili_Puanı"].astype(str).astype(float)
df.drop(df[df.Ülke=='Bangladesh - Chittagong'].index, inplace=True)

df.drop(df[df.Ülke=='Bangladesh - Dhaka'].index, inplace=True)

df.drop(df[df.Ülke=='Barbados'].index, inplace=True)

df.drop(df[df.Ülke=='Brazil - Rio de Janeiro'].index, inplace=True)

df.drop(df[df.Ülke=='Brazil - São Paulo'].index, inplace=True)

df.drop(df[df.Ülke=='Brunei Darussalam'].index, inplace=True)

df.drop(df[df.Ülke=='Cabo Verde'].index, inplace=True)

df.drop(df[df.Ülke=='China - Beijing'].index, inplace=True)

df.drop(df[df.Ülke=='China - Shanghai'].index, inplace=True)

df.drop(df[df.Ülke=='Comoros'].index, inplace=True)

df.drop(df[df.Ülke=='Dominica'].index, inplace=True)

df.drop(df[df.Ülke=='Eswatini'].index, inplace=True)

df.drop(df[df.Ülke=='Gambia, The'].index, inplace=True)

df.drop(df[df.Ülke=='Grenada'].index, inplace=True)

df.drop(df[df.Ülke=='Guinea-Bissau'].index, inplace=True)

df.drop(df[df.Ülke=='Hong Kong SAR, China'].index, inplace=True)

df.drop(df[df.Ülke=='India - Delhi'].index, inplace=True)

df.drop(df[df.Ülke=='India - Mumbai'].index, inplace=True)

df.drop(df[df.Ülke=='Indonesia - Jakarta'].index, inplace=True)

df.drop(df[df.Ülke=='Indonesia - Surabaya'].index, inplace=True)

df.drop(df[df.Ülke=='Japan - Osaka'].index, inplace=True)

df.drop(df[df.Ülke=='Japan - Tokyo'].index, inplace=True)

df.drop(df[df.Ülke=='Mexico - Mexico City'].index, inplace=True)

df.drop(df[df.Ülke=='Mexico - Monterrey'].index, inplace=True)

df.drop(df[df.Ülke=='Nigeria - Kano'].index, inplace=True)

df.drop(df[df.Ülke=='Nigeria - Lagos'].index, inplace=True)

df.drop(df[df.Ülke=='Pakistan - Karachi'].index, inplace=True)

df.drop(df[df.Ülke=='Pakistan - Lahore'].index, inplace=True)

df.drop(df[df.Ülke=='United States - Los Angeles'].index, inplace=True)

df.drop(df[df.Ülke=='United States - New York City'].index, inplace=True)

df.drop(df[df.Ülke=='Timor-Leste'].index, inplace=True)

df.drop(df[df.Ülke=='Tonga'].index, inplace=True)

df.drop(df[df.Ülke=='West Bank and Gaza'].index, inplace=True)

df.drop(df[df.Ülke=='San Marino'].index, inplace=True)

df.drop(df[df.Ülke=='Russian Federation - Moscow'].index, inplace=True)

df.drop(df[df.Ülke=='Russian Federation - Saint Petersburg'].index, inplace=True)

df.drop(df[df.Ülke=='Palau'].index, inplace=True)

df.drop(df[df.Ülke=='St. Lucia'].index, inplace=True)

df.drop(df[df.Ülke=='Hong Kong, China'].index,inplace=True)
country=pd.read_csv("../input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv")

country.head()
country=country.dropna(subset=['latitude'])

country=country.dropna(subset=['longitude'])
country.rename(columns={'country':'Ülke'},inplace=True)

combine=df.merge(country,on='Ülke')
access_token = 'pk.eyJ1IjoiYWJkdWxrZXJpbW5lc2UiLCJhIjoiY2s5aThsZWlnMDExcjNkcWFmaWUxcmh3YyJ9.s-4VLvmoPQFPXdu9Mcd6pA'

px.set_mapbox_access_token(access_token)
new_df = px.data.gapminder().query("year==2007")
for items in df['Ülke'].tolist():

    new_df_list=new_df['country'].tolist()

    if items in new_df_list:

        pass

    else:

        print(items)
for items in new_df['country'].tolist():

    df_list=df['Ülke'].tolist()

    if items in df_list:

        pass

    else:

        print(items)
new_df.replace('Iran, Islamic Rep.','Iran',inplace=True)

new_df.replace('Syrian Arab Republic','Syria',inplace=True)

new_df.replace('Korea, Rep.','South Korea',inplace=True)

new_df.replace('Egypt, Arab Rep.','Egypt',inplace=True)

new_df.replace('Russian Federation','Russia',inplace=True)

new_df.replace('Taiwan, China','Taiwan',inplace=True)
new_df.rename(columns={'country':'Ülke'},inplace=True)

ndf=df.merge(new_df,on='Ülke')
fig = px.choropleth(ndf, locations="iso_alpha",

                    color="Arazi_Yönetimi_Kalitesi_Endeksi",

                    hover_name="Ülke",

                    title='Arazi Yönetimi Kitle Endeksi',

                    color_continuous_scale=px.colors.cyclical.IceFire)

fig.show()
fig = px.choropleth(ndf, locations="iso_alpha",

                    color="Tapu_Sicili_Puanı",

                    hover_name="Ülke",

                    title='Tapu Sicil Puanı',

                    color_continuous_scale=px.colors.cyclical.IceFire)

fig.show()