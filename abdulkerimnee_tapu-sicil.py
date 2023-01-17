import pandas as pd
import geopandas as gpd
import folium
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
df=pd.read_excel('../input/registering-property/registering_property.xlsx')
df.head()
df.describe().T
df.loc[(df.Tapu_Sicili_Puanı== -9999),'Tapu_Sicili_Puanı']='NaN'
df.loc[(df.İşlem_sayısı==-9999),'İşlem_sayısı']='NaN'
df.loc[(df.İşlem_Süresi==-9999),'İşlem_Süresi']='NaN'
df.loc[(df.İşlem_Masrafı==-9999),'İşlem_Masrafı']='NaN'
df.loc[(df.Arazi_Yönetimi_Kalitesi_Endeksi==-9999),'Arazi_Yönetimi_Kalitesi_Endeksi']='NaN'
df["Arazi_Yönetimi_Kalitesi_Endeksi"] = df["Arazi_Yönetimi_Kalitesi_Endeksi"].astype(str).astype(float)
df["Tapu_Sicili_Puanı"] = df["Tapu_Sicili_Puanı"].astype(str).astype(float)
df["İşlem_sayısı"] = df["İşlem_sayısı"].astype(str).astype(float)
df["İşlem_Süresi"] = df["İşlem_Süresi"].astype(str).astype(float)
df["İşlem_Masrafı"] = df["İşlem_Masrafı"].astype(str).astype(float)
df.describe().T
df.info()
world=gpd.read_file('../input/country-state-geo-location/countries.geo.json')
world.head()
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
for items in df['Ülke'].tolist():
    world_list=world['name'].tolist()
    if items in world_list:
        pass
    else:
        print(items)
for items in world['name'].tolist():
    df_list=df['Ülke'].tolist()
    if items in df_list:
        pass
    else:
        print(items)
df.replace('Yemen, Rep.','Yemen',inplace=True)
df.replace('Syrian Arab Republic','Syria',inplace=True)
df.replace('Russian Federation','Russia',inplace=True)
df.replace('Kyrgyz Republic','Kyrgyzstan',inplace=True)
df.replace('United States','United States of America',inplace=True)
df.replace('Slovak Republic','Slovakia',inplace=True)
df.replace('Republic of Serbia','Serbia',inplace=True)
df.replace('Tanzania','United Republic of Tanzania',inplace=True)
df.replace('Taiwan, China','Taiwan',inplace=True)
df.replace('Egypt, Arab Rep.','Egypt',inplace=True)
df.replace('Iran, Islamic Rep.','Iran',inplace=True)
df.replace('Korea, Rep.','South Korea',inplace=True)
df.replace('Samoa','Somaliland',inplace=True)
df.replace('Lao PDR','Laos',inplace=True)
df.replace('Congo, Dem. Rep.','Democratic Republic of the Congo',inplace=True)
df.replace('Macedonia','North Macedonia',inplace=True)
df.replace('Venezuela, RB','Venezuela',inplace=True)
import folium
m = folium.Map(tiles='cartodbpositron')
m
country=pd.read_csv("../input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv")
country.head()
country=country.dropna(subset=['latitude'])
country=country.dropna(subset=['longitude'])
country.rename(columns={'country':'Ülke'},inplace=True)
country.replace('United States', "United States of America", inplace = True)
combine=df.merge(country,on='Ülke')
combine.head()
choropleth=folium.Choropleth(
    geo_data=world,
    name='Arazi Yönetim Kalite Endeksi',
    data=combine,
    columns=['Ülke', 'Arazi_Yönetimi_Kalitesi_Endeksi'],
    key_on='feature.properties.name',
    legend_name='Arazi Yonetimi Kalitesi Endeksi',
    fill_color='OrRd',
    nan_fill_color='black'
).add_to(m)
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],labels=False)
)
m
choropleth=folium.Choropleth(
    geo_data=world,
    name='Tapu_Sicili_Puanı',
    data=combine,
    columns=['Ülke', 'Tapu_Sicili_Puanı'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    legend_name='Tapu Sicil Puani',
    nan_fill_color='black'
).add_to(m)
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],labels=False)
)
folium.LayerControl().add_to(m)
m
choropleth=folium.Choropleth(
    geo_data=world,
    name='İşlem_sayısı',
    data=combine,
    columns=['Ülke', 'İşlem_sayısı'],
    key_on='feature.properties.name',
    fill_color='RdPu',
    legend_name='Islem Sayisi',
    nan_fill_color='black'
).add_to(m)
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],labels=False)
)
folium.LayerControl().add_to(m)
m