!pip install altair vega_datasets -q

#import geoplot
import geopandas as gpd
import pandas as pd
import unicodedata
import json
import altair as alt
from vega_datasets import *

#pd.set_option('display.max_rows', 200)
#d.set_option('display.max_columns', 30)
# Sao Paulo city data on number of confirmed COVID-19 cases and deceased on April 13th, 2020 by neighbourhood
# These figures are not public, and were specially requested using the page "Sistema e-SIC"

doentes = pd.read_csv('/kaggle/input/saopaulo/mortos_cidade_sp.csv',encoding='ISO-8859-1',sep=';')
doentes.columns = ['bairro','nao identificada','U04','total']
#  retira acentos e converte para uppercase
doentes['bairro'] = doentes['bairro'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode("utf-8").upper() )
# read a shapefile , but you need to copy all the other files too
bairros = gpd.read_file('/kaggle/input/sampa-shape/DEINFO_DISTRITO.shp')

# convert projection , ignore warning message
bairros.crs = {'init' :'epsg:5641'}             ###  SIRGAS 2000 / Brazil Mercator
bairros = bairros.to_crs({'init': 'epsg:4326'}) ### WSG84

# rename column names
bairros.columns = ['CLASSID', 'FEATID', 'REVISIONNU', 'bairro', 'SIGLA_DIST',
       'COD_DIST', 'COD_SUB', 'DATA_CRIAC', 'USUARIO_ID', 'geometry']

# Making field content identical with other dataframes - to enable merge 
bairros.at[37,'bairro'] = 'CIDADE ADEMAR'
bairros.at[38,'bairro'] = 'CIDADE DUTRA'
bairros.at[39,'bairro'] = 'CIDADE LIDER'
bairros.at[40,'bairro'] = 'CIDADE TIRADENTES'
bairros.at[1,'bairro']  = 'JARDIM SAO LUIS'
bairros.at[7,'bairro']  = 'JARDIM HELENA'
bairros.at[8,'bairro']  = 'JARDIM PAULISTA'
bairros.at[14,'bairro'] = 'JARDIM ANGELA'       
# read file containing population data per bairro/neighbourhood . subprefeitura = county
dsub = pd.read_csv('/kaggle/input/saopaulo/subprefeituras.csv',encoding='ISO-8859-1',sep=';')

# select some columns and rename them
colunas = [0,1,5,13]
dp = dsub.iloc[6:102,colunas].reset_index()
dp.columns = ['index', 'bairro', 'subprefeitura', 'populacao', 'tx pop/ha']

# remove accents from field and change string to uppercase
dp['bairro'] = dp['bairro'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode("utf-8").upper() )
# remove commas from numerical field ( in Brazil they indicate thousands)
dp['populacao'] =  dp['populacao'].apply(lambda x: int(x.replace(",", "") ))

# merge all 3 dataframes , using the same field - "bairro"
df1= pd.merge(bairros,dp,how='inner')
df = pd.merge(df1,doentes,how='inner')
# to facilate the analysis, calculate figures by subprefeitura/county and add back to merged dataframe(by neighbourhood)

df['tx_mortos'] =( df['total'] * 100000 / df['populacao']).round(1)

colunas = ['U04','total','populacao']
dsubpref = df[colunas].groupby(df['subprefeitura']).sum()
dsubpref.columns = ['U04_subpref','total_subpref','populacao_subpref']
dsubpref = dsubpref.reset_index()
dsubpref['tx_mortos_subpref'] = (dsubpref['total_subpref'] * 100000 / dsubpref['populacao_subpref']).round(1)
dsubpref.sort_values('tx_mortos_subpref', axis = 0, ascending = False, 
                 inplace = True, na_position ='last') 


# merge recently calculated fields in data , ready to plot
data = pd.merge(df,dsubpref,how="inner")
## show data for a specific subprefeitura/county
data[data.subprefeitura =='Casa Verde/Cachoeirinha'].T
# dump from pandas to json
json_gdf = data.to_json()
# load back as a GeoJSON object.
json_features = json.loads(json_gdf)

## display interactive chart , but displaying properties.fieldname legends 
# chart object
data_geojson = alt.InlineData(values=json_features, format=alt.DataFormat(property='features',type='json')) 
alt.Chart(data_geojson,title ='Densidade demografica pop/ha').mark_geoshape(
).encode(
    color=alt.Color("properties.tx pop/ha:N",type='quantitative', scale=alt.Scale(scheme='yelloworangered')),
    tooltip=['properties.bairro:O','properties.subprefeitura:O','properties.total:Q','properties.tx_mortos_subpref:Q']    
).properties(
        width=500,
        height=500
    )    
## display interactive chart , but displaying meaningful fieldnames/titles
# define inline geojson data object
data_geojson = alt.InlineData(values=json_features, format=alt.DataFormat(property='features',type='json')) 

# chart object
chart = alt.Chart(data_geojson,title ='Óbitos por bairros de SP em 13/4/20 - obitos por 100 mil habitantes').mark_geoshape(
).encode(
    color=alt.Color("properties.tx_mortos_subpref:N",type='quantitative', scale=alt.Scale(scheme='yelloworangered'),legend=alt.Legend(title="taxa de obitos por subprefeitura")),
    tooltip=[alt.Tooltip('properties.bairro:O',title='bairro'),alt.Tooltip('properties.subprefeitura:O',title='subprefeitura'),alt.Tooltip('properties.total:Q',title='total óbitos'),alt.Tooltip('properties.tx_mortos:Q',title='taxa mortos/100k hab'),alt.Tooltip('properties.tx_mortos_subpref:Q',title= 'taxa mortos/100k hab - subprefeitura')]    
).properties(
        width=500,
        height=500
    )    
chart
chart.save('saopaulo_deaths_neighborhood.html')