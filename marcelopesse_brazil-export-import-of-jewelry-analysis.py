import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # beautiful graphs

import matplotlib.pyplot as plt #math stuf

import datetime as dt #for date math

import re #regex

import json #json parser
#Let's import the dataset

#Export

df_export = pd.read_csv('../input/brazil-exportimport-information/EXP_COMPLETA.csv', sep=';')

df_export = df_export[(df_export['CO_NCM'] >= 71000000) & (df_export['CO_NCM'] < 72000000)]

df_export = df_export.reset_index(drop=True)



#Import

df_import = pd.read_csv('../input/brazil-exportimport-information/IMP_COMPLETA.csv', sep=';')

df_import = df_import[(df_import['CO_NCM'] >= 71000000) & (df_import['CO_NCM'] < 72000000)]

df_import = df_import.reset_index(drop=True)
df_country = pd.read_csv('../input/brazil-exportimport-information/PAIS.csv', sep=';', encoding='iso-8859-1')



country_dict = {}

for i, row in df_country.iterrows():

    country_dict[row['CO_PAIS']] = row['NO_PAIS_ING']



#Export

df_export['COUNTRY'] = df_export['CO_PAIS'].map(country_dict)

#Import

df_import['COUNTRY'] = df_import['CO_PAIS'].map(country_dict)
df_block = pd.read_csv('../input/brazil-exportimport-information/PAIS_BLOCO.csv', sep=';', encoding='iso-8859-1')



block_dict = {}

for i, row in df_block.iterrows():

    block_dict[row['CO_PAIS']] = row['NO_BLOCO_ING']



#Export

df_export['BLOCK'] = df_export['CO_PAIS']

df_export['BLOCK'] = df_export['BLOCK'].map(block_dict)



#Import

df_import['BLOCK'] = df_import['CO_PAIS']

df_import['BLOCK'] = df_import['BLOCK'].map(block_dict)
df_via = pd.read_csv('../input/brazil-exportimport-information/VIA.csv', sep=';', encoding='iso-8859-1')



via_dict = {}

for i, row in df_via.iterrows():

    via_dict[row['CO_VIA']] = row['NO_VIA']



#Export

df_export['VIA'] = df_export['CO_VIA']

df_export['VIA'] = df_export['VIA'].map(via_dict)

#Import

df_import['VIA'] = df_import['CO_VIA']

df_import['VIA'] = df_import['VIA'].map(via_dict)
#Export

df_export = df_export.drop(['QT_ESTAT', 'CO_UNID', 'CO_URF', 'CO_VIA'], axis=1)

#Import

df_import = df_import.drop(['QT_ESTAT', 'CO_UNID', 'CO_URF', 'CO_VIA'], axis=1)
#Export

df_export = df_export.rename(columns = {'CO_ANO':'YEAR','CO_MES':'MONTH','CO_NCM':'NCM','SG_UF_NCM':'UF','KG_LIQUIDO':'KG'}, inplace = False)

#Import

df_import = df_import.rename(columns = {'CO_ANO':'YEAR','CO_MES':'MONTH','CO_NCM':'NCM','SG_UF_NCM':'UF','KG_LIQUIDO':'KG'}, inplace = False)
via_ing_dict={

'AEREA':'Air',

'MARITIMA':'Sea',

'MEIOS PROPRIOS':'Own Means',

'RODOVIARIA':'Road',

'POSTAL':'Postal',

'VIA NAO DECLARADA':'Non Declared',

'EM MAOS':'By Hand',

'FERROVIARIA':'Rail',

'VICINAL FRONTEIRICO':'Border',

'FLUVIAL':'Fluvial',

'ENTRADA/SAIDA FICTA':'Ficta',

'COURIER':'Courier'

}



#Export

df_export['VIA'] = df_export['VIA'].map(via_ing_dict)

#Import

df_import['VIA'] = df_import['VIA'].map(via_ing_dict)
#Export

df_export = df_export[df_export['YEAR'] < 2020]

#Import

df_import = df_import[df_import['YEAR'] < 2020]
df_all = pd.concat([df_export.assign(TRADE='Export'), df_import.assign(TRADE='Import')])

df_all = df_all.reset_index(drop=True)



df_group = df_all.groupby(['YEAR','TRADE']).sum()

df_group = df_group.reset_index()



sns.set(style='darkgrid')

plt.figure(figsize=(10, 10))

ax = sns.lineplot(x='YEAR',y='VL_FOB',data=df_group,hue='TRADE')

ax.set(xlabel='Year', ylabel='US$ FOB')
#Group

GEM_MIN = 71010000

GEM_MAX = 71059999

METAL_MIN = 71060000

METAL_MAX = 71129999

JEWELRY_MIN = 71130000

JEWELRY_MAX = 71189999



#Subgroup

PRECIOUS_MIN_1 = 71131000

PRECIOUS_MAX_1 = 71131999

PRECIOUS_MIN_2 = 71141000

PRECIOUS_MAX_2 = 71141999

PLATED_MIN_1 = 71132000

PLATED_MAX_1 = 71132999

PLATED_MIN_2 = 71142000

PLATED_MAX_2 = 71142999

OTHERS_MIN = 71150000

OTHERS_MAX = 71159999

PEARLS_GEM_MIN = 71160000

PEARLS_GEM_MAX = 71169999

BIJOU_MIN = 71170000

BIJOU_MAX = 71179999

COIN_MIN = 71180000

COIN_MAX = 71189999
df_all.loc[(df_all['NCM']>=METAL_MIN) & (df_all['NCM']<=METAL_MAX),'GROUP'] = 'Metal'

df_all.loc[(df_all['NCM']>=GEM_MIN) & (df_all['NCM']<=GEM_MAX),'GROUP'] = 'Gem'

df_all.loc[(df_all['NCM']>=JEWELRY_MIN) & (df_all['NCM']<=JEWELRY_MAX),'GROUP'] = 'Jewelry'
df_all.loc[(df_all['NCM']>=PRECIOUS_MIN_1) & (df_all['NCM']<=PRECIOUS_MAX_1),'SUBGROUP'] = 'Precious'

df_all.loc[(df_all['NCM']>=PRECIOUS_MIN_2) & (df_all['NCM']<=PRECIOUS_MAX_2),'SUBGROUP'] = 'Precious'

df_all.loc[(df_all['NCM']>=PLATED_MIN_1) & (df_all['NCM']<=PLATED_MAX_1),'SUBGROUP'] = 'Plated'

df_all.loc[(df_all['NCM']>=PLATED_MIN_2) & (df_all['NCM']<=PLATED_MAX_2),'SUBGROUP'] = 'Plated'

df_all.loc[(df_all['NCM']>=OTHERS_MIN) & (df_all['NCM']<=OTHERS_MAX),'SUBGROUP'] = 'Others'

df_all.loc[(df_all['NCM']>=PEARLS_GEM_MIN) & (df_all['NCM']<=PEARLS_GEM_MAX),'SUBGROUP'] = 'Pearls & Gems'

df_all.loc[(df_all['NCM']>=BIJOU_MIN) & (df_all['NCM']<=BIJOU_MAX),'SUBGROUP'] = 'Bijou'

df_all.loc[(df_all['NCM']>=COIN_MIN) & (df_all['NCM']<=COIN_MAX),'SUBGROUP'] = 'Coin'
df_group = df_all.groupby(['YEAR','TRADE','GROUP']).sum()

df_group = df_group.reset_index()

df_group = df_group[df_group['TRADE']=='Export']

df_group = df_group.drop(columns=['MONTH','NCM','KG','TRADE'])



df_plot = df_group.pivot(columns='GROUP', index='YEAR', values='VL_FOB')



plt.figure(figsize=(10, 10))

ax.set(xlabel='Year', ylabel='US$ FOB')

df_plot.plot(kind='bar', stacked=True)
df_exchange = pd.read_csv('../input/currency-exchange-rate-usdbrl-19932019/Month.csv',dayfirst=True,parse_dates=[0])

df_exchange['Mean'] = (df_exchange['Max'] + df_exchange['Min'])/2



df_exchange = df_exchange.groupby(df_exchange['Date'].map(lambda x: x.year)).mean()

df_exchange = df_exchange.reset_index()



df_group = df_all.groupby(['YEAR','TRADE']).sum()

df_group = df_group.reset_index()



df_exp_exchange = df_group[df_group['TRADE']=='Export']

df_exp_exchange = df_exp_exchange.merge(df_exchange,left_on='YEAR',right_on='Date')

df_exp_exchange = df_exp_exchange[['VL_FOB','Mean']]

df_exp_exchange = (df_exp_exchange-df_exp_exchange.min())/(df_exp_exchange.max()-df_exp_exchange.min())



print(df_exp_exchange.corr(method='pearson'))



sns.set(style='darkgrid')

plt.figure(figsize=(10, 10))

ax.set(ylabel='EXP Value', xlabel='BRL/USD Exchange')

ax = sns.scatterplot(y='VL_FOB', x='Mean', data=df_exp_exchange)
df_group = df_all.groupby(['YEAR','TRADE','GROUP']).sum()

df_group = df_group.reset_index()

df_group = df_group[df_group['GROUP']=='Jewelry']



sns.set(style='darkgrid')

plt.figure(figsize=(10, 10))

ax.set(xlabel='Year', ylabel='US$ FOB')

ax = sns.lineplot(x='YEAR',y='VL_FOB',data=df_group,hue='TRADE')
df_group = df_all.groupby(['YEAR','TRADE','GROUP']).sum()

df_group = df_group.reset_index()

df_group = df_group[df_group['GROUP']=='Jewelry']

df_exp_exchange = df_group[df_group['TRADE']=='Export']

df_exp_exchange = df_exp_exchange[df_exp_exchange['GROUP']=='Jewelry']



df_exp_exchange = df_exp_exchange.merge(df_exchange,left_on='YEAR',right_on='Date')

df_exp_exchange = df_exp_exchange[['VL_FOB','Mean']]

df_exp_exchange = (df_exp_exchange-df_exp_exchange.min())/(df_exp_exchange.max()-df_exp_exchange.min())



print(df_exp_exchange.corr(method='pearson'))



sns.set(style='darkgrid')

plt.figure(figsize=(10, 10))

ax.set(ylabel='EXP Value', xlabel='BRL/USD Exchange')

ax = sns.scatterplot(y='VL_FOB', x='Mean', data=df_exp_exchange)
df_plot = df_all.groupby(['YEAR','TRADE','SUBGROUP']).sum()

df_plot = df_plot.reset_index()

df_plot = df_plot[df_plot['TRADE']=='Export']

df_plot = df_plot[df_plot['SUBGROUP']!='Coin']

df_plot = df_plot[df_plot['SUBGROUP']!='Others']

df_plot = df_plot.pivot(columns='SUBGROUP', index='YEAR', values='VL_FOB')

plt.figure(figsize=(10, 10))

df_plot.plot(kind='bar', stacked=True)
df_plot = df_all.groupby(['YEAR','TRADE','SUBGROUP']).sum()

df_plot = df_plot.reset_index()

df_plot = df_plot[df_plot['TRADE']=='Import']

df_plot = df_plot[df_plot['SUBGROUP']!='Coin']

df_plot = df_plot[df_plot['SUBGROUP']!='Others']

df_plot = df_plot.pivot(columns='SUBGROUP', index='YEAR', values='VL_FOB')

plt.figure(figsize=(10, 10))

df_plot.plot(kind='bar', stacked=True)
df_plot = df_all.groupby(['YEAR','TRADE','SUBGROUP']).sum()

df_plot = df_plot.reset_index()

df_plot = df_plot[df_plot['SUBGROUP']=='Precious']

#df_plot['VL_FOB'] = np.where(df_plot.TRADE=='Import',-df_plot.VL_FOB,df_plot.VL_FOB)



sns.set(style='darkgrid')

plt.figure(figsize=(10, 10))

ax.set(xlabel='Year', ylabel='US$ FOB')

ax = sns.lineplot(x='YEAR',y='VL_FOB',data=df_plot,hue='TRADE')
df_plot = df_all.groupby(['YEAR','TRADE','SUBGROUP']).sum()

df_plot = df_plot.reset_index()

df_plot = df_plot[df_plot['SUBGROUP']=='Bijou']

#df_plot['VL_FOB'] = np.where(df_plot.TRADE=='Import',-df_plot.VL_FOB,df_plot.VL_FOB)



sns.set(style='darkgrid')

plt.figure(figsize=(10, 10))

ax.set(xlabel='Year', ylabel='US$ FOB')

ax = sns.lineplot(x='YEAR',y='VL_FOB',data=df_plot,hue='TRADE')
!pip install ipywidgets

!pip install keplergl

!jupyter nbextension install --py --sys-prefix keplergl

!jupyter nbextension enable keplergl --py --sys-prefix
df_map = df_all.groupby(['UF','TRADE','SUBGROUP']).sum()

df_map = df_map.reset_index()

df_map = df_map[df_map['SUBGROUP']=='Precious']

df_map = df_map[df_map['TRADE']=='Export']

df_map = df_map.drop(columns=['NCM','KG','TRADE','SUBGROUP','YEAR','MONTH','CO_PAIS'])

df_map = df_map.reset_index(drop=True)



#We removed some UFs when we selected only rows with the subgroup 'Precious'

#For the map we need to put then back

df_uf = pd.read_csv('../input/brazil-exportimport-information/UF.csv', sep=';', encoding='iso-8859-1')

for i, row in df_uf.iterrows():

    if (df_map['UF'].str.contains(row['SG_UF']).any()):

        df_map.loc[(df_map['UF']==row['SG_UF']),'UF_NAME'] = row['NO_UF']

    else:

        new_row = {'UF':row['SG_UF'], 'VL_FOB':0, 'UF_NAME':row['NO_UF']}

        df_map = df_map.append(new_row, ignore_index=True)

#Drop UF's code that don't correlate to a State

df_map = df_map[~df_map['UF'].str.contains('EX|CB|MN|ND|RE|ZN|ED')]
brazil_geo = json.load(open('../input/brazil-geojson/brazil_geo.json'))

brazil_geo = brazil_geo['features']



uf_geo = {}



for uf in brazil_geo:

    uf_geo[uf['id']] = uf['geometry']

    

df_map['GEOMETRY'] = df_map['UF'].map(uf_geo)
config = {

  "version": "v1",

  "config": {

    "visState": {

      "filters": [],

      "layers": [

        {

          "id": "elzzyyt",

          "type": "geojson",

          "config": {

            "dataId": "dj3gzji39",

            "label": "Export Total (1997 to 2019)",

            "color": [

              255,

              203,

              153

            ],

            "columns": {

              "geojson": "GEOMETRY"

            },

            "isVisible": True,

            "visConfig": {

              "opacity": 1,

              "strokeOpacity": 0.8,

              "thickness": 1,

              "strokeColor": [

                38,

                26,

                16

              ],

              "colorRange": {

                "name": "ColorBrewer YlGn-8",

                "type": "sequential",

                "category": "ColorBrewer",

                "colors": [

                  "#ffffe5",

                  "#f7fcb9",

                  "#d9f0a3",

                  "#addd8e",

                  "#78c679",

                  "#41ab5d",

                  "#238443",

                  "#005a32"

                ]

              },

              "strokeColorRange": {

                "name": "Global Warming",

                "type": "sequential",

                "category": "Uber",

                "colors": [

                  "#5A1846",

                  "#900C3F",

                  "#C70039",

                  "#E3611C",

                  "#F1920E",

                  "#FFC300"

                ]

              },

              "radius": 10,

              "sizeRange": [

                0,

                10

              ],

              "radiusRange": [

                0,

                50

              ],

              "heightRange": [

                0,

                500

              ],

              "elevationScale": 5,

              "stroked": True,

              "filled": True,

              "enable3d": False,

              "wireframe": False

            },

            "hidden": False,

            "textLabel": [

              {

                "field": None,

                "color": [

                  255,

                  255,

                  255

                ],

                "size": 18,

                "offset": [

                  0,

                  0

                ],

                "anchor": "start",

                "alignment": "center"

              }

            ]

          },

          "visualChannels": {

            "colorField": {

              "name": "VL_FOB",

              "type": "integer"

            },

            "colorScale": "quantile",

            "sizeField": None,

            "sizeScale": "linear",

            "strokeColorField": None,

            "strokeColorScale": "quantile",

            "heightField": None,

            "heightScale": "linear",

            "radiusField": None,

            "radiusScale": "linear"

          }

        }

      ],

      "interactionConfig": {

        "tooltip": {

          "fieldsToShow": {

            "dj3gzji39": [

              {

                "name": "VL_FOB",

                "format": None

              },

              {

                "name": "UF",

                "format": None

              },

              {

                "name": "UF_NAME",

                "format": None

              }

            ]

          },

          "compareMode": False,

          "compareType": "absolute",

          "enabled": True

        },

        "brush": {

          "size": 0.5,

          "enabled": False

        },

        "geocoder": {

          "enabled": False

        },

        "coordinate": {

          "enabled": False

        }

      },

      "layerBlending": "normal",

      "splitMaps": [],

      "animationConfig": {

        "currentTime": None,

        "speed": 1

      }

    },

    "mapState": {

      "bearing": 0,

      "dragRotate": False,

      "latitude": -14.927493161931578,

      "longitude": -56.124965564305676,

      "pitch": 0,

      "zoom": 2.7402945845847593,

      "isSplit": False

    },

    "mapStyle": {

      "styleType": "dark",

      "topLayerGroups": {},

      "visibleLayerGroups": {

        "label": False,

        "road": False,

        "border": True,

        "building": False,

        "water": True,

        "land": True,

        "3d building": False

      },

      "threeDBuildingColor": [

        9.665468314072013,

        17.18305478057247,

        31.1442867897876

      ],

      "mapStyles": {}

    }

  }

}
from keplergl import KeplerGl



map_export_total = KeplerGl()

map_export_total.add_data(data=df_map, name='dj3gzji39')

map_export_total.config = config



map_export_total
df_map = df_all.groupby(['COUNTRY','TRADE','SUBGROUP']).sum()

df_map = df_map.reset_index()

df_map = df_map[df_map['SUBGROUP']=='Precious']

df_map = df_map[df_map['TRADE']=='Export']

df_map = df_map.drop(columns=['TRADE','SUBGROUP','YEAR','MONTH','NCM','CO_PAIS','KG'])

df_map = df_map.sort_values(by=['VL_FOB'],ascending=False)

df_map = df_map.reset_index(drop=True)
df_country = pd.read_csv('../input/world-capitals-gps/concap.csv', sep=',', encoding='iso-8859-1')

df_country



df_map = df_map.merge(df_country,left_on='COUNTRY',how='left',right_on='CountryName',suffixes=('_map', '_country'))
df_map = df_map.drop(columns=['CountryName','CapitalName','CountryCode','ContinentName'])

df_map[df_map['CapitalLatitude'].isna()]



df_map_missing = pd.DataFrame([

    ['Bahrein', 18314598, 26.216667, 50.583333],

    ['French Guyana', 264708, 18.35, -64.933333],

    ['Virgin Islands (USA)', 229499, 18.333333, -64.833333],

    ['Netherlands Antilles', 131703, 12.226079 -69.060087],

    ["Cote D'Ivore", 53217, 6.85, -5.3],

    ['Lebuan', 30034, 5.315894, 115.219033],

    ['Guadeloupe', 21850, 16.270000, -61.580002],

    ['Reunion', 19821, -21.114444, 55.5325],

    ['Martinique', 8856, 14.666667, -61],

    ['Brunei', 230, 4.890283, 114.942217],

    ['Hong Kong', 4441562, 22.3, 114.2],

], columns=['COUNTRY','VL_FOB','CapitalLatitude','CapitalLongitude'])



df_map = df_map[df_map['CapitalLatitude']!=0]

df_map = df_map.append(df_map_missing)

df_map = df_map.dropna(axis='rows')

df_map = df_map.sort_values(by=['VL_FOB'],ascending=False)

df_map = df_map.reset_index(drop=True)
df_map.head(20)
config = {

  "version": "v1",

  "config": {

    "visState": {

      "filters": [],

      "layers": [

        {

          "id": "a8vgbvm",

          "type": "point",

          "config": {

            "dataId": "cmt54a6kj",

            "label": "Export Total (1997 to 2019)",

            "color": [

              118,

              183,

              61

            ],

            "columns": {

              "lat": "CapitalLatitude",

              "lng": "CapitalLongitude",

              "altitude": None

            },

            "isVisible": True,

            "visConfig": {

              "radius": 10,

              "fixedRadius": False,

              "opacity": 0.8,

              "outline": False,

              "thickness": 2,

              "strokeColor": None,

              "colorRange": {

                "name": "Global Warming",

                "type": "sequential",

                "category": "Uber",

                "colors": [

                  "#5A1846",

                  "#900C3F",

                  "#C70039",

                  "#E3611C",

                  "#F1920E",

                  "#FFC300"

                ]

              },

              "strokeColorRange": {

                "name": "Global Warming",

                "type": "sequential",

                "category": "Uber",

                "colors": [

                  "#5A1846",

                  "#900C3F",

                  "#C70039",

                  "#E3611C",

                  "#F1920E",

                  "#FFC300"

                ]

              },

              "radiusRange": [

                0,

                150

              ],

              "filled": True

            },

            "hidden": False,

            "textLabel": []

          },

          "visualChannels": {

            "colorField": None,

            "colorScale": "quantile",

            "strokeColorField": None,

            "strokeColorScale": "quantile",

            "sizeField": {

              "name": "VL_FOB",

              "type": "integer"

            },

            "sizeScale": "sqrt"

          }

        }

      ],

      "interactionConfig": {

        "tooltip": {

          "fieldsToShow": {

            "cmt54a6kj": [

              {

                "name": "COUNTRY",

                "format": None

              },

              {

                "name": "VL_FOB",

                "format": None

              }

            ]

          },

          "compareMode": False,

          "compareType": "absolute",

          "enabled": True

        },

        "brush": {

          "size": 0.5,

          "enabled": False

        },

        "geocoder": {

          "enabled": False

        },

        "coordinate": {

          "enabled": False

        }

      },

      "layerBlending": "normal",

      "splitMaps": [],

      "animationConfig": {

        "currentTime": None,

        "speed": 1

      }

    },

    "mapState": {

      "bearing": 0,

      "dragRotate": False,

      "latitude": 8.884265547401553,

      "longitude": -19.06778798832715,

      "pitch": 0,

      "zoom": 0.8934685912520477,

      "isSplit": False

    },

    "mapStyle": {

      "styleType": "dark",

      "topLayerGroups": {},

      "visibleLayerGroups": {

        "label": False,

        "road": False,

        "border": True,

        "building": False,

        "water": True,

        "land": False,

        "3d building": False

      },

      "threeDBuildingColor": [

        9.665468314072013,

        17.18305478057247,

        31.1442867897876

      ],

      "mapStyles": {}

    }

  }

}
map_export_world = KeplerGl()

map_export_world.add_data(data=df_map, name='cmt54a6kj')

map_export_world.config = config



map_export_world