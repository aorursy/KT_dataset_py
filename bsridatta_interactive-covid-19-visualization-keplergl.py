!pip install ipywidgets

!pip install keplergl

!jupyter nbextension install --py --sys-prefix keplergl

!jupyter nbextension enable keplergl --py --sys-prefix
import pandas as pd

from keplergl import KeplerGl
data_path = "/kaggle/input/covid-19-italy-updated-regularly/"

prov_df = pd.read_csv(f"{data_path}/provincial_data.csv")

prov_df.rename(columns={"lat":"Latitude",'long':"Longitude"}, inplace=True)

prov_df.head()
config = {'version': 'v1',

 'config': {'visState': {'filters': [{'dataId': 'provincial',

     'id': 'vt87bok5w',

     'name': 'date',

     'type': 'timeRange',

     'value': [1582567200000, 1584921902000],

     'enlarged': True,

     'plotType': 'lineChart',

     'yAxis': {'name': 'total_positive_cases', 'type': 'integer'}}],

   'layers': [{'id': 'yxwp6y',

     'type': 'hexagon',

     'config': {'dataId': 'provincial',

      'label': 'Hexbin',

      'color': [18, 147, 154],

      'columns': {'lat': 'Latitude', 'lng': 'Longitude'},

      'isVisible': True,

      'visConfig': {'opacity': 0.69,

       'worldUnitSize': 6.1546,

       'resolution': 8,

       'colorRange': {'name': 'Sunrise 8',

        'type': 'sequential',

        'category': 'Uber',

        'colors': ['#194266',

         '#355C7D',

         '#63617F',

         '#916681',

         '#C06C84',

         '#D28389',

         '#E59A8F',

         '#F8B195'],

        'reversed': False},

       'coverage': 0.8,

       'sizeRange': [0, 541],

       'percentile': [0, 100],

       'elevationPercentile': [0, 100],

       'elevationScale': 16.7,

       'colorAggregation': 'average',

       'sizeAggregation': 'average',

       'enable3d': True},

      'textLabel': [{'field': None,

        'color': [255, 255, 255],

        'size': 18,

        'offset': [0, 0],

        'anchor': 'start',

        'alignment': 'center'}]},

     'visualChannels': {'colorField': {'name': 'total_positive_cases',

       'type': 'integer'},

      'colorScale': 'quantile',

      'sizeField': {'name': 'total_positive_cases', 'type': 'integer'},

      'sizeScale': 'linear'}}],

   'interactionConfig': {'tooltip': {'fieldsToShow': {'provincial': ['date',

       'state',

       'region_code',

       'region_denomination',

       'province_code']},

     'enabled': True},

    'brush': {'size': 0.4, 'enabled': False}},

   'layerBlending': 'additive',

   'splitMaps': [],

   'animationConfig': {'currentTime': None, 'speed': 1}},

  'mapState': {'bearing': 24,

   'dragRotate': True,

   'latitude': 42.43901323512049,

   'longitude': 11.072761885864756,

   'pitch': 50,

   'zoom': 4.9559534769326,

   'isSplit': False},

  'mapStyle': {'styleType': 'dark',

   'topLayerGroups': {},

   'visibleLayerGroups': {'label': True,

    'road': False,

    'border': False,

    'building': False,

    'water': True,

    'land': True,

    '3d building': False},

   'threeDBuildingColor': [9.665468314072013,

    17.18305478057247,

    31.1442867897876],

   'mapStyles': {}}}}
# Load an empty map

from keplergl import KeplerGl

map_1 = KeplerGl()

map_1.add_data(data=prov_df, name='provincial')

map_1.config = config
map_1
map_1.save_to_html(file_name="my_keplergl_map.html")
