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
!pip install --upgrade pip
!pip install ipywidgets

!pip install keplergl

!jupyter nbextension install --py --sys-prefix keplergl

!jupyter nbextension enable keplergl --py --sys-prefix
!jupyter --version

!python --version

from keplergl import KeplerGl
df = pd.read_csv('../input/hex-data.csv')

df.head()
# Load map with data and config

map_1 = KeplerGl(height=500)

map_1
map_1.add_data(data=df, name='data_1')
config = {

  "version": "v1",

  "config": {

    "visState": {

      "filters": [],

      "layers": [

        {

          "id": "0rceopc",

          "type": "hexagonId",

          "config": {

            "dataId": "data_1",

            "label": "H3 Hexagon",

            "color": [

              18,

              147,

              154

            ],

            "columns": {

              "hex_id": "hex_id"

            },

            "isVisible": True,

            "visConfig": {

              "opacity": 0.8,

              "colorRange": {

                "name": "Uber Viz Diverging 3.5",

                "type": "diverging",

                "category": "Uber",

                "colors": [

                  "#00939C",

                  "#2FA7AE",

                  "#5DBABF",

                  "#8CCED1",

                  "#BAE1E2",

                  "#F8C0AA",

                  "#EB9C80",

                  "#DD7755",

                  "#D0532B",

                  "#C22E00"

                ]

              },

              "coverage": 0.38,

              "enable3d": True,

              "sizeRange": [

                0,

                500

              ],

              "coverageRange": [

                0,

                1

              ],

              "elevationScale": 5

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

              "name": "float_value",

              "type": "real"

            },

            "colorScale": "quantize",

            "sizeField": {

              "name": "float_value",

              "type": "real"

            },

            "sizeScale": "sqrt",

            "coverageField": None,

            "coverageScale": "linear"

          }

        }

      ],

      "interactionConfig": {

        "tooltip": {

          "fieldsToShow": {

            "data_1": [

              "hex_id",

              "value",

              "is_true",

              "float_value",

              "empty"

            ]

          },

          "enabled": True

        },

        "brush": {

          "size": 27,

          "enabled": False

        },

        "geocoder": {

          "enabled": True

        },

        "coordinate": {

          "enabled": True

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

      "bearing": 24,

      "dragRotate": True,

      "latitude": 37.777269696578756,

      "longitude": -122.41787543387213,

      "pitch": 50,

      "zoom": 12.387828128323221,

      "isSplit": False

    },

    "mapStyle": {

      "styleType": "satellite",

      "topLayerGroups": {},

      "visibleLayerGroups": {},

      "threeDBuildingColor": [

        3.7245996603793508,

        6.518049405663864,

        13.036098811327728

      ],

      "mapStyles": {}

    }

  }

}
# save current map and apply the config settings from above cell 

map_1.save_to_html(file_name="my_keplergl_map_config.html", config=config)
# or pass in different data and config

#map_1.save_to_html(data={'data_1': df}, config=config, file_name="my_keplergl_map.html")