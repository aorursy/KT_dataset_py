
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
#--NotebookApp.iopub_data_rate_limit=1000000
df_atp = pd.read_csv("../input/df-atp/df_atp.csv",usecols = df_atp.columns[1:])
df_atp.head()
# df = pd.read_csv("../input/data-4qcsv/data_4q.csv")
# df.drop("Unnamed: 0",axis=1)
train = df_atp.iloc[:30000,:]
test = df_atp.iloc[-10000:,:]
test.head()
#@title Install the facets_overview pip package.
!pip install facets-overview
from IPython.core.display import display, HTML

# jsonstr = df.to_json(orient='records')
# HTML_TEMPLATE = """
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
#         <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
#         <facets-dive id="elem" height="600"></facets-dive>
#         <script>
#           var data = {jsonstr};
#           document.querySelector("#elem").data = data;
#         </script>"""
# html = HTML_TEMPLATE.format(jsonstr=jsonstr)
# display(HTML(html))

import base64
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
gfsg = GenericFeatureStatisticsGenerator()
proto = gfsg.ProtoFromDataFrames([{'name': 'train', 'table': train},
                                  {'name': 'test', 'table': test}])
protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
from IPython.core.display import display, HTML

HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
        <facets-overview id="elem"></facets-overview>
        <script>
          document.querySelector("#elem").protoInput = "{protostr}";
        </script>"""
html = HTML_TEMPLATE.format(protostr=protostr)
display(HTML(html))