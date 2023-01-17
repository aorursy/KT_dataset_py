from IPython.display import Image



Image("../input/formula-1-grand-prix/f1.png")
!pip install chart-studio
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization libraries

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import chart_studio.plotly as py

from plotly.graph_objs import *

from IPython.display import Image

pd.set_option('display.max_rows', None)



import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
seasons_df = pd.read_csv("../input/formula-1-race-data-19502017/seasons.csv", encoding='latin')

circuits_df = pd.read_csv("../input/formula-1-race-data-19502017/circuits.csv", encoding='latin')

drivers_df = pd.read_csv("../input/formula-1-race-data-19502017/drivers.csv", encoding='latin')

races_df = pd.read_csv("../input/formula-1-race-data-19502017/races.csv", encoding='latin')

results_df = pd.read_csv("../input/formula-1-race-data-19502017/results.csv", encoding='latin')

status_df = pd.read_csv("../input/formula-1-race-data-19502017/status.csv", encoding='latin')

constructors_df = pd.read_csv("../input/formula-1-race-data-19502017/seasons.csv", encoding='latin')
seasons_df.head()
circuits_df.head()
drivers_df.head()
races_df.head()
results_df.head()
status_df.head()
constructors_df.head()
fig = go.Figure(go.Bar(

    x = races_df['year'],

    y = races_df['name'],

    text=['Bar Chart'],

    name='Grand Prix',

    marker_color=races_df['circuitId']

))



fig.update_layout(

    height=800,

    title_text='Grand Prix Timeline',

    showlegend=True

)



fig.show()
# def group_f1(data,country):

#     cases = data.groupby(country).size()

#     cases = np.log(cases)

#     cases = cases.sort_values()

    

#     # Visualize the results

#     fig=plt.figure(figsize=(35,7))

#     plt.yticks(fontsize=8)

#     cases.plot(kind='bar',fontsize=12,color='orange')

#     plt.xlabel('')

#     plt.ylabel('Number of cases',fontsize=10)



# group_f1(covid_19_data,'Country/Region')
import geopandas as gpd

from shapely.geometry import Point, Polygon



%matplotlib inline
geometry = [Point(xy) for xy in zip(circuits_df.lat, circuits_df.lng)]
circuits_df['geometry'] = geometry

circuits_df.head()
Image("../input/formula-1-grand-prix/tableau-results/sheet-1.png")
Image("../input/formula-1-grand-prix/tableau-results/sheet-2.png")
Image("../input/formula-1-grand-prix/tableau-results/sheet-3.png")
Image("../input/formula-1-grand-prix/tableau-results/sheet-4.png")
Image("../input/formula-1-grand-prix/tableau-results/sheet-5.png")
Image("../input/formula-1-grand-prix/tableau-results/sheet-6.png")