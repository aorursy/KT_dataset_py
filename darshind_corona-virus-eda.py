import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

%matplotlib inline
patient_data = pd.read_csv('../input/corona-virus/patient.csv')

route_data = pd.read_csv('../input/corona-virus/route.csv')

time_data = pd.read_csv('../input/corona-virus/time.csv')
patient_data.head()
route_data.head()
time_data.head()
patient_data['country'].unique()
patient_clean_df = patient_data[~patient_data['sex'].isnull()]
patient_clean_df['infection_reason'].fillna('Reason not Listed',inplace=True)

patient_clean_df['birth_year'].fillna('1800',inplace=True)

patient_clean_df['region'].fillna('Not Available',inplace=True)

patient_clean_df['group'].fillna('Not Available',inplace=True)

patient_clean_df['infection_order'].fillna('Not Available',inplace=True)

patient_clean_df['infected_by'].fillna('Not Available',inplace=True)

patient_clean_df['contact_number'].fillna('Not Available',inplace=True)

patient_clean_df['contact_number'].fillna('Not Available',inplace=True)

patient_clean_df['contact_number'].fillna('Not Available',inplace=True)

patient_clean_df['contact_number'].fillna('Not Available',inplace=True)
plt.figure(figsize=(6,4))

sns.countplot(patient_clean_df['state'])

print(patient_clean_df['state'].value_counts())
plt.figure(figsize=(6,4))

sns.countplot(patient_clean_df['country'])

print(patient_clean_df['country'].value_counts())
plt.figure(figsize=(6,4))

sns.countplot(patient_clean_df['sex'])

print(patient_clean_df['sex'].value_counts())
patient_clean_df['birth_year'] = patient_clean_df['birth_year'].astype(int)

patient_clean_df['Age'] = (pd.datetime.now().year) - patient_clean_df['birth_year']

patient_clean_df['Age_Category'] = pd.cut(patient_clean_df['Age'],bins=[0,10,20,30,40,50,60,70,80,90,100,500])

patient_clean_df.head()
sns.countplot(patient_clean_df['Age_Category'])

print(patient_clean_df['Age_Category'].value_counts())
pivot_df = patient_clean_df.copy()

pivot_df.set_index('Age_Category',inplace=True)

#pivot_df.groupby(['Age_Category','sex'])['Age'].count().unstack(0)
pivot_df.groupby(['sex','Age_Category'])['Age'].count().unstack(0).plot(kind='bar')
patient_clean_df['infection_reason'].value_counts().plot(kind='barh')

print(patient_clean_df['infection_reason'].value_counts())
patient_clean_df['region'].value_counts().plot(kind='barh')

print(patient_clean_df['region'].value_counts())
import pandas as pd

import matplotlib.pyplot as plt

from bokeh.plotting import figure, show, output_file

from bokeh.tile_providers import CARTODBPOSITRON

import pandas as pd

pd.options.display.float_format = '{:.2f}'.format

import numpy as np 

import math

from ast import literal_eval

from bokeh.palettes import Viridis5

from bokeh.models import ColumnDataSource,ColorBar,BasicTicker

from bokeh.models.mappers import ColorMapper, LinearColorMapper

from bokeh.plotting import figure, show, output_notebook

from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper

from bokeh.transform import linear_cmap

from bokeh.tile_providers import CARTODBPOSITRON_RETINA

from bokeh.palettes import Category20b, Category20c, Spectral,Category20

from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_file

from bokeh.tile_providers import CARTODBPOSITRON

from ast import literal_eval

import warnings

warnings.filterwarnings('ignore')
from bokeh.io import output_file, output_notebook, show

from bokeh.models import (

  GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, ColorBar,

    DataRange1d, PanTool, WheelZoomTool, BoxSelectTool

)

from bokeh.models.mappers import ColorMapper, LinearColorMapper

from bokeh.palettes import Viridis5
df=pd.read_csv("../input/corona-virus/patient_route_data.csv",index_col=0)

df.head()
import math

from ast import literal_eval

def merc(Coords):

    Coordinates = literal_eval(Coords)

    lat = Coordinates[0]

    lon = Coordinates[1]

    

    r_major = 6378137.000

    x = r_major * math.radians(lon)

    scale = x/lon

    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + 

        lat * (math.pi/180.0)/2.0)) * scale

    return (x, y)



def make_tuple_str(x, y):

    t = (x, y)

    return str(t)
df["cor"] = df.latitude.astype(str).str.cat(df.longitude.astype(str), sep=',')
df['coords_x'] = df['cor'].apply(lambda x: merc(x)[0])

df['coords_y'] = df['cor'].apply(lambda x: merc(x)[1])
df.head()
# create ColumnDataSource

cds = ColumnDataSource(df)



hover = HoverTool(tooltips=[ ('id','@id'),('sex','@sex') ,('infection_reason','@infection_reason'),('state','state'),

                            ('longitude', '@longitude'),

                            ('latitude', '@latitude'),

    ('city','@city'),('province','@province'),('visit','@visit')],

                  mode='mouse')

p = figure(x_axis_type="mercator", y_axis_type="mercator",tools=['pan', 'wheel_zoom', 'tap', 'reset', 'crosshair',hover])

p.add_tile(CARTODBPOSITRON)

p.circle(x = df['coords_x'],

         y = df['coords_y'])



scatter = p.circle('coords_x', 'coords_y', source=cds,

                    alpha=.10,

                    selection_color='black',

                    nonselection_fill_alpha=.1)

output_notebook()

show(p)