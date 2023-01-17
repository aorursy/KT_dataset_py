# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import math

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

import time

import datetime

from time import gmtime, strftime

from pytz import timezone

from bokeh.io import output_file, output_notebook, show

from bokeh.models import (

  GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, ColorBar,

    DataRange1d, PanTool, WheelZoomTool, BoxSelectTool

)

from bokeh.models.mappers import ColorMapper, LinearColorMapper

from bokeh.palettes import Viridis5

from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper

from bokeh.plotting import figure, show, output_file

from bokeh.tile_providers import CARTODBPOSITRON

from ast import literal_eval
patient = pd.read_csv('../input/coronavirusdataset/patient.csv',index_col="patient_id")

time = pd.read_csv('../input/coronavirusdataset/time.csv')

route = pd.read_csv('../input/coronavirusdataset/route.csv',index_col="patient_id")



patient.head()
time.head()
route.head()
acc_data_r_d = [time.loc[:,"released"], time.loc[:,"deceased"]]

plt.figure(figsize=(20, 6))

#sns.lineplot(data=time.loc[:,"confirmed"])

plt.plot(time['date'],time['confirmed'])

plt.title("Accumulated confirmed cases over time", fontsize=16, size=30)

plt.xlabel("Date", fontsize=16, size=30)

plt.ylabel("Number of cases", fontsize=16, size=30)

#plt.yscale("log")

plt.xticks(rotation=90,fontsize=10)

plt.yticks(fontsize=10)

plt.show()



plt.figure(figsize=(20, 6))

#sns.lineplot(data=acc_data_r_d)

plt.plot(time['date'],time['released'])

plt.plot(time['date'],time['deceased'])

plt.title("Accumulated recovered and deceased cases over time", fontsize=16)

plt.xlabel("Date", fontsize=16, size=30)

plt.ylabel("Number of cases", fontsize=16, size=30)

plt.xticks(rotation=90,fontsize=10)

plt.yticks(fontsize=10)

#plt.legend(loc="upper left",fontsize=20)

plt.legend(['Released Cases', 'Deceased cases'],loc="upper left",fontsize=20)

plt.show()



plt.figure(figsize=(20, 6))

#sns.lineplot(data=time.loc[:,"acc_negative"])

plt.plot(time['date'],time['negative'])

plt.title("Accumulated negative cases over time", fontsize=16, size=30)

plt.xlabel("Date", fontsize=16, size=30)

plt.ylabel("Number of cases", fontsize=16, size=30)

#plt.yscale("log")

plt.xticks(rotation=90,fontsize=10)

plt.yticks(fontsize=10)

plt.show()
reason_order = list(patient["infection_reason"].value_counts().index)



plt.figure(figsize=(12, 8))

sns.countplot(y = "infection_reason",

              data=patient,

              order=reason_order)

plt.title("Known reasons of infection", fontsize=16)

plt.xlabel("Count", fontsize=16)

plt.ylabel("Reason of infection", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()

plt.figure(figsize=(12, 8))

sns.countplot(x = "sex",

            hue="state",

            hue_order=["isolated", "released", "deceased"],

            data=patient)

plt.title("Patient state by gender", fontsize=16)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()

patient_status = pd.DataFrame(patient["state"].value_counts())

patient_status.rename(columns={"state": "Count"}, inplace=True)

patient_status.index.name="State"

plt.figure(figsize=(12, 8))

sns.barplot(x = patient_status.index,

            y="Count",

            data=patient_status)

plt.title("Current state of patients", fontsize=16)

plt.xlabel("State", fontsize=16)

plt.ylabel("Count (log scale)", fontsize=16)

plt.yscale("log")

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
# If you want to get the information of selected COVID-19 patients just move the hover tool
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

    

route["cor"] = route.latitude.astype(str).str.cat(route.longitude.astype(str), sep=',')

route['coords_x'] = route['cor'].apply(lambda x: merc(x)[0])

route['coords_y'] = route['cor'].apply(lambda x: merc(x)[1])

cds = ColumnDataSource(route)



hover = HoverTool(tooltips=[ ('id','@id'),('sex','@sex') ,('infection_reason','@infection_reason'),('state','state'),

                            ('longitude', '@longitude'),

                            ('latitude', '@latitude'),

    ('city','@city'),('province','@province'),('visit','@visit')],

                  mode='mouse')

p = figure(x_axis_type="mercator", y_axis_type="mercator",tools=['pan', 'wheel_zoom', 'tap', 'reset', 'crosshair',hover])

p.add_tile(CARTODBPOSITRON)

p.circle(x = route['coords_x'],

         y = route['coords_y'])



scatter = p.circle('coords_x', 'coords_y', source=cds,

                    alpha=.10,

                    selection_color='red',

                    nonselection_fill_alpha=.1)

output_notebook()

show(p)
def pie_plot(cnt_srs, colors, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.6,

                   showlegend=True,

                   marker=dict(colors=colors,

                               line=dict(color='#000000',

                                         width=1),

                              )

                  )

    return trace

py.iplot([pie_plot(patient['region'].value_counts(), ['cyan', 'gold'], 'region')])

py.iplot([pie_plot(patient['infection_reason'].value_counts(), ['cyan', 'gold'], 'infection_reason')])

py.iplot([pie_plot(patient['group'].value_counts(), ['cyan', 'gold'], 'group')])