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
! pip install calmap

! pip install pycountry

! pip install pycountry-convert

! pip install chart-studio

! pip install plotly-geo

# # #dependencies needed
import os 

import re

import json, requests

import random

from urllib.request import urlopen



import numpy as np

import pandas as pd

from pandas.core.groupby import generic as groupby_generic



import matplotlib.pyplot as plt

from matplotlib import ticker

import pycountry_convert as pc



import seaborn as sns

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go

import plotly.figure_factory as ff

import geopandas as gpd

import calmap

import folium

import branca

import cufflinks as cf

import matplotlib

import plotly

import plotly.offline as py

import plotly.graph_objects as go
df = pd.read_csv ('../input/nys-chemical-dependence-treatment-prog-admissions/chemical-dependence-treatment-program-admissions-beginning-2007.csv') #dataset provided by OASAS Kaggle

other_df = pd.read_csv('../input/treatment-providers-new-york-state/Treatment_Providers_OASAS_Directory_.csv')#external dataset also provided by OASAS

other_df.drop_duplicates(subset="PROGRAM_NAME")#dropping names of program facilites
df
other_df
admission_type = pd.DataFrame(df.groupby(['Year'])['Admissions'].sum()).reset_index()

fig1 = go.Figure()



fig1.add_trace(go.Scatter(

    x=admission_type['Year'], 

    y=admission_type['Admissions'], 

    name='Rate of decline'))



fig1.add_trace(go.Scatter(

    x=admission_type['Year'], 

    y=admission_type['Admissions'], 

    mode='markers', 

    name='Total admits', 

    marker=dict(color='Red',line=dict(width=5, color='Red'))))



fig1.layout.update(

    title_text='Total Admissions (2007-2018)',

    

    xaxis_showgrid=False, yaxis_showgrid=False, 

    width=850,

    height=800,font=dict(

    family='Arial',

    size=20,

    color="Black"),

    

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial Bold',

            size=20,

            color='#000000',

        )

    )

)

fig1.show()
last_year = df[df['Year']>= 2018]

last_year = pd.DataFrame(last_year.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

last_year = last_year.sort_values('Admissions',ascending=True).set_index('County of Program Location')

last_year = pd.DataFrame(last_year)

last_year['group'] = last_year.index



last_year_color = ['#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100',

 '#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100',

 '#FF9100','#FF9100', '#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100',

 '#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#FF9100','#d87b00',

 '#d87b00','#d87b00','#d87b00','#d87b00']





last_year_data  = go.Data([

    go.Bar(

        y = last_year.group,

        x = last_year['Admissions'],

        text=last_year['Admissions'],

        marker=dict(

            color=last_year_color),

            orientation='h',

            textposition='outside',

            width=.70)])

last_year_layout = go.Layout(

    font=dict(

        color='#000000',

        family='Arial',

        size=11),

    height = 1000,

    width = 1085,

    margin=go.Margin(l=100),

    title = "Rate of Admissions: 2018",

    titlefont=dict(

        color='#000000',

        family='Arial  ',

        size=15),

    xaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial  ',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Counties",

        titlefont=dict(

            family='Arial  ',

            size=14,

            color='#000000',

        )

    )

)

last_year_fig  = go.Figure(data=last_year_data, layout=last_year_layout)

last_year_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(last_year_fig)
counties_df = df[df['Year']>= 2016]

df_type = pd.DataFrame(counties_df.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

df_type = df_type.sort_values('Admissions',ascending=True).set_index('County of Program Location')

df_type = pd.DataFrame(df_type)

df_type['group'] = df_type.index





df_type_color = ['#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC',

                     '#0099CC','#0099CC','#0099CC','#0099CC','#0099CC', '#0099CC','#0099CC','#0099CC','#0099CC','#0099CC',

                     '#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC',

                     '#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC',

                     '#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC','#0099CC',

                     '#0099CC','#0099CC','#0099CC','#0099CC','#003a79','#003a79','#003a79','#003a79','#003a79'

                     ]

df_type_data  = go.Data([

            go.Bar(

                y = df_type.group,

                x = df_type['Admissions'],

                text=df_type['Admissions'],

                marker=dict(color=

                    df_type_color),

                orientation='h',

                textposition='outside',

                width=.70,

            )])

df_type_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=11),

    height = 1000,

    width = 1085,

    margin=go.Margin(l=100),

    title = "Rate of Admissions: 2016-2018",

    titlefont=dict(color='#000000',

                   family='Arial  ',

                   size=15),

    xaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial  ',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Counties",

        titlefont=dict(

            family='Arial  ',

            size=14,

            color='#000000',

        )

    )

)        

df_type_fig = go.Figure(data=df_type_data, layout=df_type_layout)

df_type_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(df_type_fig)
all_years = pd.DataFrame(df.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

all_years = all_years.sort_values('Admissions',ascending=True).set_index('County of Program Location')

all_years = pd.DataFrame(all_years)

all_years['group'] = all_years.index



all_years_color = ['#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B',

                      '#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B',

                     '#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B',

                      '#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B',

                     '#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#53C60B','#2B7000','#2B7000','#2B7000','#2B7000','#2B7000'

                     ]



all_years_data  = go.Data([

            go.Bar(

                y = all_years.group,

                x = all_years['Admissions'],

                text=all_years['Admissions'],

                marker=dict(

                color=all_years_color),

                orientation='h',

                textposition='outside',

                width=.70

        )])

all_years_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=11),

    height = 1000,

    width = 1085,

    margin=go.Margin(l=100),

    title = "Rate of Admissions: 2007-2018",

    titlefont=dict(color='#000000',

                   family='Arial  ',

                   size=15),

    xaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial  ',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Counties",

        titlefont=dict(

            family='Arial  ',

            size=14,

            color='#000000',

        )

    )

)        

all_years_fig  = go.Figure(data=all_years_data, layout=all_years_layout)

all_years_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(all_years_fig)
chor_map = pd.DataFrame(df.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

chor_map = chor_map.sort_values('Admissions',ascending=True).set_index('County of Program Location')

chor_map["Fips"] = chor_map.index

chor_map["Fips"].replace({"New York": "36061", "Queens": "36081", "Bronx":"36005", "Suffolk":"36103","Kings":"36047","Erie":"36029","Westchester":"36119",

                        "Monroe":"36055","Nassau":"36059","Onondaga":"36067","Richmond":"36085","Schenectady":"36093","Albany":"36001","Dutchess":"36027",

                        "Putnam":"36079","Orange":"36071","Oneida":"36065","Niagara":"36063","Rockland":"36087", "Ulster":"36111","Broome":"36007","St Lawrence":"36089",

                        "Ontario": "36069","Rensselaer":"36083","Sullivan":"36105", "Steuben":"36101","Chautauqua":"36013", "Genesee":"36037","Chemung":"36015",

                        "Warren":"36113","Franklin":"36033", "Tompkins":"36109", "Jefferson":"36045","Clinton":"36019","Oswego":"36075","Cattaraugus":"36009",

                        "Wayne":"36117","Saratoga":"36091","Cayuga":"36011","Montgomery":"36057","Delaware":"36025","Seneca":"36099","Cortland":"36023","Livingston":"36051",

                        "Columbia":"36021","Greene":"36039","Fulton":"36035","Orleans":"36073","Chenango":"36017","Allegany":"36003","Herkimer":"36043","Otsego":"36077",

                        "Wyoming":"36121","Washington":"36115","Madison":"36053","Essex":"36031","Schoharie":"36095","Schuyler":"36097","Tioga":"36107","Yates":"36123",

                        "Lewis":"36049"}, inplace=True)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

chor_values = chor_map['Admissions'].tolist()

chor_fips = chor_map['Fips'].tolist()

endpts = list(np.mgrid[min(chor_values):max(chor_values):4j])

colorscale = ["#aecff4","#62a7f4","#064790","#042a54","#0677f4",

              "#4989bc","#60a7c7","#85c5d3","#b7e0e4","#eafcfd"]



chor_map_fig = ff.create_choropleth(

    fips=chor_fips, values=chor_values, scope=['New York'], show_state_data=True,

    colorscale=colorscale, round_legend_values=True, binning_endpoints=endpts, 

    plot_bgcolor='rgb(255,255,255)',

    paper_bgcolor='rgb(255,255,255)',

    legend_title='Rate of Admissions: 2007-2018',

    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},

    exponent_format=True

)

chor_map_fig.layout.template = None

chor_map_fig.show()
new_york_loc = df[df['County of Program Location']== 'New York']

new_york_loc = pd.DataFrame(new_york_loc.groupby(['Year','County of Program Location'])['Admissions'].sum()).reset_index()

new_york_loc = new_york_loc.sort_values('Year',ascending=True).set_index('Year').head(20)

new_york_loc = pd.DataFrame(new_york_loc)

new_york_loc['group'] = new_york_loc.index



bronx_loc = df[df['County of Program Location']== 'Bronx']

bronx_loc = pd.DataFrame(bronx_loc.groupby(['Year','County of Program Location'])['Admissions'].sum()).reset_index()

bronx_loc = bronx_loc.sort_values('Year',ascending=True).set_index('Year').head(20)

bronx_loc = pd.DataFrame(bronx_loc)

bronx_loc['group'] = bronx_loc.index



queens_loc = df[df['County of Program Location']== 'Queens']

queens_loc = pd.DataFrame(queens_loc.groupby(['Year','County of Program Location'])['Admissions'].sum()).reset_index()

queens_loc = queens_loc.sort_values('Year',ascending=True).set_index('Year').head(20)

queens_loc = pd.DataFrame(queens_loc)

queens_loc['group'] = queens_loc.index



kings_loc = df[df['County of Program Location']== 'Kings']

kings_loc = pd.DataFrame(kings_loc.groupby(['Year','County of Program Location'])['Admissions'].sum()).reset_index()

kings_loc = kings_loc.sort_values('Year',ascending=True).set_index('Year').head(20)

kings_loc = pd.DataFrame(kings_loc)

kings_loc['group'] = kings_loc.index



suffolk_loc = df[df['County of Program Location']== 'Suffolk']

suffolk_loc = pd.DataFrame(suffolk_loc.groupby(['Year','County of Program Location'])['Admissions'].sum()).reset_index()

suffolk_loc = suffolk_loc.sort_values('Year',ascending=True).set_index('Year').head(20)

suffolk_loc = pd.DataFrame(suffolk_loc)

suffolk_loc['group'] = suffolk_loc.index





top_five_fig = go.Figure()



top_five_fig.add_trace(go.Bar(x=new_york_loc.group,

                     y=new_york_loc['Admissions'],

                     name='New York',

                     marker=dict(color='rgb(102, 178, 255)'),

                     orientation='v',

                     textposition='outside',

                ))



top_five_fig.add_trace(go.Bar(x=bronx_loc.group,

                     y=bronx_loc['Admissions'],

                     name='Bronx',

                     marker=dict(color='rgb(240, 102, 255)'),

                     orientation='v',

                     textposition='outside',

                ))

top_five_fig.add_trace(go.Bar(x=kings_loc.group,

                     y=kings_loc['Admissions'],

                     name='Kings',

                     marker=dict(color='rgb(127, 0, 255)'),

                     orientation='v',

                     textposition='outside',

                ))

top_five_fig.add_trace(go.Bar(x=queens_loc.group,

                     y=queens_loc['Admissions'],

                     name='Queens',

                     marker=dict(color='rgb(51, 51, 255)'),

                     orientation='v',

                     textposition='outside',

                ))

top_five_fig.add_trace(go.Bar(x=suffolk_loc.group,

                     y=suffolk_loc['Admissions'],

                     name='Suffolk',

                     marker=dict(color='rgb(0, 0, 102)'),

                     orientation='v',

                     textposition='outside',

                ))





top_five_fig.update_layout(

    font=dict(color='#000000',

             family='Arial',

             size=11),

    height = 900,

    width = 1000,

    title = 'Top Five Leaders of Admissions: 2007-2018',

    titlefont=dict(family='Arial', 

                   size=15),

    xaxis=dict(

        title='Years',

        titlefont_size=14,

        tickfont_size=14,

        titlefont=dict(

            family='Arial',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title='Admissions',

        titlefont_size=14,

        tickfont_size=14,

        titlefont=dict(

            family='Arial',

            size=14,

            color='#000000',

        )

    ),

    legend=dict(

        x=1,

        y=1,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(250, 2, 22, 0)',

        borderwidth=10,



    ),

    barmode='group',

    bargap=0.40, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1,# gap between bars of the same location coordinate.

)

top_five_fig.show()
newly_admitted = df[df['Year']>=2018]

newly_admitted = pd.DataFrame(newly_admitted.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

newly_admitted.index = newly_admitted["County of Program Location"]

newly_admitted = newly_admitted.drop(['County of Program Location'],axis=1)



newly_admitted_color = ['#95b9cc','#a6cee3','#b7d7e8','#6e9981','#97c7ac','#FBB537','#abd2bc','#bbdbc9','#fbc35e','#fcd287']

newly_admitted_labels =['New York City','Hudson Valley','Long Island','Western NY','Finger lakes','Capital District','Central New York',

         'Southern Tier', 'North Country', 'Mohawk Valley']

newly_admitted_sizes = [100222,43800,31717,23107,21173,20564,13355,9694,6607,5765]

newly_admitted_explode = (0.0)



newly_admitted_fig = go.Figure(data=[go.Pie(

    labels=newly_admitted_labels, 

    values=newly_admitted_sizes, 

    hole=.3,sort=False)])



newly_admitted_fig.update_traces(marker=dict(colors=newly_admitted_color))

newly_admitted_fig.update_layout(

    font=dict(

        color='#000000',

        family='Arial',

        size=12.5),

    height = 500,

    width = 1200,

    margin=go.Margin(l=100),

    title = 'Percentage of Admits by Subregion: 2018',

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=18)

)

newly_admitted_fig.show()
new_admits = df[df['Year']>=2016]

new_admits = pd.DataFrame(new_admits.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

new_admits.index = new_admits["County of Program Location"]

new_admits = new_admits.drop(['County of Program Location'],axis=1)



new_admits_color = ['#95b9cc','#a6cee3','#b7d7e8','#6e9981','#97c7ac','#FBB537','#abd2bc','#bbdbc9','#fcd287','#fbc35e']

new_admits_labels =['New York City','Hudson Valley','Long Island','Western NY','Finger lakes','Captial District','Central New York',

        'Southern Tier','Mohawk Valley','North Country']

new_admits_sizes = [305057,129722,97855,67192,63296,60583,40078,29051,18813,17749]

explode = (0.0)



new_admits_fig = go.Figure(data=[go.Pie(

    labels=new_admits_labels, values=new_admits_sizes, hole=.3,sort=False)])

new_admits_fig.update_traces(marker=dict(colors=new_admits_color))

new_admits_fig.update_layout(

    font=dict(color='#000000',

             family='Arial',

             size=12.5),

    height = 500,

    width = 1200,

    margin=go.Margin(l=100),

    title = 'Percentage of Admits by Subregion: 2016-2018',

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=18)

)

new_admits_fig.show()
sub_all = pd.DataFrame(df.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

sub_all.index = sub_all["County of Program Location"]

sub_all = sub_all.drop(['County of Program Location'],axis=1)



sub_all_color = ['#95b9cc','#a6cee3','#b7d7e8','#6e9981','#FBB537','#97c7ac','#abd2bc','#bbdbc9','#fcd287','#fbc35e']

sub_all_labels =['New York City','Hudson Valley','Long Island','Western NY','Captial District','Finger lakes','Central New York',

        'Southern Tier','Mohawk Valley','North Country']

sub_all_sizes = [1460058,507182,394681,266946,245825,236563,150023,93414,72656,69685]

explode = (0.0)



sub_all_fig = go.Figure(data=[go.Pie(

    labels=sub_all_labels,values=sub_all_sizes, hole=.3,sort=False)])

sub_all_fig.update_traces(marker=dict(colors=sub_all_color))

sub_all_fig.update_layout(

    font=dict(color='#000000',

             family='Arial',

             size=12.5),

    height = 500,

    width = 1200,

    margin=go.Margin(l=100),

    title = 'Percentage of Admits by Subregion: 2007-2018',

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=18)

)

sub_all_fig.show()
last_sub = df[df['Year']>=2018]

last_sub = pd.DataFrame(last_sub.groupby(['Primary Substance Group'])['Admissions'].sum()).reset_index()

last_sub = last_sub.sort_values('Admissions',ascending=False).set_index('Primary Substance Group')

last_sub = pd.DataFrame(last_sub)

last_sub['group'] = last_sub.index





last_sub_color = ['#FF9100','#ff9914','#ffa227','#ffaa3b','#ffb34e','#ffbb62']



last_sub_data  = go.Data([

            go.Bar(

                y = last_sub['Admissions'],

                x = last_sub.group,

                text=last_sub['Admissions'],

                marker=dict(

                color=last_sub_color),

                orientation='v',

                textposition='outside'

        )])

last_sub_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=12),

    height = 560,

    width = 1000,

    margin=go.Margin(l=100),

    title = "Admissions by Substance: 2018",

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=15),

    xaxis=dict(

        title="Substances",

        titlefont=dict(

            family='Arial',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial ',

            size=14,

            color='#000000',

        )

    )

)        

last_sub_fig  = go.Figure(data=last_sub_data, layout=last_sub_layout)

last_sub_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(last_sub_fig)
new_subs = df[df['Year']>=2016]

new_subs = pd.DataFrame(new_subs.groupby(['Primary Substance Group'])['Admissions'].sum()).reset_index()

new_subs = new_subs.sort_values('Admissions',ascending=False).set_index('Primary Substance Group')

new_subs = pd.DataFrame(new_subs)

new_subs['group'] = new_subs.index





new_subs_color = ['#00b6f3','#08c1ff','#1bc6ff','#2fcbff','#43d0ff','#56d5ff',]



new_subs_data  = go.Data([

            go.Bar(

                y = new_subs['Admissions'],

                x = new_subs.group,

                text=new_subs['Admissions'],

                marker=dict(

                color=new_subs_color),

                orientation='v',

                textposition='outside'

        )])

new_subs_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=12),

    height = 560,

    width = 1000,

    margin=go.Margin(l=100),

    title = "Admissions by Substance: 2016-2018",

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=15),

    xaxis=dict(

        title="Substances",

        titlefont=dict(

            family='Arial ',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial',

            size=14,

            color='#000000',

        )

    )

)        

new_subs_fig  = go.Figure(data=new_subs_data, layout=new_subs_layout)

new_subs_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(new_subs_fig)
prim_subs = df

prim_subs = pd.DataFrame(prim_subs.groupby(['Primary Substance Group'])['Admissions'].sum()).reset_index()

prim_subs = prim_subs.sort_values('Admissions',ascending=False).set_index('Primary Substance Group')

prim_subs = pd.DataFrame(prim_subs)

prim_subs['group'] = prim_subs.index



prim_subs_color = ['#63eb0d','#6df21a','#79f32c','#85f43f','#90f551','#9cf664']

prim_subs_data  = go.Data([

            go.Bar(

                y = prim_subs['Admissions'],

                x = prim_subs.group,

                text=prim_subs['Admissions'],

                marker=dict(

                color=prim_subs_color),

                orientation='v',

                textposition='outside'

        )])

prim_subs_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=12),

    height = 560,

    width = 1000,

    margin=go.Margin(l=100),

    title = "Admissions by Substance: 2007-2018",

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=15),

    xaxis=dict(

        title="Substances",

        titlefont=dict(

            family='Arial',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial',

            size=14,

            color='#000000',

        )

    )

)

prim_subs_fig  = go.Figure(data=prim_subs_data, layout=prim_subs_layout)

prim_subs_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(prim_subs_fig)
new_age = df[df['Year']>= 2018]

new_age = pd.DataFrame(new_age.groupby(['Age Group'])['Admissions'].sum()).reset_index()

new_age_fig = go.Figure()



new_age_fig.add_trace(go.Scatter(

    x=new_age['Age Group'],

    y=new_age['Admissions'],

    name='Rate of Growth',

    line = dict(shape = 'linear'),connectgaps = True))



new_age_fig.add_trace(go.Scatter(

    x=new_age['Age Group'], 

    y=new_age['Admissions'], 

    mode='markers', 

    name='Total Admits', 

    marker=dict(color='orange',line=dict(width=5, color='orange'))))



new_age_fig.layout.update(

    font=dict(color='#000000',

             family='Arial',

             size=13),

    title_text='Total Admissions by Age Group: 2018',

    xaxis_showgrid=False,

    yaxis_showgrid=False, 

    width=800,height=600,

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=15),

    

    xaxis=dict(

        title="Age Groups",

        titlefont=dict(

            family='Arial Bold',

            size=15,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial Bold',

            size=15,

            color='#000000',

        )

    )

)

new_age_fig.layout.paper_bgcolor = 'White'

new_age_fig.show()

recent_age = df[df['Year']>= 2016]

recent_age = pd.DataFrame(recent_age.groupby(['Age Group'])['Admissions'].sum()).reset_index()

recent_age_fig = go.Figure()



recent_age_fig.add_trace(go.Scatter(x=recent_age['Age Group'], y=recent_age['Admissions'], name='Rate of Growth'))

recent_age_fig.add_trace(go.Scatter(x=recent_age['Age Group'], y=recent_age['Admissions'], mode='markers', name='Total Admits', marker=dict(color='darkblue',line=dict(width=5, color='darkblue'))))

recent_age_fig.layout.update(

    font=dict(color='#000000',

             family='Arial',

             size=13),

    title_text='Total Admissions by Age Group: 2016-2018',

    xaxis_showgrid=False,

    yaxis_showgrid=False, 

    width=800,height=600,

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=15),

    

    xaxis=dict(

        title="Age Groups",

        titlefont=dict(

            family='Arial Bold',

            size=15,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial Bold',

            size=15,

            color='#000000',

        )

    )

)

recent_age_fig.layout.paper_bgcolor = 'White'

recent_age_fig.show()
overall_age = pd.DataFrame(df.groupby(['Age Group'])['Admissions'].sum()).reset_index()

overall_age_fig = go.Figure()



overall_age_fig.add_trace(go.Scatter(

    x=overall_age['Age Group'], 

    y=overall_age['Admissions'], 

    name='Rate of Growth'))

overall_age_fig.add_trace(go.Scatter(

    x=overall_age['Age Group'], 

    y=overall_age['Admissions'], 

    mode='markers', name='Total Admits', 

    marker=dict(color='Green',line=dict(width=5, color='Green'))))

overall_age_fig.layout.update(

    title_text='Total Admissions by Age Group: 2007-2018',

    xaxis_showgrid=False,

    yaxis_showgrid=False, 

    width=800,height=600,

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=15),

                   

        xaxis=dict(

        title="Age Groups",

        titlefont=dict(

            family='Arial Bold',

            size=15,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial Bold',

            size=15,

            color='#000000',

        )

    )

)

overall_age_fig.layout.paper_bgcolor = 'White'

overall_age_fig.show()
pro_cat = df

pro_cat = pd.DataFrame(pro_cat.groupby(['Program Category'])['Admissions'].sum()).reset_index()

pro_cat = pro_cat.sort_values('Admissions',ascending=False).set_index('Program Category')

pro_cat = pd.DataFrame(pro_cat)

pro_cat['group'] = pro_cat.index



programs = df[df['Year']>=2016]

programs = pd.DataFrame(programs.groupby(['Program Category'])['Admissions'].sum()).reset_index()

programs = programs.sort_values('Admissions',ascending=False).set_index('Program Category')

programs = pd.DataFrame(programs)

programs['group'] = programs.index



new_prog = df[df['Year']>=2018]

new_prog = pd.DataFrame(new_prog.groupby(['Program Category'])['Admissions'].sum()).reset_index()

new_prog = new_prog.sort_values('Admissions',ascending=False).set_index('Program Category')

new_prog = pd.DataFrame(new_prog)

new_prog['group'] = new_prog.index



program_cat_fig = go.Figure()



program_cat_fig.add_trace(go.Bar(x=pro_cat.group,

                     y=pro_cat['Admissions'],

                     name='2007-2018',

                     marker_color='rgb(83, 198, 11)',

                ))

program_cat_fig.add_trace(go.Bar(x=programs.group,

                y=programs['Admissions'],

                name='2016-2018',

                marker_color='rgb(0, 153, 204)'

                ))

program_cat_fig.add_trace(go.Bar(x=new_prog.group,

                y=new_prog['Admissions'],

                name='2018',

                marker_color='rgb(255, 145, 0)',

                ))





program_cat_fig.update_layout(

    font=dict(color='#000000',

             family='Arial',

             size=12),

    height = 700,

    width = 1000,

    margin=go.Margin(l=100),

    title = 'Admissions by Program Category',

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=19),

    

    xaxis=dict(

        title='Treatment Types',

        titlefont_size=16,

        tickfont_size=16,

        titlefont=dict(

            family='Arial Bold',

            size=18,

            color='#000000',

        )

    ),

    yaxis=dict(

        title='Admissions',

        titlefont_size=16,

        tickfont_size=16,

        titlefont=dict(

            family='Arial Bold',

            size=18,

            color='#000000',

        )

    ),

    legend=dict(

        x=1,

        y=1,

        borderwidth=30,

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.13, # gap between bars of adjacent location coordinates.

    bargroupgap=0 # gap between bars of the same location coordinate.

)

program_cat_fig.show()
x=['New York', 'Bronx', 'Queens', 'Kings', 'Suffolk', 'Nassau', 'Erie','Dutchess','Putnam','Rockland','Orange','Ulster','Sullivan']

op_fig = go.Figure(go.Bar(

    x=x, 

    y=[170501,112055,82309,135588,135107,88627,117896,20891,17054,17181,31908,12733,8664], 

    name='Outpatient'))

op_fig.add_trace(go.Bar(x=x, y=[24759, 21136, 27599, 17693,64351,4840,18765,29158,28833,14847,13515,11429,0], name='Inpatient'))

op_fig.add_trace(go.Bar(x=x, y=[237254, 84777, 140280, 68693, 51242, 29343,39882,21306,24468,10151,13627,11639,9034], name='Crisis'))

op_fig.add_trace(go.Bar(x=x, y=[50246, 35505, 9363, 23649, 2705,2309,4668,816,249,889,1255,536,0], name='Opiod'))

op_fig.add_trace(go.Bar(x=x, y=[38023, 23867, 36255,19192, 14114, 2043,10569,7125,5443,987,2382,5983,5408], name='Residential'))



op_fig.update_layout(

    font=dict(color='#000000',

             family='Arial',

             size=13),

    height = 700,

    width = 900,

    margin=go.Margin(l=100),

    title = 'Admissions by Program Category: 2007-2018',

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=16),

    

    xaxis=dict(

        title='Southern Counties',

        titlefont_size=15,

        tickfont_size=15,

        titlefont=dict(

            family='Arial Bold',

            size=14,

            color='#000000',

        )

    ),

    yaxis=dict(

        title='Admissions',

        titlefont_size=15,

        tickfont_size=15,

        titlefont=dict(

            family='Arial Bold',

            size=14,

            color='#000000',

        )

    ),

    legend=dict(

        x=1,

        y=1,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(250, 2, 22, 0)',

        borderwidth=10,



    ),

    barmode='group',

    bargap=0.20, # gap between bars of adjacent location coordinates.

    bargroupgap=0.3,# gap between bars of the same location coordinate.





)

op_fig.update_layout(barmode='stack',

                     xaxis={'categoryorder':'array', 'categoryarray':['New York','Queens','Bronx','Kings','Suffolk','Erie','Nassau','Dutchess','Putnam','Rockland','Orange','Ulster','Sullivan']})

op_fig.show()
new_type = pd.DataFrame(other_df.groupby(['PROGRAM_NAME','PROGRAM_COUNTY','PROGRAM_TYPE']).sum()).reset_index()

new_type.drop_duplicates(subset="PROGRAM_NAME")
treatment_avail = new_type.groupby('PROGRAM_COUNTY')['PROGRAM_TYPE'].value_counts()

treatment_avail = pd.DataFrame(treatment_avail.groupby('PROGRAM_COUNTY').sum()).reset_index()

treatment_avail = treatment_avail.sort_values('PROGRAM_TYPE',ascending=True)



treatment_avail_color = ['#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                      '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C','#DC143C','#DC143C','#DC143C',

                     '#DC143C','#DC143C','#DC143C','#DC143C','#DC143C','#DC143C','#DC143C','#FF0000',

                      '#FF0000','#FF0000','#FF0000','#FF0000'

                     ]

treatment_avail_data  = go.Data([

            go.Bar(y=treatment_avail["PROGRAM_COUNTY"],

                x = treatment_avail['PROGRAM_TYPE'],

                text=treatment_avail["PROGRAM_TYPE"],

                marker=dict(

                color=treatment_avail_color),

                orientation='h',

                textposition='outside',

                width=.70,



        )])

treatment_avail_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=15),

    height = 1300,

    width =980,

    margin=go.Margin(l=100),

    title = "Sum of Treatment Programs",

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=19),



    xaxis=dict(

        title="Counties",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Total Programs",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    )

)

treatment_avail_fig  = go.Figure(data=treatment_avail_data, layout=treatment_avail_layout)

treatment_avail_fig.update_layout(uniformtext_minsize=3.9, uniformtext_mode='show')

py.iplot(treatment_avail_fig)

new_original = pd.DataFrame(df.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

new_original = new_original.sort_values('Admissions',ascending=True).set_index('County of Program Location')



new_second = pd.DataFrame(treatment_avail.groupby('PROGRAM_COUNTY').sum()).reset_index()

new_second["PROGRAM_COUNTY"].replace({"Saint Lawrence": "St Lawrence"}, inplace=True)

new_second = new_second.sort_values('PROGRAM_TYPE',ascending=True).set_index('PROGRAM_COUNTY')



my_round = (new_second['PROGRAM_TYPE'].map(int)/ new_original['Admissions'].map(int))

my_round.index.name = "Counties"

new_round = my_round * 100000

new_round = new_round.fillna(0)

new_round = round(new_round, 1)

print(new_round.to_markdown())
per_capita = pd.DataFrame(data=new_round)

per_capita.columns = ['Programs Available per County'] 

per_capita = per_capita.sort_values('Programs Available per County',ascending=True)







per_capita_color = ['#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b',

                     '#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b',

                     '#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b',

                     '#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b',

                     '#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b','#c4bb0b',

                     '#7A7407','#7A7407','#7A7407','#7A7407','#7A7407']





per_capita_data  = go.Data([

            go.Bar(

                x = per_capita['Programs Available per County'],

                y = per_capita.index,

                text=per_capita['Programs Available per County'],

                marker=dict(

                color=per_capita_color),

                orientation='h',

                textposition='outside',

                width=.70,

        )])

per_capita_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=13),

    height = 1300,

    width =980,

    margin=go.Margin(l=100),

    title = "Available Treatment Programs per Capita",

    #`

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=19),

    xaxis=dict(

        title="Per 100,000 People",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Counties",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    )

)        

per_capita_fig  = go.Figure(data=per_capita_data, layout=per_capita_layout)

per_capita_fig.update_layout(uniformtext_minsize=3.9, uniformtext_mode='show')

per_capita_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(per_capita_fig)
counties_data = other_df

counties_data = counties_data.groupby('PROGRAM_COUNTY')['PROGRAM_COUNTY'].value_counts(ascending=True)



counties_data_color = ['#415478','#556e9f','#6b83b0','#608892','#fbbf55','#80a4ac','#98b5bc','#fcd287','#b0c6cb','#fddba0']



counties_data_labels =['New York City','Hudson Valley','Long Island','Western NY','Captial District','Finger lakes','Central New York','North Country','Southern Tier','Mohawk Valley']

counties_data_sizes = [321,133,101,78,71,69,38,30,25,24]

explode = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

counties_data_fig = go.Figure(data=[go.Pie(

    labels=counties_data_labels,values=counties_data_sizes, hole=.3,sort=False)])

counties_data_fig.update_traces(marker=dict(colors=counties_data_color))

counties_data_fig.update_layout(

    title_text="Sum of Treatment Programs in New York", width=1000)

counties_data_fig.show()
sub_all = pd.DataFrame(df.groupby(['County of Program Location'])['Admissions'].sum()).reset_index()

sub_all.index = sub_all["County of Program Location"]

sub_all = sub_all.drop(['County of Program Location'],axis=1)



sub_all_color = ['#95b9cc','#a6cee3','#b7d7e8','#6e9981','#FBB537','#97c7ac','#abd2bc','#bbdbc9','#fcd287','#fbc35e']

sub_all_labels =['New York City','Hudson Valley','Long Island','Western NY','Captial District','Finger lakes','Central New York',

        'Southern Tier','Mohawk Valley','North Country']

sub_all_sizes = [1460058,507182,394681,266946,245825,236563,150023,93414,72656,69685]

explode = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)



sub_all_fig = go.Figure(data=[go.Pie(

    labels=sub_all_labels,values=sub_all_sizes, hole=.3,sort=False)])

sub_all_fig.update_traces(marker=dict(colors=sub_all_color))

sub_all_fig.update_layout(

    title_text=" Percentage of Admissions by Subregions: 2007-2018", width=1000)

sub_all_fig.show()

city_count  = new_type['PROGRAM_TYPE'].value_counts()

city_count = city_count[:10]

city_count_color = ['#004789','#ff9100','#9500ff','#53c60b','#fc0505']



city_count_data  = go.Data([

            go.Bar(

                y = city_count.values,

                x = city_count.index,

                text=city_count.values,

                marker=dict(

                color=city_count_color),

                orientation='v',

                textposition='outside'

        )])

city_count_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=18),

    height = 800,

    width = 1000,

    margin=go.Margin(l=100),

    title = "Total Number of Oasas Program Categories",

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=20),

    xaxis=dict(

        title="Program Category",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Total Sum of Programs",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    )

)        

city_count_fig  = go.Figure(data=city_count_data, layout=city_count_layout)

city_count_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(city_count_fig)

comparing_types = pd.DataFrame(df.groupby(['Program Category'])['Admissions'].sum()).reset_index()

comparing_types = comparing_types.sort_values('Admissions',ascending=False).set_index('Program Category')



comparing_types_color = ['#004789','#53c60b','#fc0505','#ff9100','#9500ff','#fc058d']



comparing_types_data  = go.Data([

            go.Bar(

                y = comparing_types['Admissions'],

                x = comparing_types.index,

                text=comparing_types['Admissions'],

                marker=dict(

                color=comparing_types_color),

                orientation='v',

                textposition='outside'

        )])

comparing_types_layout = go.Layout(

    font=dict(color='#000000',

             family='Arial',

             size=18),

    height = 800,

    width = 1000,

    margin=go.Margin(l=100),

    title = "Rate of Admissions by Program Category: 2007-2018",

    titlefont=dict(color='#000000',

                   family='Arial Bold',

                   size=20),

    xaxis=dict(

        title="Program Category",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    ),

    yaxis=dict(

        title="Admissions",

        titlefont=dict(

            family='Arial Bold',

            size=17,

            color='#000000',

        )

    )

)        

comparing_types_fig  = go.Figure(data=comparing_types_data, layout=comparing_types_layout)

comparing_types_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

py.iplot(comparing_types_fig)