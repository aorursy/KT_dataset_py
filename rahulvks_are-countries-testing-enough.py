# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# offline plotly
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# interactive plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# color pallette
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow
from bokeh.layouts import column, row
from bokeh.models import Panel, Tabs, LinearAxis, Range1d, BoxAnnotation, LabelSet, Span
from bokeh.models.tools import HoverTool
from bokeh.palettes import Category20, Spectral3, Spectral4, Spectral8
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.transform import dodge
from datetime import datetime as dt
from math import pi
output_notebook()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import gc
gc.collect()
#COVID-19 OWID testing data
#path = "/kaggle/input/covid19-owid-data/"
path = "/kaggle/input/worldmeter/"
#Worldmeter
path2 = "/kaggle/input/worldmeter/"
#Covide from worldmeter
df = pd.read_csv(path2 + "20200906.csv",index_col=0)
remove_row = ['World','North America','Europe','Asia','South America','Africa']
df = df[~df['Country'].isin(remove_row)]
df = df.sort_values(by = 'Cases',ascending=False).reset_index(drop=True)
df = df.drop(columns='#')
df.fillna(0,inplace=True)

remove_row = ['World','North America','Europe','Asia','South America','Africa']

df_test_date = pd.read_csv(path + 'owid-covid-data.csv')
df_test_date = df_test_date[~df_test_date['location'].isin(remove_row)]
df_test_date['date'] = pd.to_datetime(df_test_date['date'])

df.columns
df['Is_UAE'] = df['Country']=='UAE'
df['Is_India'] = df['Country']=='India'
df['Testing_Per1000'] = df['Tests'] / 1000
df['Cases_Per1000'] = df['Cases'] / 1000
df['Death_Per1000'] = df['NCases'] / 1000
df['NewCases_Per1000'] = df['Deaths'] / 1000
df['Population_Per1000'] = df['Population'] / 1000


df_temp = df.copy()


def plot_topn(col, n , xa , xy, title):
    df_f = df_temp.sort_values(col, ascending=False).head(n)
    fig = px.bar(df_f, x=col, y='Country', text=col, 
                 orientation='h',hover_data=["Active","CPM","TPM","Population"],color='Is_India'
                ,color_discrete_sequence=['#FFA500', '#393e46'],opacity=0.8,)
    fig.update_layout(title=title, xaxis_title=xa, yaxis_title=xy, 
                      yaxis_categoryorder = 'total ascending',
                      showlegend=False,template = 'plotly_white')
    fig.show()

        
    
def plot_lown(col, n , xa , xy, title):
    df_f = df_temp.sort_values(col, ascending=False).tail(n)
    fig = px.bar(df_f, x=col, y='Country', text=col, 
                 orientation='h',hover_data=["Active","CPM","TPM","Population"],color='Is_India'
                ,color_discrete_sequence=['#FFA500', '#393e46'],opacity=0.8,)
    fig.update_layout(title=title, xaxis_title=xa, yaxis_title=xy, 
                      yaxis_categoryorder = 'total ascending',
                      showlegend=False,template = 'plotly_white')
    fig.show()

def plot_1000(col, n , xa , xy, title):
    df_f = df_temp.sort_values(col, ascending=False).tail(n)
    fig = px.bar(df_f, x=col, y='Country', text=col, 
                 orientation='h',hover_data=["Active","CPM","TPM","Population"],color='Is_India'
                ,color_discrete_sequence=['#FFA500', '#393e46'],opacity=0.8,)
    fig.update_layout(title=title, xaxis_title=xa, yaxis_title=xy, 
                      yaxis_categoryorder = 'total ascending',
                      showlegend=False,template = 'plotly_white')
    fig.show()
    
    
    
    
#2    
total_test = df_test_date.copy()
total_test['date'] = total_test['date'].apply(pd.to_datetime)
total_test.set_index(["location"], inplace = True)
total_test = total_test.loc[['United States', 'Russia','United Kingdom', 'India','Australia','United Arab Emirates',"Israel","Brazil"
                            ,"Saudi Arabia"]]
total_test.reset_index(inplace = True)
total_test.sort_values('date', ascending= True,inplace=True)   

# plot - test
fig2 = px.scatter(total_test, 
                 x='date', 
                 y='total_tests', 
                 color='location',template= 'plotly_white',log_y=False,title="Testing Rate In Countries"
            )


# plot - test
fig21 = px.scatter(total_test, 
                 x='date', 
                 y='total_tests', 
                 color='location',template= 'plotly_white',log_y=True,title="Testing Rate In Countries"
            )



# plot - cases
fig22 = px.scatter(total_test, 
                 x='date', 
                 y='new_cases', 
                 color='location',template= 'plotly_white',log_y=True,title="New Cases In Countries"
            )

# plot - deaths
fig23 = px.scatter(total_test, 
                 x='date', 
                 y='total_deaths', 
                 color='location',template= 'plotly_white',log_y=True,title="Number of Deaths In Countries"
            )



#3
#fig3 = px.scatter(df, x='Tests', y='Cases', color='Country',size='Population',
#                 hover_name="Country",hover_data=['Country','CPM','TPM','Population'],
#           log_x=True, log_y=True, title='Total Test vs Total Cases, Size - Population',,
#           color_continuous_scale=px.colors.sequential.Plasma,size_max = 30,height =600,template = 'simple_white'
#      
#           )
##4
fig4 = px.scatter(df, x='Testing_Per1000', y='Cases_Per1000', color='NewCases_Per1000',size='Active',
                 hover_name="Country",hover_data=['Country','CPM','TPM','Population'],
           log_x=True, log_y=True, title='Total Test Vs Cases (Per 1000),Size Active / Colur New Cases',
           color_continuous_scale=px.colors.sequential.Plasma,size_max = 30,height =600,template = 'simple_white',opacity=1
      
           )



#4.1 Popultion Per Million
df['Population'] = df['Population'] / 1000000

fig41 = px.scatter(df, x='TPM', y='CPM', color='Population',size='Population',
                 hover_name="Country",hover_data=['Country','CPM','TPM','Population'],
           log_x=True, log_y=True, title='Total Test (Per Million) Vs Total Case (Per Million), Size Population',
           color_continuous_scale=px.colors.sequential.Plasma,size_max = 30,height =600,template = 'simple_white'
      ,opacity=1
           )


#5 

fig5 = px.choropleth(df, locations=df['Country'],
                    color=df['CPM'],locationmode='country names', 
                    hover_name=df['Country'],hover_data=['Cases','Tests','TPM','Population'], 
                    color_continuous_scale=px.colors.diverging.BrBG,template = 'simple_white',
                    range_color=[1,6000])

#6 
fig6 = px.choropleth(df, locations=df['Country'],
                    color=df['CPM'],locationmode='country names', 
                    hover_name=df['Country'],hover_data=['Cases','Tests','TPM','Population'], 
                    color_continuous_scale=px.colors.diverging.BrBG,template='simple_white',range_color=[1,6000]
            )

plot_topn('Tests', 15,'Total Test','Country',"Total Test Taken Per Country")
fig2.update_traces(marker=dict(size=3.5),
                  mode='lines+markers',)
fig2.add_annotation( # add a text callout with arrow
    text="Testing start very late <br>and slow in UK and India", x='2020-03-17', y=100000, arrowhead=4, ax=0,
            ay=-70,showarrow=True)
fig2.show()
##
fig21=fig21.update_traces(marker=dict(size=3.5),
                  mode='lines+markers',)
fig21.show()
##
fig22=fig22.update_traces(marker=dict(size=3.5),
                  mode='lines+markers',)
fig22.show()
##
fig23=fig23.update_traces(marker=dict(size=3.5),
                  mode='lines+markers',)
fig23.show()
plot_topn('TPM', 20,'Test Per 1 Million','Country',"Total Test Per Million vs Country")
fig41.update_coloraxes(colorscale=px.colors.sequential.Cividis_r)
fig41.update(layout_coloraxis_showscale=True)
fig41.show()

fig4.update_coloraxes(colorscale=px.colors.sequential.Cividis_r)
fig4.update(layout_coloraxis_showscale=True)
fig4.show()
fig5.update_layout(    title='Covid-19 Total Test (Per Milion People)',
    template='plotly_white')

fig5.show()

fig6.update_layout(    title='Covid-19 Total Cases (Per Million People)',
    template='plotly_white',
)
fig6.show()
df = df[df['Cases'] > 250000]

source_3 = ColumnDataSource(data = dict(
    state = df.Country.values,
    people_per_lab = df.CPM.values, #CASES
    area_per_lab = df.TPM.values #TEST
))

tooltips_3 = [
    ("Country", "@state"),
    ("Cases", "@people_per_lab{0.00} M"),
    ("Test", "@area_per_lab{0.00} K")
]

h_mid = max(df.CPM.values /100)/2
v_mid = max(df.TPM.values /100)/2

print(h_mid)
print(v_mid)

source_labels = ColumnDataSource(data = dict(
    
state = df[(df.CPM >= v_mid ) | (df.TPM >= h_mid  )].Country.values,
    
people_per_lab = df[(df.CPM >= v_mid ) | (df.TPM >= h_mid )].CPM.values,
    
area_per_lab = df[(df.TPM >= v_mid ) | (df.CPM >= h_mid )].TPM.values 
    
))

labels = LabelSet(x = "people_per_lab", y = "area_per_lab", text = "state", 
        source = source_labels, level = "glyph", x_offset = -19, y_offset = -23, render_mode = "canvas")

v3 = figure(plot_width = 800, plot_height = 800, tooltips = tooltips_3, title = "Country")
v3.circle("people_per_lab", "area_per_lab", source = source_3, size = 13, color = "blue", alpha = 0.41)

tl_box = BoxAnnotation(right = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "orange")
tr_box = BoxAnnotation(left = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "red")
bl_box = BoxAnnotation(right = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "green")
br_box = BoxAnnotation(left = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "orange")

v3.add_layout(tl_box)
v3.add_layout(tr_box)
v3.add_layout(bl_box)
v3.add_layout(br_box)

v3.add_layout(labels)

v3.xaxis.axis_label = "Total Cases (Per Million)"
v3.yaxis.axis_label = "Total Test (Per Million)" 


## Popultion
source_3 = ColumnDataSource(data = dict(
    state = df.Country.values,
    people_per_lab = df.Population.values, #CASES
    area_per_lab = df.TPM.values #TEST
))

tooltips_3 = [
    ("Country", "@state"),
    ("Cases", "@people_per_lab{0.00} M"),
    ("Test", "@area_per_lab{0.00} K")
]

h_mid = max(df.CPM.values /100)/2
v_mid = max(df.TPM.values /100)/2

print(h_mid)
print(v_mid)

source_labels = ColumnDataSource(data = dict(
    
state = df[(df.Population >= v_mid ) | (df.TPM >= h_mid  )].Country.values,
    
people_per_lab = df[(df.Population >= v_mid ) | (df.TPM >= h_mid )].Population.values,
    
area_per_lab = df[(df.TPM >= v_mid ) | (df.Population >= h_mid )].TPM.values 
    
))

labels = LabelSet(x = "people_per_lab", y = "area_per_lab", text = "state", 
        source = source_labels, level = "glyph", x_offset = -19, y_offset = -23, render_mode = "canvas")

v4 = figure(plot_width = 800, plot_height = 800, tooltips = tooltips_3, title = "Country")
v4.circle("people_per_lab", "area_per_lab", source = source_3, size = 13, color = "blue", alpha = 0.41)

tl_box = BoxAnnotation(right = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "orange")
tr_box = BoxAnnotation(left = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "red")
bl_box = BoxAnnotation(right = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "green")
br_box = BoxAnnotation(left = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "orange")

v4.add_layout(tl_box)
v4.add_layout(tr_box)
v4.add_layout(bl_box)
v4.add_layout(br_box)

v4.add_layout(labels)

v4.xaxis.axis_label = "Total Population (Per Million) "
v4.yaxis.axis_label = "Total Test (Per Million) " 
show(column(row(v3)))
show(column(row(v4)))
#bokeh plot - https://www.kaggle.com/rohanrao