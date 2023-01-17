!pip install pygal

!pip install pygal_maps_fr

!pip install pygal_maps_world

!pip install lmfit
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.graph_objs import *

from lmfit.model import Model

px.defaults.height = 400

ddir ="/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-cases.csv"

df =  pd.read_csv(ddir)

col = ['travel_history_country','locally_acquired','additional_info',

       'additional_source','additional_source','case_source','method_note']

ddir1 ="/kaggle/input/uncover/UNCOVER/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv"

ddir2 ="/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-mortality.csv"

ddir3 ="/kaggle/input/uncover/UNCOVER/github/covid19-epidemic-french-national-data.csv"

df3 =  pd.read_csv(ddir3)

df2 =  pd.read_csv(ddir2)

df1 =  pd.read_csv(ddir1)



df.drop(col,axis=1,inplace=True)

df.head()
layout = Layout(

    paper_bgcolor='black',

    plot_bgcolor='black'

)

sex_h = df[df['sex']=='Male'] 

sex_f = df[df['sex']=='Female'] 

fig = go.Figure(layout=layout)

fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),

                     y=sex_h['age'].value_counts(),name="Male",marker_color='yellow',

                    opacity=0.5)

                      )

fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),

                     y=sex_f['age'].value_counts(),name='Female',marker_color='lightgreen',

                    opacity=0.5)

                      )

fig.update_layout(barmode='relative', title_text='comparing ages of confirmed cases based on sex')

fig.show()
fig = go.Figure(data=[go.Pie(labels=df['age'].value_counts().keys().drop("Not Reported"),

                             values=df['age'].value_counts().drop("Not Reported"),

                             title='Ages of confirmed cases',

                             hole=.3,pull=[0.2])],layout=layout)

fig.show()
import plotly.express as px

px.defaults.color_continuous_scale = px.colors.plotlyjs

px.defaults.template = "ggplot2"

fig = px.scatter(df1,x='totaltestresults', 

                 y='positive',

                 color="state",size='total',marginal_x='histogram'

)

fig.show()
df_ny = df1[df1['state']=='NY']

x = []

for i in range(len(df_ny.index)):

    x.append(i)

y = df_ny['positive']

def exp_func(x,a,b):

    return a*np.exp(b*x)

exponmodel = Model(exp_func)

params = exponmodel.make_params(a=5, b=0.01)

result = exponmodel.fit(y, params, x=x)

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_ny['date'], y=df_ny['positive'],line_color='rgb(231,107,243)',

    name='Positive cases',fill='tonexty'))

fig.add_trace(go.Scatter(x=df_ny['date'], y=result.best_fit,line_color='yellow',opacity=0.1,

    name='Positive exponential',fill='tozeroy'))

fig.update_layout(

Layout(

    paper_bgcolor='black',

    plot_bgcolor='black'),title_text='Fitting exponential curve to  positive cases ')



fig.update_traces(mode='lines')

fig.show()
fig = px.pie(values=df2['age'].value_counts().drop("Not Reported"), 

             names=df2['age'].value_counts().keys().drop("Not Reported"),

             title='',hole=.3)

fig.update_layout(

Layout(

    paper_bgcolor='black',

    plot_bgcolor='black'),title_text='Ages of  death cases ')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
layout = Layout(

    paper_bgcolor='black',

    plot_bgcolor='black'

)

sex_h = df[df['sex']=='Male'] 

sex_f = df[df['sex']=='Female'] 

fig = go.Figure(layout=layout)

fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),

                     y=sex_h['age'].value_counts(),name="Male",marker_color='purple',

                    opacity=0.5)

                      )

fig.add_trace(go.Bar(x=df['age'].value_counts().keys(),

                     y=sex_f['age'].value_counts(),name='Female',marker_color='orange',

                    opacity=0.5)

                      )

fig.update_layout(barmode='relative', title_text='comparing ages of death cases based on sex')

fig.show()
dffr = df3[df3['date']=='2020-03-30']

dffr[dffr['maille_code']=='FRA']

dffr = dffr[:101]

values = []

for v in dffr['deces']:

    values.append(v)

keys = []  

for k in dffr['maille_code']:

    keys.append(k.split('-')[-1])

res = {keys[i]: int(values[i]) for i in range(len(keys))}   

from IPython.display import HTML

import pygal

html_pygal = u"""

    <!DOCTYPE html>

    <html>

        <head>

            <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>

            <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/pygal-tooltips.js"></script>

        </head>

        <body><figure>{pygal_render}</figure></body>

    </html>

"""

from pygal.maps.fr import aggregate_regions

fr_chart = pygal.maps.fr.Departments(human_readable=True)

fr_chart.title = 'Covid-19 mortality by French department'

fr_chart.add('In 2020-03-30',res)

#fr_chart.render()

HTML(html_pygal.format(pygal_render=fr_chart.render(is_unicode=True)))