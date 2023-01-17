

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

suicide=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

suicide.head()
from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.graph_objs as go

from ipywidgets import widgets

from IPython.display import display, clear_output, Image



init_notebook_mode(connected=True)



target1 =suicide.loc[(suicide['sex']=='female') & (suicide['age']=='15-24 years') & (suicide['country']=='United States')].copy()

line1 = go.Scatter(x = target1.year, y = target1.suicides_no, mode = "lines",name = "15-24 years", 

                marker = { "color" : '#EFC7FB'}, text= target1.year)



target2 =suicide.loc[(suicide['sex']=='female') & (suicide['age']=='5-14 years') & (suicide['country']=='United States')].copy()

line2 = go.Scatter(x = target2.year, y = target2.suicides_no, mode = "lines",name = "5-14 years", 

                marker = { "color" : '#6A83DE'}, text= target2.year)



target3 =suicide.loc[(suicide['sex']=='female') & (suicide['age']=='35-54 years') & (suicide['country']=='United States')].copy()

line3 = go.Scatter(x = target3.year, y = target3.suicides_no, mode = "lines",name = "35-54 years", 

                marker = { "color" : '#ABDE6A'}, text= target3.year)



target4 =suicide.loc[(suicide['sex']=='female') & (suicide['age']=='55-74 years') & (suicide['country']=='United States')].copy()

line4 = go.Scatter(x = target4.year, y = target4.suicides_no, mode = "lines",name = "55-74 years", 

                marker = { "color" : '#DE6AA0'}, text= target4.year)



target5 =suicide.loc[(suicide['sex']=='female') & (suicide['age']=='75+ years') & (suicide['country']=='United States')].copy()

line5 = go.Scatter(x = target5.year, y = target5.suicides_no, mode = "lines",name = "75+ years", 

                marker = { "color" : '#DEBB6A'}, text= target5.year)

                    



country = widgets.Dropdown(

    options=list(suicide['country'].unique()),

    value='United States',

    description='country',

)



sex = widgets.Dropdown(

    options=list(suicide['sex'].unique()),

    value='female',

    description='gender',

)







f = go.FigureWidget(data=[line1, line2, line3, line4, line5],

                    layout=go.Layout(

                        title=dict(

                            text='Suicide rate for US females'

                        ),xaxis= dict(title= 'year',ticklen= 5,zeroline= False)

                    ))





def response(change):

    temp_df = suicide.loc[(suicide['sex']==sex.value) & (suicide['country']==country.value)].copy()

   

    x1 = temp_df.loc[temp_df['age']=='15-24 years']['year']

    y1 = temp_df.loc[temp_df['age']=='15-24 years']['suicides_no']

    x2 = temp_df.loc[temp_df['age']=='5-14 years']['year']

    y2 = temp_df.loc[temp_df['age']=='5-14 years']['suicides_no']

    x3 = temp_df.loc[temp_df['age']=='35-54 years']['year']

    y3 = temp_df.loc[temp_df['age']=='35-54 years']['suicides_no']

    x4 = temp_df.loc[temp_df['age']=='55-74 years']['year']

    y4 = temp_df.loc[temp_df['age']=='55-74 years']['suicides_no']

    x5 = temp_df.loc[temp_df['age']=='75+ years']['year']

    y5 = temp_df.loc[temp_df['age']=='75+ years']['suicides_no']

    

    with f.batch_update():

        f.data[0].x = x1

        f.data[0].y = y1

        f.data[1].x = x2

        f.data[1].y = y2

        f.data[2].x = x3

        f.data[2].y = y3

        f.data[3].x = x4

        f.data[3].y = y4

        f.data[4].x = x5

        f.data[4].y = y5

        f.layout.title='Suicide rates for '+country.value+" "+sex.value





country.observe(response, names="value")

sex.observe(response, names="value")

widgets.VBox([country, sex, f])
