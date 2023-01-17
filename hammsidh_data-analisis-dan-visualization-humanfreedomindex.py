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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.express as px



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
df = pd.read_csv('/kaggle/input/the-human-freedom-index/hfi_cc_2018.csv')

df.head()
# Filter Eropa Barat Tahun 2016

fl1 = df.year == 2016

fl2 = df.region == 'Western Europe'

dffl = df[fl1 & fl2]



# Filter Eropa Barat Tahun 2015

fl15 = df.year == 2015

fl25 = df.region == 'Western Europe'

dffl5 = df[fl15 & fl25]
# Money Growth South Asia 2015

tr1 = go.Bar(

                 x = dffl5['countries'],

                 y = dffl5['ef_money_growth'],

                 name = 'Pertumbuhan Uang 2015',

                 marker = dict(color='crimson',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = dffl5.countries)



# Money Growth South Asia 2016

tr2 = go.Bar(

                 x = dffl['countries'],

                 y = dffl['ef_money_growth'],

                 name = 'Pertumbuhan Uang 2016',

                 marker = dict(color='rgba(0, 0, 255, 0.5)',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = dffl.countries)

dn = [tr1, tr2]

layoutnew = go.Layout(barmode='group', title='Perbandingan Pertumbuhan Uang di Eropa Barat Pada Tahun 2015 dan Tahun 2016')

fig = go.Figure(data=dn, layout=layoutnew)

py.iplot(fig)
inv1 = df.year == 2016

inv2 = df.region == 'East Asia'

dfinv = df[inv1 & inv2]



tr1 = go.Bar(

                 x = dfinv['countries'],

                 y = dfinv['ef_government_consumption'],

                 name = 'Konsumsi dari Pemerintah',

                 marker = dict(color='crimson',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = dfinv.countries)



tr2 = go.Bar(

                 x = dfinv['countries'],

                 y = dfinv['ef_government_enterprises'],

                 name = 'Bisnis dan Investasi ke Pemerintah',

                 marker = dict(color='rgba(0, 0, 255, 0.5)',

                              line = dict(color='rgba(0,0,0)', width=0.5)),

                 text = dfinv.countries)

dn = [tr1, tr2]

layoutnew = go.Layout(barmode='group', title='Pertumbuhan Bisnis di Negara Asia Timur')

fig = go.Figure(data=dn, layout=layoutnew)

fig.update_layout(barmode='stack')

py.iplot(fig)
import plotly.express as px



woa1 = df.year == 2016

woa2 = df.region == 'Middle East & North Africa'

dfwoa = df[woa1 & woa2]



# Use column names of df for the different parameters x, y, color, ...

fig = px.scatter(dfwoa, x="countries", y="pf_religion", color="pf_ss_women",

                 title="Keterkaitan Kebebasan Beragama dan Keamanan Wanita ",

                )



fig.show()
didn = df[df['countries'] == 'Indonesia']



fig = px.line(didn, x="year", y="hf_rank", title="Rangking Kebebasan Manusia di Indonesia")

fig.show()
tr1 = go.Scatter(

                     x = dffl5['countries'],

                     y = dffl5['ef_money_inflation'],

                     mode = 'lines+markers',

                     name = 'Inflasi 2015',

                     line_shape='hvh',

                     marker = dict(color = 'crimson'),

                     text= dffl5['countries'])

tr2 = go.Scatter(

                     x = dffl['countries'],

                     y = dffl['ef_money_inflation'],

                     mode = 'lines+markers',

                     name = 'Inflasi 2016',

                     line_shape='hvh',

                     marker = dict(color = 'rgba(20, 150, 20, 0.5)'),

                     text= dffl['countries'])

dn = [tr1, tr2]

layoutnew = dict(title = 'Perkembangan Inflasi dari tahun 2015 ke 2016 pada Negara Bagian Eropa Barat',

                 xaxis = dict(title = 'Countries', ticklen = 3, zeroline = False))

fig = dict(data=dn, layout=layoutnew)

py.iplot(fig)
ssa1 = df.year == 2016

ssa2 = df.region == 'South Asia'

dfssa = df[ssa1 & ssa2]



fig = {

        'data': [ 

             {

                'values' : dfssa['pf_ss'],

                'labels' : dfssa['countries'],

                'domain' : {'x': [0, 1]},

                'name' : 'Kebebasan pada Asia Selatan',

                'hoverinfo' : 'label+percent+name',

                'hole' : 0.3,

                'type' : 'pie'

              },

             ],

         'layout' : {

                     'title' : 'Kemanan dan Kenyamanan di Negara Bagian Middle East & North Africa',

                     'annotations' : [

                                        { 'font' : {'size' : 20},

                                          'showarrow' : False,

                                          'text' : ' ',

                                          'x' : 0.20,

                                          'y' : 1

                                         },

                                      ]    

                     }

        }

py.iplot(fig)
fig = go.Figure()

fig.add_trace(go.Scatter(x=didn['year'], y=didn['pf_religion'], fill='tozeroy',name = 'Kebebasan Beragama')) # fill down to xaxis

fig.add_trace(go.Scatter(x=didn['year'], y=didn['pf_religion_harassment'], fill='tonexty',name = 'Pelecehan Agama')) # fill to trace0 y



fig.show()
tr01 = go.Scatter(

                     x = dfwoa['countries'],

                     y = dfwoa['pf_ss_women'],

                     mode = 'lines+markers',

                     name = 'Kemanan pada Wanita',

                     marker = dict(color = 'crimson'),

                     text= dfwoa['countries'])



tr02 = go.Scatter(

                     x = dfwoa['countries'],

                     y = dfwoa['pf_movement_women'],

                     mode = 'lines+markers',

                     name = 'Kebebasan Bergerak pada Wanita',

                     marker = dict(color = 'rgba(20, 150, 20, 0.5)'),

                     text= dfwoa['countries'])

dn = [tr01, tr02]

layoutnew = dict(title = 'Kemananan dan Kebebasan pada Perbandingan Kebebasan Bergerak di Middle East and Northern Africa',

                 xaxis = dict(title = 'Countries', ticklen = 3, zeroline = False))

fig = dict(data=dn, layout=layoutnew)

py.iplot(fig)
dataocean = df[df.region == 'Oceania']

dataocean = df.filter(['pf_ss_women', 'pf_ss_disappearances_violent', 'pf_ss_women_missing'])

dataocean['index'] = np.arange(1, len(dataocean)+1)



import plotly.figure_factory as ff

fig = ff.create_scatterplotmatrix(dataocean, diag='box', index = 'index', colormap='Portland', colormap_type='cat', height=700, width=700)

py.iplot(fig)
asia1 = df.year == 2016

asia2 = df.region == 'East Asia'

asia3 = df.region == 'South Asia'

asia4 = df.region == 'Caucasus & Central Asia'



dfasia = df[(asia2 | asia3 | asia4) & asia1]



fig = px.scatter(dfasia, x="hf_rank", y="ef_money",

                 size="hf_rank", color="region",

                 hover_name="countries", log_x=True, size_max=60)

fig.show()
dataset = df

dataset = dataset.loc[:, ['year', 'countries', 'pf_expression']]

dataset.tail()

ssw = [dict(

    type = 'choropleth',

    locations = dataset['countries'],

    locationmode = 'country names',

    z = dataset['pf_expression'],

    text = dataset['countries'],

    colorscale = [[0,"rgb(5, 10, 172)"],[2,"rgb(40, 60, 190)"],[4,"rgb(70, 100, 245)"],\

            [6,"rgb(90, 120, 245)"],[8,"rgb(106, 137, 247)"],[10,"rgb(220, 220, 220)"]],

    autocolorscale = False,

    reversescale = True,

    marker = dict(line = dict(color = 'rgb(150,150,150)',width = 0.5 )),

    colorbar = dict(autotick=False, tickprefix= '', title='Ranking Kebebasan Berekspresi'),

)]



layout = dict(

    title = 'Ranking Kebebasan Berekspresi di seluruh Dunia',

    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'), mapbox_center = {"lat": 37.0902, "lon": -95.7129})

)



fig = dict(data=ssw, layout=layout)

py.iplot( fig, validate=False, filename='expression-freedom')