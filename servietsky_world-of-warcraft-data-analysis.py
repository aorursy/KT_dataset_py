# import som Libraries



import pandas as pd

import pandas_profiling as pp

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from datetime import timedelta

import tqdm

import numpy as np



!pip install pyvis



from pyvis.network import Network

import pyvis





import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
# Read Data and make som updates



location_coords = pd.read_csv('../input/warcraft-avatar-history/location_coords.csv',  encoding='windows-1252')

locations = pd.read_csv('../input/warcraft-avatar-history/locations.csv',  encoding='windows-1252')

wowah_data = pd.read_csv('../input/warcraft-avatar-history/wowah_data.csv',  encoding='windows-1252')

zones = pd.read_csv('../input/warcraft-avatar-history/zones.csv',  encoding='windows-1252')



location_coords.columns = location_coords.columns.str.replace(' ', '')

locations.columns = locations.columns.str.replace(' ', '')

wowah_data.columns = wowah_data.columns.str.replace(' ', '')

zones.columns = zones.columns.str.replace(' ', '')



wowah_data['timestamp'] = pd.to_datetime(wowah_data['timestamp'])

wowah_data['dates'] = wowah_data['timestamp'].dt.date

wowah_data.loc[wowah_data['timestamp'] >= '2008-11-18','extention'] = 'WOTLK'

wowah_data.loc[wowah_data['timestamp'] < '2008-11-18', 'extention'] = 'BC'



dict_color = {'Death Knight': '#C41F3B',

                'Shaman': '#0070DE',

                'Druid': '#FF7D0A',

                'Rogue': '#FFF569',

                'Priest': '#FFFFFF',

                'Paladin': '#F58CBA',

                'Warrior': '#C79C6E',

                'Warlock': '#8787ED',

                'Mage': '#40C7EB',

                'Hunter': '#A9D271'}



wowah_data['Class_color']  = wowah_data.charclass.map(dict_color)



wowah_data['Date'] =  pd.to_datetime(wowah_data['timestamp'], format='%Y-%m-%d')

wowah_data["Day_of_Week"] = wowah_data.Date.dt.weekday

wowah_data["First_day_of_the_week"] = wowah_data.Date - wowah_data.Day_of_Week * timedelta(days=1)

wowah_data.drop(['Day_of_Week', 'Date'], axis = 1, inplace = True)

wowah_data["First_day_of_the_week"] = wowah_data["First_day_of_the_week"].dt.date



col = {}

import random

for i in wowah_data.zone.unique() :

    color = "%06x" % random.randint(0, 0xFFFFFF)

    col[i] = '#' + color 

wowah_data['color_zone'] = wowah_data['zone'].map(col)
tmp = pd.DataFrame(wowah_data.groupby(['race']).count()['char']).reset_index().sort_values(by = 'char')

tmp['char_%'] = tmp['char'].div(tmp.char.sum())*100

tmp.drop(['char'], axis = 1, inplace = True)

Dataplot = pd.DataFrame(wowah_data.groupby(['race', 'charclass']).count()['char']).reset_index().sort_values(by = 'char')



fig = px.bar(tmp.merge(Dataplot, left_on = 'race', right_on = 'race' ), x='race', y='char',

             hover_data=['race', 'char'], color = 'charclass',  color_discrete_map =  dict_color,

             labels={'char':'Character Race Population', 'charclass': 'Character Class', 'rece' : 'Character Race'}, height=500)



fig.add_trace(go.Scatter(

    x=['Orc', 'Troll', 'Tauren', 'Undead', 'Blood Elf'],

    y=[1000000, 1200000, 2500000, 2700000, 4000000],

    text=pd.DataFrame(tmp['char_%'].round(1).astype(str) + '%')['char_%'].values,

    mode="text",

))



fig.show()



del Dataplot, tmp
Dataplot = pd.DataFrame(wowah_data.groupby('charclass').count()['char']).reset_index().sort_values(by = 'char')

Dataplot['pers'] = Dataplot['char'].div(108267.34).round(1).astype(str) + '%'



fig = px.bar(Dataplot, x='charclass', y='char', color = 'charclass',

             hover_data=['charclass', 'char'], 

             color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class', 'pers' : 'Percentage'}, height=400, text = 'pers')



datatmp1 = wowah_data[wowah_data['timestamp'] < '2008-11-18'].groupby([pd.Grouper(key='timestamp', freq='19d'), 'charclass']).count().reindex().groupby('charclass').mean()

datatmp1['pers'] = datatmp1['char'].div(datatmp1.char.sum()/100).round(1).astype(str) + '%'

datatmp2 = wowah_data[wowah_data['timestamp'] >= '2008-11-18'].groupby([pd.Grouper(key='timestamp', freq='30d'), 'charclass']).count().reindex().groupby('charclass').mean()

datatmp2['pers'] = datatmp2['char'].div(datatmp2.char.sum()/100).round(1).astype(str) + '%'



datatmp1['ext'] = 'BC'

datatmp2['ext'] = 'WOTLK'

Dataplot = pd.concat([datatmp1,datatmp2]).reset_index()



fig2 = px.bar(Dataplot, x='charclass', y='char', color = 'charclass',

             hover_data=['charclass', 'char'], 

             color_discrete_map =  dict_color, facet_row='ext',

             labels={'char':'Class Population', 'charclass' : 'Class', 'ext' : 'Extention', 'pers' : 'Percentage'}, height=400, text = 'pers')

fig.show()

fig2.show()



del Dataplot, datatmp1, datatmp2



Dataplot = pd.DataFrame(wowah_data.groupby(['dates', 'charclass']).count()['char']).reset_index().sort_values(by = 'dates')

Dataplot2 = pd.DataFrame(wowah_data.groupby(['dates']).count()['char']).reset_index().sort_values(by = 'dates')

Dataplot = Dataplot.merge(Dataplot2, left_on = 'dates', right_on = 'dates')

Dataplot['pers'] = Dataplot['char_x'].div(Dataplot.char_y.values /100).round(1)



fig = px.line(Dataplot, x="dates", y="pers", color='charclass',

             hover_data=['charclass', 'pers'], 

             color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class', 'pers' : 'Class Population Percentage'}, height=400)



fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2008-11-18',

            y0=0,

            x1='2008-11-18',

            y1=40,

            line=dict(

                color="Black",

                width=2

            )))

fig.add_trace(go.Scatter(

    x=['2008-9-30', '2008-12-30'],

    y=[25, 25],

    text=["Burning Crusade",

          "WOTLK",],

    mode="text",

))



Dataplot = pd.DataFrame(wowah_data.groupby(['dates', 'charclass']).count()['char']).reset_index().sort_values(by = 'dates')



fig2 = make_subplots(rows=3, cols=1)



fig2 = px.line(Dataplot, x="dates", y="char", color='charclass',

             hover_data=['charclass', 'char'], 

             color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class'}, height=400)



fig2.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2008-11-18',

            y0=0,

            x1='2008-11-18',

            y1=10000,

            line=dict(

                color="Black",

                width=2

            )))

fig2.add_trace(go.Scatter(

    x=['2008-9-30', '2008-12-30'],

    y=[8000, 7000],

    text=["Burning Crusade",

          "WOTLK",],

    mode="text",

))



fig.show()

fig2.show()

del Dataplot, Dataplot2
Dataplot = pd.DataFrame(wowah_data.groupby(['dates', 'race']).count()['char']).reset_index().sort_values(by = 'dates')

Dataplot2 = pd.DataFrame(wowah_data.groupby(['dates']).count()['char']).reset_index().sort_values(by = 'dates')

Dataplot = Dataplot.merge(Dataplot2, left_on = 'dates', right_on = 'dates')

Dataplot['pers'] = Dataplot['char_x'].div(Dataplot.char_y.values /100).round(1)



fig = px.line(Dataplot, x="dates", y="pers", color='race',

             hover_data=['race', 'pers'], 

             color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class', 'pers' : 'Race Population Percentage', 'race' : 'Race'}, height=400)



fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2008-11-18',

            y0=0,

            x1='2008-11-18',

            y1=50,

            line=dict(

                color="Black",

                width=2

            )))

fig.add_trace(go.Scatter(

    x=['2008-9-30', '2008-12-30'],

    y=[30, 30],

    text=["Burning Crusade",

          "WOTLK",],

    mode="text",

))



Dataplot = pd.DataFrame(wowah_data.groupby(['dates', 'race']).count()['char']).reset_index().sort_values(by = 'dates')



fig2 = make_subplots(rows=3, cols=1)



fig2 = px.line(Dataplot, x="dates", y="char", color='race',

             hover_data=['race', 'char'], 

             color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class', 'pers' : 'Race Population Percentage', 'race' : 'Race'}, height=400)



fig2.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2008-11-18',

            y0=0,

            x1='2008-11-18',

            y1=25000,

            line=dict(

                color="Black",

                width=2

            )))

fig2.add_trace(go.Scatter(

    x=['2008-9-30', '2008-12-30'],

    y=[18000, 15000],

    text=["Burning Crusade",

          "WOTLK",],

    mode="text",

))



fig.show()

fig2.show()

del Dataplot, Dataplot2
Dataplot = wowah_data.copy()

Dataplot['lvl_inter'] = pd.cut(Dataplot['level'], 8)

dict_ = {'(0.921, 10.875]' : '[1-10]', '(10.875, 20.75]' : '[10-20]', '(50.375, 60.25]' : '[50-60]', '(60.25, 70.125]' : '[60-70]', '(40.5, 50.375]' : '[40-50]', '(20.75, 30.625]' : '[20-30]', '(30.625, 40.5]' : '[30-40]', '(70.125, 80.0]' : '[70-80]'}

Dataplot['lvl_inter2'] = Dataplot['lvl_inter'].astype(str).map(dict_)



Dataplot = pd.DataFrame(Dataplot.groupby(['dates', 'lvl_inter2']).count()['char']).reset_index().sort_values(by = 'dates')

Dataplot2 = pd.DataFrame(Dataplot.groupby(['dates']).sum()['char']).reset_index().sort_values(by = 'dates')

Dataplot = Dataplot.merge(Dataplot2, left_on = 'dates', right_on = 'dates')

Dataplot['pers'] = Dataplot['char_x'].div(Dataplot.char_y.values /100).round(1)



fig = px.line(Dataplot, x="dates", y="pers", color='lvl_inter2',

             hover_data=['lvl_inter2', 'pers'], 

             color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class', 'lvl_inter2' : 'Level', 'pers' : 'Level Population Percentage', 'dates' : 'Date'}, height=400)



fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2008-11-18',

            y0=0,

            x1='2008-11-18',

            y1=90,

            line=dict(

                color="Black",

                width=2

            )))

fig.add_trace(go.Scatter(

    x=['2008-5-30', '2008-12-30'],

    y=[40, 40],

    text=["Burning Crusade",

          "WOTLK",],

    mode="text",

))



Dataplot = wowah_data.copy()

Dataplot['lvl_inter'] = pd.cut(Dataplot['level'], 8)

dict_ = {'(0.921, 10.875]' : '[1-10]', '(10.875, 20.75]' : '[10-20]', '(50.375, 60.25]' : '[50-60]', '(60.25, 70.125]' : '[60-70]', '(40.5, 50.375]' : '[40-50]', '(20.75, 30.625]' : '[20-30]', '(30.625, 40.5]' : '[30-40]', '(70.125, 80.0]' : '[70-80]'}

Dataplot['lvl_inter2'] = Dataplot['lvl_inter'].astype(str).map(dict_)



Dataplot = pd.DataFrame(Dataplot.groupby(['dates', 'lvl_inter2']).count()['char']).reset_index().sort_values(by = 'dates')

fig2 = px.line(Dataplot, x="dates", y="char", color='lvl_inter2',

             hover_data=['lvl_inter2', 'char'], 

             color_discrete_map =  dict_color,

             labels={'char':'Level Population', 'charclass' : 'Class', 'lvl_inter2' : 'Level', 'dates' : 'Date'}, height=400)



fig2.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2008-11-18',

            y0=0,

            x1='2008-11-18',

            y1=50000,

            line=dict(

                color="Black",

                width=2

            )))

fig2.add_trace(go.Scatter(

    x=['2008-5-30', '2008-12-30'],

    y=[40000, 10000],

    text=["Burning Crusade",

          "WOTLK",],

    mode="text",

))



fig.show()

fig2.show()



del Dataplot, Dataplot2
Dataplot = pd.DataFrame(wowah_data.groupby(['dates', 'zone']).count()['char']).reset_index().sort_values(by = ['dates', 'char'])

Dataplot = Dataplot.groupby(['dates']).tail(2).reset_index().sort_values(by = ['dates', 'char'])



fig = make_subplots(rows=3, cols=1)



fig = px.scatter(Dataplot, x="dates", y="char", color='zone',

             hover_data=['zone', 'char'], 

#              color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class', 'dates' : 'Date'}, height=400)



fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2008-11-18',

            y0=0,

            x1='2008-11-18',

            y1=7000,

            line=dict(

                color="Black",

                width=2

            )))

fig.add_trace(go.Scatter(

    x=['2008-9-30', '2008-12-30'],

    y=[6000, 5000],

    text=["Burning Crusade",

          "WOTLK",],

    mode="text",

))





Dataplot = pd.DataFrame(wowah_data.groupby(['First_day_of_the_week', 'zone']).count()).reset_index()[['char', 'First_day_of_the_week','zone']].sort_values(by = ['First_day_of_the_week', 'char'])

Dataplot.First_day_of_the_week = pd.to_datetime(Dataplot.First_day_of_the_week).dt.strftime('%m/%d/%Y')

# Dataplot =  Dataplot



fig2 = px.bar(Dataplot, x="zone", y="char",

                 color='char',

             hover_data=['char', 'First_day_of_the_week','zone'],

                 animation_group="zone",

                 animation_frame="First_day_of_the_week",

#              color_discrete_map =  dict_color,

             labels={'char':'Class Population', 'charclass' : 'Class'},

                 height=1000)

fig2.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=10))

fig2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000



fig.show()

fig2.show()



del Dataplot
dataplot = wowah_data.groupby('dates')['char'].nunique().to_frame().reset_index()



fig1 = px.line(dataplot, x="dates", y="char" ,hover_data=['dates', 'char'],

              labels={'char':'Unique Player', 'charclass' : 'Class', 'lvl_inter2' : 'lever', 'pers' : 'Percentage', 'dates' : 'Date'}, height=400).update_traces(mode='lines+markers')



fig1.show()

del dataplot
dataplot = wowah_data.copy()



dataplot['weekday'] = dataplot.timestamp.dt.day_name()

dataplot = dataplot.groupby(['dates','weekday'])['char'].nunique().to_frame().reset_index()



fig2 = px.line(dataplot, x="dates", y="char", color = 'weekday' ,hover_data=['dates', 'char'],

              labels={'char':'Unique Player', 'charclass' : 'Class', 'lvl_inter2' : 'lever', 'pers' : 'Percentage', 'dates' : 'Date'}, height=400).update_traces(mode='lines+markers')

# fig.data[slice(0, 6, 1)].update(mode='markers+lines')

fig2.update_layout(legend_orientation="h")



dataplot = wowah_data.copy()

dataplot['weekday'] = dataplot.timestamp.dt.day_name()

dataplot = dataplot.groupby(['weekday'])['char'].nunique().to_frame().reset_index()

dataplot = dataplot.sort_values(by = 'char')



fig3 = px.bar(dataplot, x='weekday', y='char', color = 'weekday',

             hover_data=['weekday', 'char'],  color_discrete_map =  dict_color,

             labels={'char':'Actif Players', 'charclass': 'Character Class', 'rece' : 'Character Race'}, height=500)

fig2.show()

fig3.show()



del dataplot
dataplot = pd.read_csv('../input/retention/retention.csv')



fig4 = px.line(dataplot.groupby('dates')['char'].count().reset_index(), x="dates", y="char",hover_data=['dates', 'char'],

              labels={'char':'New Players', 'charclass' : 'Class', 'lvl_inter2' : 'lever', 'pers' : 'Percentage', 'dates' : 'Date'}, height=400).update_traces(mode='lines+markers')



dataplot = wowah_data.groupby('char')['timestamp'].max().to_frame().reset_index()

dataplot = dataplot.merge(wowah_data.groupby('char')['timestamp'].min().to_frame().reset_index(), left_on = 'char', right_on = 'char')

dataplot['ret'] = (dataplot['timestamp_x'] - dataplot['timestamp_y']).dt.days

dataplot.sort_values('ret', inplace = True)

dataplot = dataplot.groupby(['ret']).count()['char'].reset_index()



fig5 = px.line(dataplot, x="ret", y="char",hover_data=['char', 'ret'],

              labels={'char':'players Population Tenur (log)', 'charclass' : 'Class', 'ret' : 'Players Tenur (days)', 'pers' : 'Percentage', 'dates' : 'Date' }, height=400).update_traces(mode='lines+markers')

fig5.update_layout(yaxis_type="log")



fig4.show()

fig5.show()



del dataplot
Dataplot = wowah_data.copy()

Dataplot['lvl_guild'] = pd.cut(Dataplot[Dataplot['guild'] >= 0]['guild'], 8)

dict_ = {'(0.493, 64.375]' : '[1-64]', '(64.375, 127.75]' : '[65-127]', '(127.75, 191.125],' : '[128-191]', '(191.125, 254.5]' : '[192-254]', '(254.5, 317.875]' : '[255-317]', '(317.875, 381.25]' : '[318-381]', '(381.25, 444.625]' : '[382-444]', '(444.625, 508.0]' : '[445-508]'}

Dataplot['lvl_guild'] = Dataplot['lvl_guild'].astype(str).map(dict_)

Dataplot['lvl_guild'].fillna(-1, inplace = True)

Dataplot = Dataplot.sort_values('guild')



Dataplot2 = pd.DataFrame(Dataplot.groupby(['lvl_guild']).count()['char']).reset_index()

Dataplot = Dataplot.groupby(['lvl_guild', 'charclass'])['char'].count().to_frame().reset_index()

Dataplot = Dataplot.merge(Dataplot2, left_on = 'lvl_guild', right_on = 'lvl_guild')

# Dataplot['pers'] = Dataplot.groupby('lvl_guild').sum()['char_x'].div(Dataplot.char_y /100).round(1)



fig = px.bar(Dataplot, x="lvl_guild", y="char_x", 

             color='charclass',

             hover_data=['charclass'], 

             color_discrete_map =  dict_color,

             labels={'lvl_guild':'Guild Members', 'charclass' : 'Class', 'pers' : 'Class Population Percentage', 'char_x' : 'Guild Members Population' }, height=400)



fig.add_trace(go.Scatter(

    x=['-1',  '[1-64]','[192-254]', '[255-317]', '[318-381]', '[382-444]', '[445-508]',  '[65-127]'],

    y=[330*10**4, 180*10**4, 120*10**4,         130*10**4,     60*10**4, 30*10**4, 20*10**4,310*10**4],

    text=(Dataplot.groupby('lvl_guild').sum()['char_x'].div(Dataplot.char_y.sum()/1000).round(1).astype(str) + '%').values,

    mode="text",

))



Dataplot = wowah_data.copy()

Dataplot['lvl_guild'] = pd.cut(Dataplot[Dataplot['guild'] >= 0]['guild'], 8)

dict_ = {'(0.493, 64.375]' : '[1-64]', '(64.375, 127.75]' : '[65-127]', '(127.75, 191.125],' : '[128-191]', '(191.125, 254.5]' : '[192-254]', '(254.5, 317.875]' : '[255-317]', '(317.875, 381.25]' : '[318-381]', '(381.25, 444.625]' : '[382-444]', '(444.625, 508.0]' : '[445-508]'}

Dataplot['lvl_guild'] = Dataplot['lvl_guild'].astype(str).map(dict_)

Dataplot['lvl_guild'].fillna(-1, inplace = True)

Dataplot = Dataplot.sort_values('guild')



Dataplot2 = pd.DataFrame(Dataplot.groupby(['lvl_guild']).count()['char']).reset_index()

Dataplot = Dataplot.groupby(['lvl_guild', 'race'])['char'].count().to_frame().reset_index()

Dataplot = Dataplot.merge(Dataplot2, left_on = 'lvl_guild', right_on = 'lvl_guild')

# Dataplot['pers'] = Dataplot.groupby('lvl_guild').sum()['char_x'].div(Dataplot.char_y /100).round(1)



fig2 = px.bar(Dataplot, x="lvl_guild", y="char_x", 

             color='race',

             hover_data=['race'], 

             color_discrete_map =  dict_color,

             labels={'lvl_guild':'Guild Members', 'charclass' : 'Class', 'pers' : 'Class Population Percentage','char_x' : 'Guild Members Population'}, height=400)



fig2.add_trace(go.Scatter(

    x=['-1',  '[1-64]','[192-254]', '[255-317]', '[318-381]', '[382-444]', '[445-508]',  '[65-127]'],

    y=[330*10**4, 180*10**4, 120*10**4,         130*10**4,     60*10**4, 30*10**4, 20*10**4,310*10**4],

    text=(Dataplot.groupby('lvl_guild').sum()['char_x'].div(Dataplot.char_y.sum()/1000*2).round(1).astype(str) + '%').values,

    mode="text",

))



fig.show()

fig2.show()



del Dataplot, Dataplot2

Dataplot = wowah_data.copy()

Dataplot['lvl_guild'] = pd.cut(Dataplot[Dataplot['guild'] >= 0]['guild'], 8)

dict_ = {'(0.493, 64.375]' : '[1-64]', '(64.375, 127.75]' : '[65-127]', '(127.75, 191.125],' : '[128-191]', '(191.125, 254.5]' : '[192-254]', '(254.5, 317.875]' : '[255-317]', '(317.875, 381.25]' : '[318-381]', '(381.25, 444.625]' : '[382-444]', '(444.625, 508.0]' : '[445-508]'}

Dataplot['lvl_guild'] = Dataplot['lvl_guild'].astype(str).map(dict_)

Dataplot['lvl_guild'].fillna(-1, inplace = True)

Dataplot = Dataplot.sort_values('guild')



Dataplot2 = pd.DataFrame(Dataplot.groupby(['lvl_guild']).count()['char']).reset_index()

Dataplot3 = pd.DataFrame(Dataplot.groupby(['charclass']).count()['char']).reset_index()

Dataplot = Dataplot.groupby(['lvl_guild', 'charclass'])['char'].count().to_frame().reset_index()

Dataplot = Dataplot.merge(Dataplot2, left_on = 'lvl_guild', right_on = 'lvl_guild')

Dataplot = Dataplot.merge(Dataplot3, left_on = 'charclass', right_on = 'charclass')

Dataplot['char_pers'] = Dataplot['char_x'].div(Dataplot['char']/100).round(1)

Dataplot

fig = px.bar(Dataplot, x="lvl_guild", y='char_pers', 

             color='charclass',

             hover_data=['charclass'], 

             color_discrete_map =  dict_color,

             labels={'lvl_guild':'Guild Members', 'charclass' : 'Class', 'pers' : 'Class Population Percentage', 'char_x' : 'Guild Members Population',

                    'y' : 'Character Percentage %', 'char_pers' : 'Character Percentage %'}, height=400)



Dataplot = wowah_data.copy()

Dataplot['lvl_guild'] = pd.cut(Dataplot[Dataplot['guild'] >= 0]['guild'], 8)

dict_ = {'(0.493, 64.375]' : '[1-64]', '(64.375, 127.75]' : '[65-127]', '(127.75, 191.125],' : '[128-191]', '(191.125, 254.5]' : '[192-254]', '(254.5, 317.875]' : '[255-317]', '(317.875, 381.25]' : '[318-381]', '(381.25, 444.625]' : '[382-444]', '(444.625, 508.0]' : '[445-508]'}

Dataplot['lvl_guild'] = Dataplot['lvl_guild'].astype(str).map(dict_)

Dataplot['lvl_guild'].fillna(-1, inplace = True)

Dataplot = Dataplot.sort_values('guild')



Dataplot2 = pd.DataFrame(Dataplot.groupby(['lvl_guild']).count()['char']).reset_index()

Dataplot3 = pd.DataFrame(Dataplot.groupby(['race']).count()['char']).reset_index()

Dataplot = Dataplot.groupby(['lvl_guild', 'race'])['char'].count().to_frame().reset_index()

Dataplot = Dataplot.merge(Dataplot2, left_on = 'lvl_guild', right_on = 'lvl_guild')

Dataplot = Dataplot.merge(Dataplot3, left_on = 'race', right_on = 'race')

Dataplot['race_pers'] = Dataplot['char_x'].div(Dataplot['char']/100).round(1)



fig2 = px.bar(Dataplot, x="lvl_guild", y="race_pers", 

             color='race',

             hover_data=['race'], 

             color_discrete_map =  dict_color,

             labels={'lvl_guild':'Guild Members', 'charclass' : 'Class', 'pers' : 'Class Population Percentage','char_x' : 'Guild Members Population',

                     'y' : 'Race Percentage %', 'race_pers' : 'Race Percentage %'}, height=400)



fig.show()

fig2.show()



del Dataplot, Dataplot2, Dataplot3
Dataplot = wowah_data.copy()

Dataplot['lvl_guild'] = pd.cut(Dataplot[Dataplot['guild'] >= 0]['guild'], 8)

dict_ = {'(0.493, 64.375]' : '[1-64]', '(64.375, 127.75]' : '[65-127]', '(127.75, 191.125],' : '[128-191]', '(191.125, 254.5]' : '[192-254]', '(254.5, 317.875]' : '[255-317]', '(317.875, 381.25]' : '[318-381]', '(381.25, 444.625]' : '[382-444]', '(444.625, 508.0]' : '[445-508]'}

Dataplot['lvl_guild'] = Dataplot['lvl_guild'].astype(str).map(dict_)

Dataplot['lvl_guild'].fillna(-1, inplace = True)

Dataplot = Dataplot.sort_values('guild')

Dataplot['lvl_inter'] = pd.cut(Dataplot['level'], 8)

dict_ = {'(0.921, 10.875]' : '[1-10]', '(10.875, 20.75]' : '[10-20]', '(50.375, 60.25]' : '[50-60]', '(60.25, 70.125]' : '[60-70]', '(40.5, 50.375]' : '[40-50]', '(20.75, 30.625]' : '[20-30]', '(30.625, 40.5]' : '[30-40]', '(70.125, 80.0]' : '[70-80]'}

Dataplot['lvl_inter2'] = Dataplot['lvl_inter'].astype(str).map(dict_)

Dataplot1 = Dataplot[Dataplot['extention']=='BC']

fig = px.parallel_categories(Dataplot1.sample(n=10000, random_state=42), dimensions = ['race', 'charclass', 'lvl_guild', 'lvl_inter2'],

                             color = 'level',

                            labels={'race':'Race', 'charclass' : 'Class', 'lvl_guild' : 'Guild Members','lvl_inter2' : 'Character Levels'},

                            title = 'BC')

fig.show()



Dataplot2 = Dataplot[Dataplot['extention']=='WOTLK']

fig2 = px.parallel_categories(Dataplot2.sample(n=10000, random_state=42), dimensions = ['race', 'charclass', 'lvl_guild', 'lvl_inter2'],

                             color = 'level',

                             labels={'race':'Race', 'charclass' : 'Class', 'lvl_guild' : 'Guild Members','lvl_inter2' : 'Character Levels'},

                             title = 'WOTLK')

fig2.show()



del Dataplot, Dataplot1, Dataplot2
Dataplot = wowah_data[wowah_data['extention'] == 'BC'].sort_values(by = 'timestamp')



Dataplot['zone2'] = Dataplot.groupby(['char'])['zone'].shift(1)

Dataplot = Dataplot.dropna()[['char', 'zone', 'zone2']]



Dataplot = Dataplot.groupby(Dataplot.columns.tolist(),as_index=False).size().to_frame().reset_index()

Dataplot = Dataplot.groupby(['zone', 'zone2'])[0].sum().to_frame().reset_index()

Dataplot = Dataplot[(Dataplot['zone'] != Dataplot['zone2']) & (Dataplot[0] > 600)]



got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook = True)



got_net.barnes_hut()



zones = pd.DataFrame(pd.concat([Dataplot['zone'], Dataplot['zone2']]).sort_values().unique())

zones = zones[zones[0] != '-'][0].values



poids = pd.DataFrame(pd.concat([Dataplot.groupby(['zone'])[0].max(), 

                                Dataplot.groupby(['zone2'])[0].max()])).sort_index()

poids = poids.groupby(poids.index).sum()[0].values



col = {}

import random

for i in zones :

    color = "%06x" % random.randint(0, 0xFFFFFF)

    col[i] = '#' + color 

colors = list(col.values())

colors = list(pd.DataFrame(colors)[0].astype(str).values)



got_net.add_nodes(zones, 

                  size=np.exp(np.log10(poids/10)),

                  title= zones,

                  color=colors

                 )



edge_data = zip(Dataplot['zone'], Dataplot['zone2'], Dataplot[0])



for e in tqdm.tqdm(edge_data):

    src = e[0]

    dst = e[1]

    w = e[2]

    got_net.add_edge(src, dst, width = np.exp(np.log10(w/10)))



neighbor_map = got_net.get_adj_list()

for node in got_net.nodes:

    node["title"] += "<br />" + " Neighbors:<br>" + "<br />".join(neighbor_map[node["id"]])

    node["labelHighlightBold"] = True



got_net.show("Azeroth BC.html")
Dataplot = wowah_data[wowah_data['extention'] == 'WOTLK'].sort_values(by = 'timestamp')



Dataplot['zone2'] = Dataplot.groupby(['char'])['zone'].shift(1)

Dataplot = Dataplot.dropna()[['char', 'zone', 'zone2']]



Dataplot = Dataplot.groupby(Dataplot.columns.tolist(),as_index=False).size().to_frame().reset_index()

Dataplot = Dataplot.groupby(['zone', 'zone2'])[0].sum().to_frame().reset_index()

Dataplot = Dataplot[(Dataplot['zone'] != Dataplot['zone2']) & (Dataplot[0] > 100)]



got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook = True)



got_net.barnes_hut()



zones = pd.DataFrame(pd.concat([Dataplot['zone'], Dataplot['zone2']]).sort_values().unique())

zones = zones[zones[0] != '-'][0].values



poids = pd.DataFrame(pd.concat([Dataplot.groupby(['zone'])[0].max(), 

                                Dataplot.groupby(['zone2'])[0].max()])).sort_index()

poids = poids.groupby(poids.index).sum()[0].values



col = {}

import random

for i in zones :

    color = "%06x" % random.randint(0, 0xFFFFFF)

    col[i] = '#' + color 

colors = list(col.values())

colors = list(pd.DataFrame(colors)[0].astype(str).values)



got_net.add_nodes(zones, 

                  size=np.exp(np.log10(poids)),

                  title= zones,

                  color=colors

                 )



edge_data = zip(Dataplot['zone'], Dataplot['zone2'], Dataplot[0])



for e in tqdm.tqdm(edge_data):

    src = e[0]

    dst = e[1]

    w = e[2]

    got_net.add_edge(src, dst, width = np.exp(np.log10(w)))



neighbor_map = got_net.get_adj_list()

for node in got_net.nodes:

    node["title"] += "<br />" + " Neighbors:<br>" + "<br />".join(neighbor_map[node["id"]])

    node["labelHighlightBold"] = True



got_net.show("Azeroth WOTLK.html")