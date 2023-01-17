import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

#from bubbly.bubbly import bubbleplot 

#from __future__ import division

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



IS_LOCAL = False

import os

if(IS_LOCAL):

    PATH="../input/120-years-of-olympic-history-athlets-and-results"

else:

    PATH="../input"

print(os.listdir(PATH))
athlete_events_df = pd.read_csv(PATH+"/athlete_events.csv")

noc_regions_df = pd.read_csv(PATH+"/noc_regions.csv")
print("Athletes and Events data -  rows:",athlete_events_df.shape[0]," columns:", athlete_events_df.shape[1])

print("NOC Regions data -  rows:",noc_regions_df.shape[0]," columns:", noc_regions_df.shape[1])
athlete_events_df.head(5)
noc_regions_df.head(5)
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data(athlete_events_df)
missing_data(noc_regions_df)
tmp = athlete_events_df.groupby(['Year', 'City'])['Season'].value_counts()

df = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()
df.head(3)
trace = go.Scatter(

    x = df['Year'],

    y = df['Athlets'],

    name="Athlets per Olympic game",

    marker=dict(

        color="Blue",

    ),

    mode = "markers"

)

data = [trace]

layout = dict(title = 'Athlets per Olympic game',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of athlets'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-athlets1')
dfS = df[df['Season']=='Summer']; dfW = df[df['Season']=='Winter']



traceS = go.Scatter(

    x = dfS['Year'],y = dfS['Athlets'],

    name="Summer Games",

    marker=dict(color="Red"),

    mode = "markers+lines"

)

traceW = go.Scatter(

    x = dfW['Year'],y = dfW['Athlets'],

    name="Winter Games",

    marker=dict(color="Blue"),

    mode = "markers+lines"

)



data = [traceS, traceW]

layout = dict(title = 'Athlets per Olympic game',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of athlets'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-athlets2')
traceS = go.Scatter(

    x = dfS['Year'],y = dfS['Athlets'],

    name="Summer Games",

    marker=dict(color="Red"),

    mode = "markers+lines",

    text=dfS['City'],

)

traceW = go.Scatter(

    x = dfW['Year'],y = dfW['Athlets'],

    name="Winter Games",

    marker=dict(color="Blue"),

    mode = "markers+lines",

    text=dfW['City']

)



data = [traceS, traceW]



fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Number athlets: Summer Games', 'Number athlets: Winter Games'))

fig.append_trace(traceS, 1, 1)

fig.append_trace(traceW, 1, 2)



iplot(fig, filename='events-athlets2')
tmp = athlete_events_df.groupby('Year')['City'].value_counts()

df2 = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()

df2 = df2.merge(df)
iplot(ff.create_table(df2.head(3)), filename='jupyter-table2')
dfS = df2[df2['Season']=='Summer']; dfW = df2[df2['Season']=='Winter']



traceS = go.Bar(

    x = dfS['Year'],y = dfS['Athlets'],

    name="Summer Games",

    marker=dict(color="Red"),

    text=dfS['City']

)

traceW = go.Bar(

    x = dfW['Year'],y = dfW['Athlets'],

    name="Winter Games",

    marker=dict(color="Blue"),

    text=dfS['City']

)



data = [traceS, traceW]

layout = dict(title = 'Athlets per Olympic game',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of athlets'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-athlets3')
traceS = go.Bar(

    x = dfS['Year'],y = dfS['Athlets'],

    name="Summer Games",

     marker=dict(

                color='rgb(238,23,11)',

                line=dict(

                    color='black',

                    width=0.75),

                opacity=0.7,

            ),

    text=dfS['City'],

    

)

traceW = go.Bar(

    x = dfW['Year'],y = dfW['Athlets'],

    name="Winter Games",

    marker=dict(

                color='rgb(11,23,245)',

                line=dict(

                    color='black',

                    width=0.75),

                opacity=0.7,

            ),

    text=dfS['City']

)



data = [traceS, traceW]

layout = dict(title = 'Athlets per Olympic game',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of athlets'),

          hovermode = 'closest',

          barmode='stack'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-athlets4')
traceS = go.Box(

    x = dfS['Athlets'],

    name="Summer Games",

    

     marker=dict(

                color='rgba(238,23,11,0.5)',

                line=dict(

                    color='red',

                    width=1.2),

            ),

    text=dfS['City'],

    orientation='h',

    

)

traceW = go.Box(

    x = dfW['Athlets'],

    name="Winter Games",

    marker=dict(

                color='rgba(11,23,245,0.5)',

                line=dict(

                    color='blue',

                    width=1.2),

            ),

    text=dfS['City'],  orientation='h',

)



data = [traceS, traceW]

layout = dict(title = 'Athlets per Olympic game',

          xaxis = dict(title = 'Number of athlets',showticklabels=True),

          yaxis = dict(title = 'Season', showticklabels=True, tickangle=-90), 

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-athlets5')
tmp = athlete_events_df.groupby(['Year', 'City','Season'])['Sport'].nunique()

df = pd.DataFrame(data={'Sports': tmp.values}, index=tmp.index).reset_index()
df.head(3)
dfS = df[df['Season']=='Summer']; dfW = df[df['Season']=='Winter']



traceS = go.Bar(

    x = dfS['Year'],y = dfS['Sports'],

    name="Summer Games",

     marker=dict(

                color='rgb(238,23,11)',

                line=dict(

                    color='red',

                    width=1),

                opacity=0.5,

            ),

    text= dfS['City'],

)

traceW = go.Bar(

    x = dfW['Year'],y = dfW['Sports'],

    name="Winter Games",

    marker=dict(

                color='rgb(11,23,245)',

                line=dict(

                    color='blue',

                    width=1),

                opacity=0.5,

            ),

    text=dfS['City']

)



data = [traceS, traceW]

layout = dict(title = 'Sports per Olympic edition',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of sports'),

          hovermode = 'closest',

          barmode='stack'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-sports1')
tmp = athlete_events_df.groupby(['Year', 'City','Season'])['Sport'].value_counts()

df = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()

df.head()
dfS = df[df['Season']=='Summer']; dfW = df[df['Season']=='Winter']





traceS = go.Scatter(

    x = dfS['Year'],y = dfS['Athlets'],

    name="Summer Games",

     marker=dict(

                color='rgb(238,23,11)',

                line=dict(

                    color='red',

                    width=1),

                opacity=0.5,

            ),

    text= "City:"+dfS['City']+" Sport:"+dfS['Sport'],

    mode = "markers"

)

traceW = go.Scatter(

    x = dfW['Year'],y = dfW['Athlets'],

    name="Winter Games",

    marker=dict(

                color='rgb(11,23,245)',

                line=dict(

                    color='blue',

                    width=1),

                opacity=0.5,

            ),

   text= "City:"+dfW['City']+" Sport:"+dfW['Sport'],

    mode = "markers"

)



data = [traceS, traceW]

layout = dict(title = 'Number of athlets per sport for each Olympic edition',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of athlets per sport'),

          hovermode='closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-sports1')
tmp = athlete_events_df.groupby(['Year', 'City','Season'])['Sport'].value_counts()

df = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()

df.head(3)
sports = (athlete_events_df.groupby(['Sport'])['Sport'].nunique()).index
def draw_trace(dataset, sport):

    dfS = dataset[dataset['Sport']==sport];

    trace = go.Box(

        x = dfS['Athlets'],

        name=sport,

         marker=dict(

                    line=dict(

                        color='black',

                        width=0.8),

                ),

        text=dfS['City'], 

        orientation = 'h'

    )

    return trace





def draw_group(dataset, title,height=800):

    data = list()

    for sport in sports:

        data.append(draw_trace(dataset, sport))





    layout = dict(title = title,

              xaxis = dict(title = 'Number of athlets',showticklabels=True),

              yaxis = dict(title = 'Sport', showticklabels=True, tickfont=dict(

                family='Old Standard TT, serif',

                size=8,

                color='black'),), 

              hovermode = 'closest',

              showlegend=False,

                  width=800,

                  height=height,

             )

    fig = dict(data=data, layout=layout)

    iplot(fig, filename='events-sports1')



# select only Summer Olympics

df_S = df[df['Season']=='Summer']

# draw the boxplots for the Summer Olympics

draw_group(df_S, "Athlets per Sport (Summer Olympics)")
# select only Winter Olympics

df_W = df[df['Season']=='Winter']

# draw the boxplots for the Summer Olympics

draw_group(df_W, "Athlets per Sport (Winter Olympics)",600)
piv = pd.pivot_table(df_S, values="Athlets",index=["Year"], columns=["Sport"], fill_value=0)

m = piv.values
trace = go.Heatmap(z = m, y= list(piv.index), x=list(piv.columns),colorscale='Reds',reversescale=False)

data=[trace]

layout = dict(title = "Number of athlets per year and sport (Summer Olympics)",

              xaxis = dict(title = 'Sport',

                        showticklabels=True,

                           tickangle = 45,

                        tickfont=dict(

                                size=10,

                                color='black'),

                          ),

              yaxis = dict(title = 'Year', 

                        showticklabels=True, 

                        tickfont=dict(

                            size=10,

                            color='black'),

                      ), 

              hovermode = 'closest',

              showlegend=False,

                  width=1000,

                  height=800,

             )

fig = dict(data=data, layout=layout)

iplot(fig, filename='labelled-heatmap')
piv = pd.pivot_table(df_W, values="Athlets",index=["Year"], columns=["Sport"], fill_value=0)

m = piv.values
trace = go.Heatmap(z = m, y= list(piv.index), x=list(piv.columns),colorscale='Blues',reversescale=True)

data=[trace]

layout = dict(title = "Number of athlets per year and sport (Winter Olympics)",

              xaxis = dict(title = 'Sport',

                        showticklabels=True,

                           tickangle = 30,

                        tickfont=dict(

                                size=8,

                                color='black'),

                          ),

              yaxis = dict(title = 'Year', 

                        showticklabels=True, 

                        tickfont=dict(

                            size=10,

                            color='black'),

                      ), 

              hovermode = 'closest',

              showlegend=False,

                  width=800,

                  height=800,

             )

fig = dict(data=data, layout=layout)

iplot(fig, filename='labelled-heatmap')
labels = ['Sunny side of pyramid','Shaddy side of pyramid','Sky']

values = [300,150,1200]

colors = ['gold', 'brown', 'lightblue']



BOTTOM_OF_THE_PYRAMID_ACCORDING_TO_NEWTON_LAWS = 220



trace = go.Pie(labels=labels, values=values,

               hoverinfo='label', textinfo='none', 

               textfont=dict(size=20),

               rotation=BOTTOM_OF_THE_PYRAMID_ACCORDING_TO_NEWTON_LAWS,

               marker=dict(colors=colors, 

                           line=dict(color='#000000', width=1)))

iplot([trace], filename='styled_pie_chart')
tmp = athlete_events_df.groupby(['Season'])['Sport'].value_counts()

df = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()

df.head(3)
df_S = df[df['Season']=='Summer']



trace = go.Pie(labels=df_S['Sport'], 

               values=df_S['Athlets'],

               hoverinfo='label+value+percent', 

               textinfo='value+percent', 

               textfont=dict(size=8),

               rotation=180,

               marker=dict(colors=colors, 



                           line=dict(color='#000000', width=1)

                        )

            )



data = [trace]

layout = dict(title = "Number of athlets per sport (Summer Olympics)",

                  width=800,

                  height=1200,

              legend=dict(orientation="h")

             )

fig = dict(data=data,layout=layout)

iplot(fig, filename='styled_pie_chart')
df_S = df[df['Season']=='Winter']



trace = go.Pie(labels=df_S['Sport'], 

               values=df_S['Athlets'],

               hoverinfo='label+value+percent', 

               textinfo='value+percent', 

               textfont=dict(size=8),

               rotation=180,

               marker=dict(colors=colors, 



                           line=dict(color='#000000', width=1)

                        )

            )



data = [trace]

layout = dict(title = "Number of athlets per sport (Winter Olympics)",

                  width=800,

                  height=800,

              legend=dict(orientation="h")

             )

fig = dict(data=data,layout=layout)

iplot(fig, filename='styled_pie_chart')
olympics_df = athlete_events_df.merge(noc_regions_df)
print("All Olympics data -  rows:",olympics_df.shape[0]," columns:", olympics_df.shape[1])
olympics_df.head(3)
olympics_df=olympics_df.rename(columns = {'region':'Country'})
tmp = olympics_df.groupby(['Country'])['Year'].nunique()

df = pd.DataFrame(data={'Editions': tmp.values}, index=tmp.index).reset_index()

df.head(2)
trace = go.Choropleth(

            locations = df['Country'],

            locationmode='country names',

            z = df['Editions'],

            text = df['Country'],

            autocolorscale =False,

            reversescale = True,

            colorscale = 'rainbow',

            marker = dict(

                line = dict(

                    color = 'rgb(0,0,0)',

                    width = 0.5)

            ),

            colorbar = dict(

                title = 'Editions',

                tickprefix = '')

        )



data = [trace]

layout = go.Layout(

    title = 'Olympic countries',

    geo = dict(

        showframe = True,

        showlakes = False,

        showcoastlines = True,

        projection = dict(

            type = 'natural earth'

        )

    )

)



fig = dict( data=data, layout=layout )

iplot(fig)
tmp = olympics_df.groupby(['Country', 'Season'])['Year'].nunique()

df = pd.DataFrame(data={'Editions': tmp.values}, index=tmp.index).reset_index()

df.head(2)
dfS = df[df['Season']=='Summer']; dfW = df[df['Season']=='Winter']



def draw_map(dataset, title, colorscale, reversescale=False):

    trace = go.Choropleth(

                locations = dataset['Country'],

                locationmode='country names',

                z = dataset['Editions'],

                text = dataset['Country'],

                autocolorscale =False,

                reversescale = reversescale,

                colorscale = colorscale,

                marker = dict(

                    line = dict(

                        color = 'rgb(0,0,0)',

                        width = 0.5)

                ),

                colorbar = dict(

                    title = 'Editions',

                    tickprefix = '')

            )



    data = [trace]

    layout = go.Layout(

        title = title,

        geo = dict(

            showframe = True,

            showlakes = False,

            showcoastlines = True,

            projection = dict(

                type = 'orthographic'

            )

        )

    )

    fig = dict( data=data, layout=layout )

    iplot(fig)

    

draw_map(dfS, 'Olympic countries (Summer games)', "Reds")
draw_map(dfW, 'Olympic countries (Winter games)', "Blues", True)
tmp = olympics_df.groupby(['Year','Sport'])['Country'].value_counts()

dataset = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()

dataset.head()
female_h = olympics_df[olympics_df['Sex']=='F']['Height'].dropna()

male_h = olympics_df[olympics_df['Sex']=='M']['Height'].dropna()



hist_data = [female_h, male_h]

group_labels = ['Female Height', 'Male Height']



fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

fig['layout'].update(title='Athlets Height distribution plot')

iplot(fig, filename='dist_only')
female_w = olympics_df[olympics_df['Sex']=='F']['Weight'].dropna()

male_w = olympics_df[olympics_df['Sex']=='M']['Weight'].dropna()



hist_data = [female_w, male_w]

group_labels = ['Female Weight', 'Male Weight']



fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

fig['layout'].update(title='Athlets Weight distribution plot')

iplot(fig, filename='dist_only')
female_a = olympics_df[olympics_df['Sex']=='F']['Age'].dropna()

male_a = olympics_df[olympics_df['Sex']=='M']['Age'].dropna()



hist_data = [female_a, male_a]

group_labels = ['Female Age', 'Male Age']



fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

fig['layout'].update(title='Athlets Age distribution plot')

iplot(fig, filename='dist_only')
tmp = olympics_df.groupby(['Sport'])['Height', 'Weight'].agg('mean').dropna()

df1 = pd.DataFrame(tmp).reset_index()

tmp2 = olympics_df.groupby(['Sport'])['ID'].count()

df2 = pd.DataFrame(tmp2).reset_index()

dataset = df1.merge(df2)
hover_text = []

for index, row in dataset.iterrows():

    hover_text.append(('Sport: {}<br>'+

                      'Number of athlets: {}<br>'+

                      'Mean Height: {}<br>'+

                      'Mean Weight: {}<br>').format(row['Sport'],

                                            row['ID'],

                                            round(row['Height'],2),

                                            round(row['Weight'],2)))

dataset['hover_text'] = hover_text
data = []

for sport in dataset['Sport']:

    ds = dataset[dataset['Sport']==sport]

    trace = go.Scatter(

        x = ds['Height'],

        y = ds['Weight'],

        name = sport,

        marker=dict(

            symbol='circle',

            sizemode='area',

            sizeref=10,

            size=ds['ID'],

            line=dict(

                width=2

            ),),

        text = ds['hover_text']

    )

    data.append(trace)

                         

layout = go.Layout(

    title='Athlets height and weight mean - grouped by sport',

    xaxis=dict(

        title='Height [cm]',

        gridcolor='rgb(128, 128, 128)',

        zerolinewidth=1,

        ticklen=1,

        gridwidth=0.5,

    ),

    yaxis=dict(

        title='Weight [kg]',

        gridcolor='rgb(128, 128, 128)',

        zerolinewidth=1,

        ticklen=1,

        gridwidth=0.5,

    ),

    paper_bgcolor='rgb(255,255,255)',

    plot_bgcolor='rgb(254, 254, 254)',

    showlegend=False,

)





fig = dict(data = data, layout = layout)



iplot(fig, filename='athlets_body_measures')

                         
tmp = olympics_df.groupby(['Sport', 'Year'])['Height', 'Weight'].agg('mean').dropna()

df1 = pd.DataFrame(tmp).reset_index()

tmp2 = olympics_df.groupby(['Sport', 'Year'])['ID'].count()

df2 = pd.DataFrame(tmp2).reset_index()

dataset = df1.merge(df2)
dataset.head(3)
hover_text = []

for index, row in dataset.iterrows():

    hover_text.append(('Year: {}<br>'+

                       'Sport: {}<br>'+

                      'Number of athlets: {}<br>'+

                      'Mean Height: {}<br>'+

                      'Mean Weight: {}<br>').format(row['Year'], 

                                            row['Sport'],

                                            row['ID'],

                                            round(row['Height'],2),

                                            round(row['Weight'],2)))

dataset['hover_text'] = hover_text
years = (olympics_df.groupby(['Year'])['Year'].nunique()).index

sports = (olympics_df.groupby(['Sport'])['Sport'].nunique()).index

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# fill in most of layout

figure['layout']['xaxis'] = {'range': [140, 200], 'title': 'Height'}

figure['layout']['yaxis'] = {'range': [20, 200],'title': 'Weight'}

figure['layout']['hovermode'] = 'closest'

figure['layout']['showlegend'] = False

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1896',

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}



figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]

sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Year:',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}

# make data

year = 1896

for sport in sports:

    dataset_by_year = dataset[dataset['Year'] == year]

    dataset_by_year_and_season = dataset_by_year[dataset_by_year['Sport'] == sport]



    data_dict = {

        'x': list(dataset_by_year_and_season['Height']),

        'y': list(dataset_by_year_and_season['Weight']),

        'mode': 'markers',

        'text': list(dataset_by_year_and_season['hover_text']),

        'marker': {

            'sizemode': 'area',

            'sizeref': 1,

            'size': list(dataset_by_year_and_season['ID'])

        },

        'name': sport

    }

    figure['data'].append(data_dict)

# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for sport in sports:

        dataset_by_year = dataset[dataset['Year'] == int(year)]

        dataset_by_year_and_season = dataset_by_year[dataset_by_year['Sport'] == sport]



        data_dict = {

            'x': list(dataset_by_year_and_season['Height']),

            'y': list(dataset_by_year_and_season['Weight']),

            'mode': 'markers',

            'text': list(dataset_by_year_and_season['hover_text']),

            'marker': {

                'sizemode': 'area',

                'sizeref': 1,

                'size':  list(dataset_by_year_and_season['ID'])

            },

            'name': sport

        }

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)

figure['layout']['sliders'] = [sliders_dict]

iplot(figure)
tmp = olympics_df.groupby(['Sex'])['Height', 'Weight'].agg('mean').dropna()

df1 = pd.DataFrame(tmp).reset_index()

tmp2 = olympics_df.groupby(['Sex'])['ID'].count()

df2 = pd.DataFrame(tmp2).reset_index()

dataset = df1.merge(df2)
hover_text = []

for index, row in dataset.iterrows():

    hover_text.append(('Sex: {}<br>'+

                      'Number of athlets: {}<br>'+

                      'Mean Height: {}<br>'+

                      'Mean Weight: {}<br>').format(row['Sex'],

                                            row['ID'],

                                            round(row['Height'],2),

                                            round(row['Weight'],2)))

dataset['hover_text'] = hover_text
data = []

for sex in dataset['Sex']:

    ds = dataset[dataset['Sex']==sex]

    trace = go.Scatter(

        x = ds['Height'],

        y = ds['Weight'],

        name = sex,

        marker=dict(

            symbol='circle',

            sizemode='area',

            sizeref=10,

            size=ds['ID'],

            line=dict(

                width=2

            ),),

        text = ds['hover_text']

    )

    data.append(trace)

                         

layout = go.Layout(

    title='Athlets height and weight mean - grouped by Sex',

    xaxis=dict(

        title='Height [cm]',

        gridcolor='rgb(128, 128, 128)',

        zerolinewidth=1,

        ticklen=1,

        gridwidth=0.5,

    ),

    yaxis=dict(

        title='Weight [kg]',

        gridcolor='rgb(128, 128, 128)',

        zerolinewidth=1,

        ticklen=1,

        gridwidth=0.5,

    ),

    paper_bgcolor='rgb(255,255,255)',

    plot_bgcolor='rgb(254, 254, 254)',

    showlegend=False,

)





fig = dict(data = data, layout = layout)



iplot(fig, filename='athlets_body_measures2')

                         
tmp = olympics_df.groupby(['Sex', 'Year'])['Height', 'Weight'].agg('mean').dropna()

df1 = pd.DataFrame(tmp).reset_index()

tmp2 = olympics_df.groupby(['Sex', 'Year'])['ID'].count()

df2 = pd.DataFrame(tmp2).reset_index()

dataset = df1.merge(df2)
hover_text = []

for index, row in dataset.iterrows():

    hover_text.append(('Year: {}<br>'+

                       'Sex: {}<br>'+

                      'Number of athlets: {}<br>'+

                      'Mean Height: {}<br>'+

                      'Mean Weight: {}<br>').format(row['Year'], 

                                            row['Sex'],

                                            row['ID'],

                                            round(row['Height'],2),

                                            round(row['Weight'],2)))

dataset['hover_text'] = hover_text
years = (olympics_df.groupby(['Year'])['Year'].nunique()).index

sexes = (olympics_df.groupby(['Sex'])['Sex'].nunique()).index

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# fill in most of layout

figure['layout']['xaxis'] = {'range': [100, 200], 'title': 'Height'}

figure['layout']['yaxis'] = {'range': [20, 200],'title': 'Weight'}

figure['layout']['hovermode'] = 'closest'

figure['layout']['showlegend'] = False

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1896',

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}



figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]

sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Year:',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}

# make data

year = 1896

for sex in sexes:

    dataset_by_year = dataset[dataset['Year'] == year]

    dataset_by_year_and_season = dataset_by_year[dataset_by_year['Sex'] == sex]



    data_dict = {

        'x': list(dataset_by_year_and_season['Height']),

        'y': list(dataset_by_year_and_season['Weight']),

        'mode': 'markers',

        'text': list(dataset_by_year_and_season['hover_text']),

        'marker': {

            'sizemode': 'area',

            'sizeref': 1,

            'size': list(dataset_by_year_and_season['ID'])

        },

        'name': sex

    }

    figure['data'].append(data_dict)

# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for sex in sexes:

        dataset_by_year = dataset[dataset['Year'] == int(year)]

        dataset_by_year_and_season = dataset_by_year[dataset_by_year['Sex'] == sex]



        data_dict = {

            'x': list(dataset_by_year_and_season['Height']),

            'y': list(dataset_by_year_and_season['Weight']),

            'mode': 'markers',

            'text': list(dataset_by_year_and_season['hover_text']),

            'marker': {

                'sizemode': 'area',

                'sizeref': 1,

                'size':  list(dataset_by_year_and_season['ID'])

            },

            'name': sex

        }

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)

figure['layout']['sliders'] = [sliders_dict]

iplot(figure)
tmp = olympics_df.groupby(['Sport', 'Sex'])['Height', 'Weight', 'Age'].agg('mean').dropna()

df1 = pd.DataFrame(tmp).reset_index()

tmp2 = olympics_df.groupby(['Sport', 'Sex'])['ID'].count()

df2 = pd.DataFrame(tmp2).reset_index()

dataset = df1.merge(df2)
dataset.head()
hover_text = []

for index, row in dataset.iterrows():

    hover_text.append(('Sex: {}<br>'+

                       'Sport: {}<br>'

                       'Number of athlets: {}<br>'+

                       'Mean Age: {}<br>'

                       'Mean Height: {}<br>'+

                       'Mean Weight: {}<br>').format(row['Sex'],

                                            row['Sport'],

                                            row['ID'],

                                            round(row['Age'],2), 

                                            round(row['Height'],2),

                                            round(row['Weight'],2)))

dataset['hover_text'] = hover_text


def plot_bubble_chart(dataset,title):

    data = []

    for sport in dataset['Sport']:

        ds = dataset[dataset['Sport']==sport]

        trace = go.Scatter(

            x = ds['Height'],

            y = ds['Weight'],

            name = sport,

            marker=dict(

                symbol='circle',

                sizemode='area',

                sizeref=50,

                size=np.power(ds['Age'],3),

                line=dict(

                    width=2

                ),),

            text = ds['hover_text']

        )

        data.append(trace)



    layout = go.Layout(

        title= title,

        xaxis=dict(

            title='Height [cm]',

            gridcolor='rgb(128, 128, 128)',

            zerolinewidth=1,

            ticklen=1,

            gridwidth=0.5,

            range=[150,200]

        ),

        yaxis=dict(

            title='Weight [kg]',

            gridcolor='rgb(128, 128, 128)',

            zerolinewidth=1,

            ticklen=1,

            gridwidth=0.5,

            range=[45,100]

        ),

        paper_bgcolor='rgb(255,255,255)',

        plot_bgcolor='rgb(254, 254, 254)',

        showlegend=False,

    )

    fig = dict(data = data, layout = layout)

    iplot(fig, filename='athlets_body_measures')

    

dF = dataset[dataset['Sex']=='F']

plot_bubble_chart(dF,'Female athlets height and weight mean - grouped by sport')
dM = dataset[dataset['Sex']=='M']

plot_bubble_chart(dM,'Male athlets height and weight mean - grouped by sport')
tmp = olympics_df.groupby(['Country', 'Medal'])['ID'].agg('count').dropna()

df = pd.DataFrame(tmp).reset_index()
dfG = df[df['Medal']=='Gold']

dfS = df[df['Medal']=='Silver']

dfB = df[df['Medal']=='Bronze']



def draw_map(dataset, title, colorscale):

    trace = go.Choropleth(

                locations = dataset['Country'],

                locationmode='country names',

                z = dataset['ID'],

                text = dataset['Country'],

                autocolorscale =False,

                reversescale = True,

                colorscale = colorscale,

                marker = dict(

                    line = dict(

                        color = 'rgb(0,0,0)',

                        width = 0.5)

                ),

                colorbar = dict(

                    title = 'Medals',

                    tickprefix = '')

            )

    data = [trace]

    layout = go.Layout(

        title = title,

        geo = dict(

            showframe = True,

            showlakes = False,

            showcoastlines = True,

            projection = dict(

                type = 'natural earth'

            )

        )

    )

    fig = dict( data=data, layout=layout )

    iplot(fig)
draw_map(dfG, "Countries with Gold Medals",'Greens')
draw_map(dfS, "Countries with Silver Medals",'Greys')
draw_map(dfB, "Countries with Bronze Medals",'Reds')
tmp = olympics_df.groupby(['Year', 'City','Season', 'Medal'])['ID'].agg('count').dropna()

df = pd.DataFrame(tmp).reset_index()

dfG = df[df['Medal']=='Gold']

dfS = df[df['Medal']=='Silver']

dfB = df[df['Medal']=='Bronze']
dfG.head()


traceG = go.Bar(

    x = dfG['Year'],y = dfG['ID'],

    name="Gold",

     marker=dict(

                color='gold',

                line=dict(

                    color='black',

                    width=1),

                opacity=0.5,

            ),

    text = dfG['City']+ " (" + dfG['Season'] + ")",

)

traceS = go.Bar(

    x = dfS['Year'],y = dfS['ID'],

    name="Silver",

    marker=dict(

                color='Grey',

                line=dict(

                    color='black',

                    width=1),

                opacity=0.5,

            ),

    text=dfS['City']+ " (" + dfS['Season'] + ")",

)



traceB = go.Bar(

    x = dfB['Year'],y = dfB['ID'],

    name="Bronze",

    marker=dict(

                color='Brown',

                line=dict(

                    color='black',

                    width=1),

                opacity=0.5,

            ),

    text=dfB['City']+ " (" + dfB['Season'] + ")",

)



data = [traceG, traceS, traceB]

layout = dict(title = 'Medals per Olympic edition',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of medals'),

          hovermode = 'closest',

          barmode='stack'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-sports1')
tmp = olympics_df.groupby(['Sport', 'Medal'])['ID'].agg('count').dropna()

df = pd.DataFrame(tmp).reset_index()

dfG = df[df['Medal']=='Gold']

dfS = df[df['Medal']=='Silver']

dfB = df[df['Medal']=='Bronze']
traceG = go.Bar(

    x = dfG['Sport'],y = dfG['ID'],

    name="Gold",

     marker=dict(

                color='gold',

                line=dict(

                    color='black',

                    width=1),

                opacity=0.5,

            ),

    text = dfG['Sport'],

    #orientation = 'h'

)

traceS = go.Bar(

    x = dfS['Sport'],y = dfS['ID'],

    name="Silver",

    marker=dict(

                color='Grey',

                line=dict(

                    color='black',

                    width=1),

                opacity=0.5,

            ),

    text=dfS['Sport'],

    #orientation = 'h'

)



traceB = go.Bar(

    x = dfB['Sport'],y = dfB['ID'],

    name="Bronze",

    marker=dict(

                color='Brown',

                line=dict(

                    color='black',

                    width=1),

                opacity=0.5,

            ),

    text=dfB['Sport'],

   # orientation = 'h'

)



data = [traceG, traceS, traceB]

layout = dict(title = 'Medals per sport',

          xaxis = dict(title = 'Sport', showticklabels=True, tickangle=45,

            tickfont=dict(

                size=8,

                color='black'),), 

          yaxis = dict(title = 'Number of medals'),

          hovermode = 'closest',

          barmode='stack',

          showlegend=False,

          width=900,

          height=600,

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='events-sports1')