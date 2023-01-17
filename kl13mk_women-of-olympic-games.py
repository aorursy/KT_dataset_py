# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# load the data
ae = pd.read_csv('../input/athlete_events.csv')
noc = pd.read_csv('../input/noc_regions.csv')

#let's merge them
ae_df = ae.merge(noc, how='left')

# some regions are not filled, for simplicity I am filling them with 'Team' value
ae_df['region_filled'] = ae_df['region'].fillna(ae['Team'])

# let's add a region to the name
ae_df['Name_Region'] = ae_df['Name'] + ' (' + ae_df['region_filled'] + ')'

# let's split the set for summer and winter games
ae_summer = ae_df[ae_df['Season']=='Summer'].copy()
ae_winter = ae_df[ae_df['Season']=='Winter'].copy()

# let's split by men and women
fae = ae_df[ae_df['Sex']=='F'].copy()
mae = ae_df[ae_df['Sex']=='M'].copy()
fae['Medal'].fillna(value= 'No Medal', inplace=True)
mae['Medal'].fillna(value= 'No Medal', inplace=True)

# Any results you write to the current directory are saved as output.
ae_df.head()
# just some basic stats
basic_stats = ae_df.groupby(['Year', 'Season', 'City'])['Name'].agg('nunique').reset_index()
basic_stats.head()

# this is needed, since in 1956 the Summer Games were held in Stockholm and Melbourne 5 months apart
conditions = (ae_df['Year'] == 1956) &\
             (ae_df['Season'] == 'Summer')
ae_df.loc[conditions, 'City'] = 'Melbourne/Stockholm'
ae_summer.loc[conditions, 'City'] = 'Melbourne/Stockholm'
summer_games = ae_summer.groupby(['Year', 'City', 'Sex'])['Event', 'Name']\
                        .agg({'Event': 'nunique', 'Name': 'nunique'})\
                        .reset_index()\
                        .pivot_table(index=['Year', 'City'], columns='Sex', values=['Event', 'Name'])\
                        .fillna(0)\
                        .reset_index()
summer_games.columns = ['Year', 'City', 'Count of Events - Female', 'Count of Events - Male', 'Female Athletes', 'Male Athletes']

summer_games.iloc[:, 2:] = summer_games.iloc[:, 2:].astype(int)

# I will need common scaling for the comparison of count of events
pcount_full = summer_games['Female Athletes'].tolist() + summer_games['Male Athletes'].tolist()

# scaling for bubble chart
scaled_pcount_f = (summer_games['Female Athletes'] - min(pcount_full)) / (max(pcount_full) - min(pcount_full)) + 0.1
scaled_pcount_m = (summer_games['Male Athletes'] - min(pcount_full)) / (max(pcount_full) - min(pcount_full)) + 0.1

# I need to create the text for the labels. There must be better way to do it, I am looking for clues :-) 
text_f = []
for i in range(summer_games.shape[0]):
    text = 'In ' + \
           str(summer_games.iloc[i,0]) + \
           ', in ' + \
           str(summer_games.iloc[i, 1]) + \
           ', <br>' + \
           str(summer_games.iloc[i,4]) + \
           ' Female Athletes <br>' + \
           'competed in ' + \
           str(summer_games.iloc[i,2]) + \
           ' events.'
    text_f.append(text)

text_m = []
for i in range(summer_games.shape[0]):
    text = 'In ' + \
           str(summer_games.iloc[i,0]) + \
           ', in ' + \
           str(summer_games.iloc[i, 1]) + \
           ', <br>' + \
           str(summer_games.iloc[i,5]) + \
           ' Male Athletes <br>' + \
           'competed in ' + \
           str(summer_games.iloc[i,3]) + \
           ' events.'
    text_m.append(text)
    
    
# laying out the bubble chart
trace0 = go.Scatter(
    x=summer_games['Year'],
    y=summer_games['Count of Events - Female'],
    mode='markers',
    marker=dict(
        size=scaled_pcount_f*50,
        symbol='circle-open',
        line=dict(
            width=4
        ),
        color='#2c73d2'
    ),
    name='Female Athletes',
    text=text_f,
    hoverinfo='text'
)
trace1 = go.Scatter(
    x=summer_games['Year'],
    y=summer_games['Count of Events - Male'],
    mode='markers',
    marker=dict(
        size=scaled_pcount_m*50,
        symbol='circle-open',
        line=dict(
            width=4
        ),
        color='#f9f871'),
    name='Male Athletes',
    text=text_m,
    hoverinfo='text'
)

layout1 = go.Layout(
    title='Chasing Equality - Summer Olympics',
    titlefont=dict(
        size=36,
        color='#4b8480'
    ),
    xaxis=dict(
        title='Year',
        color='#4b8480',
        titlefont=dict(
            size=20
        ),
        showline=True,
        linewidth=1,
        linecolor='#4b8480',
        showgrid=False
    ),
    yaxis=dict(
        title='Number of Events',
        color='#4b8480',
        titlefont=dict(
            size=20
        ),
        showline=True,
        linewidth=1,
        linecolor='#4b8480',
        showgrid=False,
        zeroline=False,
    ),
    legend=dict(orientation='h')
)

data1 = [trace0, trace1]

fig1 = go.Figure(data=data1, layout=layout1)

iplot(fig1)

winter_games = ae_winter.groupby(['Year', 'City', 'Sex'])['Event', 'Name']\
                        .agg({'Event': 'nunique', 'Name': 'nunique'})\
                        .reset_index()\
                        .pivot_table(index=['Year', 'City'], columns='Sex', values=['Event', 'Name'])\
                        .fillna(0)\
                        .reset_index()
winter_games.columns = ['Year', 'City', 'Count of Events - Female', 'Count of Events - Male', 'Female Athletes', 'Male Athletes']

winter_games.iloc[:, 2:] = winter_games.iloc[:, 2:].astype(int)

# I will need common scaling for the comparison of count of events
pcount_full = winter_games['Female Athletes'].tolist() + winter_games['Male Athletes'].tolist()

# scaling for bubble chart
scaled_pcount_f = (winter_games['Female Athletes'] - min(pcount_full)) / (max(pcount_full) - min(pcount_full)) + 0.1
scaled_pcount_m = (winter_games['Male Athletes'] - min(pcount_full)) / (max(pcount_full) - min(pcount_full)) + 0.1

# I need to create the text for the labels. There must be better way to do it, I am looking for clues :-) 
text_f = []
for i in range(winter_games.shape[0]):
    text = 'In ' + \
           str(winter_games.iloc[i,0]) + \
           ', in ' + \
           str(winter_games.iloc[i, 1]) + \
           ', <br>' + \
           str(winter_games.iloc[i,4]) + \
           ' Female Athletes <br>' + \
           'competed in ' + \
           str(winter_games.iloc[i,2]) + \
           ' events.'
    text_f.append(text)

text_m = []
for i in range(winter_games.shape[0]):
    text = 'In ' + \
           str(winter_games.iloc[i,0]) + \
           ', in ' + \
           str(winter_games.iloc[i, 1]) + \
           ', <br>' + \
           str(winter_games.iloc[i,5]) + \
           ' Male Athletes <br>' + \
           'competed in ' + \
           str(winter_games.iloc[i,3]) + \
           ' events.'
    text_m.append(text)
    
    
# laying out the bubble chart
trace0 = go.Scatter(
    x=winter_games['Year'],
    y=winter_games['Count of Events - Female'],
    mode='markers',
    marker=dict(
        size=scaled_pcount_f*50,
        symbol='circle-open',
        line=dict(
            width=4
        ),
        color='#2c73d2'
    ),
    name='Female Athletes',
    text=text_f,
    hoverinfo='text'
)
trace1 = go.Scatter(
    x=winter_games['Year'],
    y=winter_games['Count of Events - Male'],
    mode='markers',
    marker=dict(
        size=scaled_pcount_m*50,
        symbol='circle-open',
        line=dict(
            width=4
        ),
        color='#ABD9FF'),
    name='Male Athletes',
    text=text_m,
    hoverinfo='text'
)

layout1 = go.Layout(
    title='Chasing Equality - Winter Olympics',
    titlefont=dict(
        size=36,
        color='#A6ABBD'
    ),
    xaxis=dict(
        title='Year',
        color='#A6ABBD',
        titlefont=dict(
            size=20
        ),
        showline=True,
        linewidth=1,
        linecolor='#A6ABBD',
        showgrid=False
    ),
    yaxis=dict(
        title='Number of Events',
        color='#A6ABBD',
        titlefont=dict(
            size=20
        ),
        showline=True,
        linewidth=1,
        linecolor='#A6ABBD',
        showgrid=False,
        zeroline=False,
    ),
    legend=dict(orientation='h')
)

data1 = [trace0, trace1]

fig1 = go.Figure(data=data1, layout=layout1)

iplot(fig1)

# counting the number of events, in which female and male athletes competed, by year and sport
summer_sports = ae_summer.groupby(['Year', 'Sport', 'Sex'])['Event']\
                         .agg('nunique')\
                         .reset_index()\
                         .pivot_table(index=['Year','Sport'], columns='Sex', values='Event', fill_value=0)\
                         .reset_index()

# assigning a score
summer_sports['Score'] = (summer_sports['F'] - summer_sports['M']) / (summer_sports['F'] + summer_sports['M'])

# final pivot for heatplot
summer_sports_pivot = summer_sports.pivot_table(index='Sport', columns='Year', values='Score').reset_index()

# plot set up
trace2 = go.Heatmap(z=np.array(summer_sports_pivot.iloc[:,1:]),
                    x=summer_sports_pivot.columns.tolist()[1:],
                    y=summer_sports_pivot['Sport'],
                    colorscale=[[0, '#f9f871'], [0.5, '#00d9b3'], [1, '#2c73d2']],
                    xgap=0.9,
                    ygap=0.9,
                    colorbar=dict(
                        title='Equality score',
                        titleside='right',
                        titlefont=dict(
                            size=20,
                            color='#4b8480'),
                        outlinecolor='white',
                        tickcolor='#4b8480',
                        tickfont=dict(color='#4b8480'),
                        tickmode='array',
                        tickvals=[-0.95, 0, 0.95],
                        ticktext=['Only<br>Men', 'Perfectly<br>Equal', 'Only<br>Women']
                    )
                   )

data2 = [trace2]
layout2 = go.Layout(
    title='In summer, men lift weights, women swim(synchronously)',
    titlefont=dict(
        size=24,
        color='#4b8480'
    ),
    height=800,
    #margin=dict(l=180,t=100),
    xaxis=dict(title='Year', 
               titlefont=dict(
                   size=16
               ),
               showgrid=False, 
               color='#4b8480'
              ),
    yaxis=dict(
               color='#4b8480',
               tickfont=dict(
                   size=9
               ),
               automargin=True,
               showgrid=False
              )
)
fig2 = go.Figure(data=data2, layout=layout2)

iplot(fig2)

# counting the number of events, in which female and male athletes competed, by year and sport
winter_sports = ae_winter.groupby(['Year', 'Sport', 'Sex'])['Event']\
                         .agg('nunique')\
                         .reset_index()\
                         .pivot_table(index=['Year','Sport'], columns='Sex', values='Event', fill_value=0)\
                         .reset_index()

# assigning a score
winter_sports['Score'] = (winter_sports['F'] - winter_sports['M']) / (winter_sports['F'] + winter_sports['M'])

# final pivot for heatplot
winter_sports_pivot = winter_sports.pivot_table(index='Sport', columns='Year', values='Score').reset_index()

# plot set up
trace2 = go.Heatmap(z=np.array(winter_sports_pivot.iloc[:,1:]),
                    x=winter_sports_pivot.columns.tolist()[1:],
                    y=winter_sports_pivot['Sport'],
                    colorscale=[[0, '#ABD9FF'], [0.5, '#76a5e9'], [1, '#2c73d2']],
                    xgap=0.9,
                    ygap=0.9,
                    colorbar=dict(
                        title='Equality score',
                        titleside='right',
                        titlefont=dict(
                            size=20,
                            color='#A6ABBD'),
                        outlinecolor='white',
                        tickcolor='#A6ABBD',
                        tickfont=dict(
                            color='#A6ABBD'
                        ),
                        tickmode='array',
                        tickvals=[-0.95, 0, 0.95],
                        ticktext=['Only<br>Men', 'Perfectly<br>Equal', 'Only<br>Women']
                    ),
                    zmin=-1,
                    zmax=1
                   )

data2 = [trace2]
layout2 = go.Layout(
    title='But in winter, everyone skates or skiis',
    titlefont=dict(
        size=24,
        color='#A6ABBD'
    ),
    height=800,
    #margin=dict(l=180,t=100),
    xaxis=dict(title='Year', 
               titlefont=dict(
                   size=16
               ),
               showgrid=False, 
               color='#A6ABBD'
              ),
    yaxis=dict(
               color='#A6ABBD',
               tickfont=dict(
                   size=12
               ),
               automargin=True,
               showgrid=False
              )
)
fig2 = go.Figure(data=data2, layout=layout2)

iplot(fig2)

# step 2
summer_athl = ae_summer.groupby(['region_filled', 'Sex'])['Name']\
                      .count()\
                      .reset_index()\
                      .pivot_table(index='region_filled',
                                   columns='Sex',
                                   values='Name',
                                   fill_value=0)\
                      .reset_index()
summer_athl['Total Athletes'] = summer_athl['F'] + summer_athl['M']
col_rename = {'F': 'Female Athletes', 'M': 'Male Athletes'}
summer_athl = summer_athl.rename(columns=col_rename)

# step 3
summer_med = ae_summer.dropna(subset=['Medal'])
summer_med_pv = summer_med.groupby(['region_filled','Sex'])['Medal']\
                          .count()\
                          .reset_index()\
                          .pivot_table(index='region_filled',
                                       columns='Sex',
                                       values='Medal',
                                       fill_value=0)\
                          .reset_index()
summer_med_pv['Total'] = summer_med_pv['F'] + summer_med_pv['M']

# step 4
summer_stats = summer_athl.merge(summer_med_pv, how='left')\
                          .fillna(0)
summer_stats_col_rename = {'F': 'Female Medals',
                           'M': 'Male Medals',
                           'Total': 'Total Medals'}
summer_stats = summer_stats.rename(columns=summer_stats_col_rename)
summer_stats['MpFA'] = summer_stats['Female Medals'] / summer_stats['Female Athletes']
summer_stats['MpMA'] = summer_stats['Male Medals'] / summer_stats['Male Athletes']
summer_stats['MpA'] = summer_stats['Total Medals'] / summer_stats['Total Athletes']

# step 5
top_20_med_summer = summer_stats\
                    .sort_values(by='Total Medals', ascending=False)\
                    .iloc[:20, :]\
                    .sort_values(by='MpA')

###--plot--###
trace3 = go.Bar(x=top_20_med_summer['MpFA'],
                y=top_20_med_summer['region_filled'],
                name='Women',
                marker=dict(color='#2c73d2'),
                orientation='h')

trace4 = go.Bar(x=top_20_med_summer['MpMA'],
                y=top_20_med_summer['region_filled'],
                name='Men',
                marker=dict(color='#f9f871'),
                orientation='h')

layout3 = go.Layout(
    title='Medals per Athlete - Summer Games<br>Top 20 Countries',
    font=dict(
        color='#4b8480'
    ),
    titlefont=dict(
        size=28
    ),
    height=800,
    xaxis=dict(
        title='Medals per Athlete', 
        color='#4b8480',
        titlefont=dict(
            size=16
        ), 
        showline=True,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(color='#4b8480',showgrid=False, showline=False, automargin=True, ticksuffix='  ')
)

data3 = [trace4,trace3]
fig3 = go.Figure(data=data3, layout=layout3)
iplot(fig3)
winter_athl = ae_winter.groupby(['region_filled', 'Sex'])['Name']\
                      .count()\
                      .reset_index()\
                      .pivot_table(index='region_filled',
                                   columns='Sex',
                                   values='Name',
                                   fill_value=0)\
                      .reset_index()
winter_athl['Total Athletes'] = winter_athl['F'] + winter_athl['M']
col_rename = {'F': 'Female Athletes', 'M': 'Male Athletes'}
winter_athl = winter_athl.rename(columns=col_rename)

# step 3
winter_med = ae_winter.dropna(subset=['Medal'])
winter_med_pv = winter_med.groupby(['region_filled','Sex'])['Medal']\
                          .count()\
                          .reset_index()\
                          .pivot_table(index='region_filled',
                                       columns='Sex',
                                       values='Medal',
                                       fill_value=0)\
                          .reset_index()
winter_med_pv['Total'] = winter_med_pv['F'] + winter_med_pv['M']

# step 4
winter_stats = winter_athl.merge(winter_med_pv, how='left')\
                          .fillna(0)
winter_stats_col_rename = {'F': 'Female Medals',
                           'M': 'Male Medals',
                           'Total': 'Total Medals'}
winter_stats = winter_stats.rename(columns=winter_stats_col_rename)
winter_stats['MpFA'] = winter_stats['Female Medals'] / winter_stats['Female Athletes']
winter_stats['MpMA'] = winter_stats['Male Medals'] / winter_stats['Male Athletes']
winter_stats['MpA'] = winter_stats['Total Medals'] / winter_stats['Total Athletes']

# step 5
top_20_med_winter = winter_stats\
                    .sort_values(by='Total Medals', ascending=False)\
                    .iloc[:20, :]\
                    .sort_values(by='MpA')

###--plot--###
trace5 = go.Bar(x=top_20_med_winter['MpFA'],
                y=top_20_med_winter['region_filled'],
                name='Women',
                marker=dict(color='#2c73d2'),
                orientation='h')

trace6 = go.Bar(x=top_20_med_winter['MpMA'],
                y=top_20_med_winter['region_filled'],
                name='Men',
                marker=dict(color='#ABD9FF'),
                orientation='h')

layout4 = go.Layout(
    title='Medals per Athlete - Winter Games<br>Top 20 Countries',
    font=dict(
        color='#A6ABBD'
    ),
    titlefont=dict(
        size=28
    ),
    height=800,
    xaxis=dict(
        title='Medals per Athlete', 
        color='#A6ABBD',
        titlefont=dict(
            size=16
        ), 
        showline=True,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        color='#A6ABBD',
        showgrid=False,
        showline=False,
        automargin=True,
        ticksuffix='  ')
)

data4 = [trace6,trace5]
fig4 = go.Figure(data=data4, layout=layout4)
iplot(fig4)
# let's see how many medals of each color each athlete earned
# we can also check, how many events they participated in with no medal at all
fmedals = fae.groupby(['Name_Region', 'Medal'])['Event']\
             .count()\
             .reset_index()\
             .pivot_table(index='Name_Region',
                          columns='Medal',
                          values='Event',
                          fill_value=0)\
             .reset_index()

# let's reorder columns and add Total as the sum of Bronze, Silver, Gold
fmedals['Total'] = fmedals['Bronze'] + fmedals['Silver'] + fmedals['Gold']
fmedals = fmedals[['Name_Region', 'No Medal', 'Bronze', 'Silver', 'Gold', 'Total']]

# take only ones with 10 or more medals
f10meds = fmedals[fmedals['Total'] >= 10].sort_values(by='Total').copy()

# now onto the plot
data_m = []

for i in range(f10meds.shape[0]):
    trace = go.Scatter(
        x=list(range(1, f10meds.iloc[i,5]+1)),
        y=[f10meds.iloc[i,0]] * f10meds.iloc[i,5],
        mode='markers',
        marker=dict(
            size=30,
            color=['#A48E65'] * f10meds.iloc[i,2]+\
                  ['#DFE0DF'] * f10meds.iloc[i,3]+\
                  ['#D5A419'] * f10meds.iloc[i,4],
            line=dict(color='white',
                      width=1)
        ),
        showlegend=False,
        text=str(f10meds.iloc[i,0]) +\
        '<br>won ' +\
        str(f10meds.iloc[i, 5]) +\
        ' medals in total.',
        hoverinfo='text'
    )
    data_m.append(trace)

layout_m = go.Layout(
    title='Top Olympic Medalists - Women',
    titlefont=dict(
        size=24,
        color='#4b8480'
    ),
    height=800, 
    hovermode='closest', 
    yaxis=dict(
        color='#4b8480',
        automargin=True,
    ), 
    xaxis=dict(
        color='#4b8480',
        showgrid=False,
        zeroline=False))
fig_m = go.Figure(data=data_m, layout=layout_m)
iplot(fig_m, show_link=False)
# let's take the subset of top medalists to contain data necessary for plotting
f10meds_full = fae[fae['Name_Region'].isin(f10meds['Name_Region'])]
f10meds_full = f10meds_full[['Name_Region',
                             'Season',
                             'Age',
                             'Year',
                             'City',
                             'Event',
                             'Medal']]

# for better display, let's do this:
def insert_break(s):
    '''
    This function inserts break point in the
    middle of the Event's name.
    arg: string
    returns: string with breakpoint inserted
    '''
    split_s = s.split(' ')
    split_s.insert(int(len(split_s)/2), '<br>')
    s = ' '.join(split_s)
    return s

# then, let's map the medals to colors
medals_map = {'No Medal': '#2c73d2',
              'Bronze': '#A48E65',
              'Silver': '#DFE0DF',
              'Gold' : '#D5A419'}

f10meds_full['Event_br'] = f10meds_full['Event'].apply(insert_break)
f10meds_full['Medal_clr'] = f10meds_full['Medal'].map(medals_map)

# for hover text
text_all = []
for i in range(f10meds_full.shape[0]):
    text = 'In ' +\
        f10meds_full.iloc[i, 1] +\
        ' Games in ' +\
        f10meds_full.iloc[i, 4] +\
        ' (' +\
        str(f10meds_full.iloc[i, 3]) +\
        '),<br>' +\
        f10meds_full.iloc[i, 0] +\
        '<br>won ' +\
        f10meds_full.iloc[i, 6] +\
        ' in<br>' +\
        f10meds_full.iloc[i, 7] +\
        '<br>She was ' +\
        str(f10meds_full.iloc[i, 2]) +\
        ' at the time.'
    text_all.append(text)
f10meds_full['Text'] = text_all

# the main loop iterates over the dataframe with top medalists and creates figure elements
data_m = []
buttons_m = []

for i in range(f10meds.shape[0]):
    subset = f10meds_full[f10meds_full['Name_Region'] == str(f10meds.iloc[i,0])]
    x_ax_ticks = pd.unique(subset['Year'])
    # I need to do below, because I need to display something first, without user interaction
    # looking for help here ;-)
    
    if i == f10meds.shape[0] - 1:
        trace_m = go.Scatter(
            x=subset['Year'],
            y=subset['Event_br'],
            mode='markers',
            marker=dict(size=30, color=subset['Medal_clr']),
            text=subset['Text'],
            hoverinfo='text',
            visible=True
            )
    else:
        trace_m = go.Scatter(
            x=subset['Year'],
            y=subset['Event_br'],
            mode='markers',
            marker=dict(size=30, color=subset['Medal_clr']),
            text=subset['Text'],
            hoverinfo='text',
            visible=False
            )
        
    data_m.append(trace_m)
    
    visible_traces = np.full(f10meds.shape[0], False)
    visible_traces[i] = True
    
    button_m = dict(
        label=str(f10meds.iloc[i,0]),
        method= 'update',
        args=[
            dict(
                visible=visible_traces
            ),
            dict(
                xaxis=dict(
                    tickvals=x_ax_ticks,
                    showgrid=False,
                    tickfont=dict(
                        size=14
                    )
                )
            )
        ]
    )
    buttons_m.append(button_m)
    
    updatemenus = [
        dict(
            buttons=buttons_m,
            active=f10meds.shape[0] - 1,
            showactive=True,
            x=0.5,
            xanchor='center',
            y=1.1,
            yanchor='top'
            )
        ]

layout_m = go.Layout(
    title='Top medalists - breakdown',
    height=600,
    font=dict(
        color='#4b8480'
        ),
    titlefont=dict(
        size=24
        ),
    xaxis=dict(
        tickvals=x_ax_ticks,
        showgrid=False,
        tickfont=dict(size=14)
        ),
    yaxis=dict(
        automargin=True
        ),
    hovermode='closest',
    updatemenus=updatemenus
)

fig = go.Figure(data=data_m, layout=layout_m)

    
iplot(fig, show_link=False)
# first, let's compute the averages - I am discarding the NaNs
f_avg_age = fae.groupby('Year')['Age'].mean().reset_index()

# same for medalists
f_avg_age_m = fae[fae['Medal']!='No Medal'].groupby('Year')['Age'].mean().reset_index()

# texts for hovers
f_avg_age_text_all = []
f_avg_age_text_meds = []
for i in range(len(f_avg_age)):
    text_all = 'In ' +\
            str(f_avg_age.iloc[i, 0]) +\
            ' average age<br>of female athlete was ' +\
            str(round(f_avg_age.iloc[i, 1], 2))
    text_medals = 'In ' +\
                str(f_avg_age_m.iloc[i, 0]) +\
                ' average age<br>of female medalist was ' +\
                str(round(f_avg_age_m.iloc[i, 1], 2))
    f_avg_age_text_all.append(text_all)
    f_avg_age_text_meds.append(text_medals)
    
f_avg_age['Text'] = f_avg_age_text_all
f_avg_age_m['Text'] = f_avg_age_text_meds

# and plot it
trace_faa = go.Scatter(
    x=f_avg_age['Year'],
    y=f_avg_age['Age'],
    mode='lines+markers',
    text=f_avg_age['Text'],
    hoverinfo='text', 
    marker=dict(
        color='#2c73d2'
    ),
    name='All Athletes')

trace_faam = go.Bar(
    x=f_avg_age_m['Year'],
    y=f_avg_age_m['Age'],
    text=f_avg_age_m['Text'],
    hoverinfo='text',
    marker=dict(color='#D5A419'), name='Only Medalists'
)

layout_faa = go.Layout(
    title='Mean Age of Female Athletes<br>Over Time',
    titlefont=dict(
        size=24
        ),
    font=dict(
        color='#4b8480'
        ),
    xaxis=dict(
        title='Year'
        ), 
    yaxis=dict(
        color='#4b8480',
        title='Average Age',
        showline=True
        ), 
    annotations=[
        dict(
            x=1940,
            y= 40,
            text='With minor exceptions, up until 1952<br>\
                average age of medalist was much lower<br>\
                (take a look at 1932!)<br>\
                from the average age in full population',
            font=dict(
                size=10
            ),
            showarrow=False
        ),
        dict(
            x=1990,
            y= 40,
            text='But from 1952 until recently<br>\
                average medalist was over year above<br>\
                the mean for the full population<br>\
                with growing mean age overall',
            font=dict(
                size=10
            ),
            showarrow=False
        )
    ],
    legend=dict(orientation='h')
)

data_faa = [trace_faa, trace_faam]
fig_faa = go.Figure(data=data_faa, layout=layout_faa)

iplot(fig_faa, show_link=False)

# whole population
trace_b_faa = go.Box(
    x=fae['Age'],
    marker=dict(
        color='#2c73d2'
        ),
    name='All<br>Athletes',
    showlegend=False
)

# only medalists
trace_b_faam = go.Box(
    x=fae[fae['Medal']!='No Medal']['Age'],
    marker=dict(
        color='#D5A419'
        ), 
    name='Only<br>Medalists',
    showlegend=False
)

data_b = [trace_b_faam, trace_b_faa]
layout_b = go.Layout(
    title='Do you expect to win an Olympic Medal at 69?',
    titlefont=dict(size=24),
    font=dict(
        color='#4b8480'
        ),
    xaxis=dict(
        title='Average Age',
        tickfont=dict(
            size=14
        ),
        automargin=True
    ),
    annotations=[
        dict(
            x=12,
            y=0.3,
            text='The youngest<br>\
                Olympic Medalist<br>\
                was only 11!',
            font=dict(
                size=10
            ),
            showarrow=False
        ),
        dict(
            x=26,
            y=0.5,
            text='Average Medalist has been<br>\
                a year older than<br>\
                the Average Athlete at Olympics',
            font=dict(size=10),
            showarrow=False),
        dict(
            x=67,
            y=0.2,
            text='The oldest<br>\
                Olympic Medalist<br>\
                was 69 years old!',
            font=dict(size=10),
            showarrow=False)]
)
fig_b = go.Figure(data=data_b, layout=layout_b)
iplot(fig_b)
fmedalists = fae[fae['Medal']!='No Medal']
trace_h = go.Histogram(x=fmedalists['Age'], marker=dict(color='#D5A419'))
layout_h = go.Layout(
    title='Distribution of Female Medalists\'s Age',
    titlefont=dict(
        size=24
        ),
    font=dict(
        color='#4b8480'
        ),
    xaxis=dict(
        title='Age',
        color='#4b8480',
        titlefont=dict(
            size=14
            )
        ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            size=14
            ),
        zeroline=False
        ),
    bargap=0.1
)
data_h = [trace_h]
fig_h = go.Figure(data=data_h, layout=layout_h)
iplot(fig_h)
fmedalists[(fmedalists['Age']==11) | (fmedalists['Age']==69)][['Name', 'Age', 'Year', 'City', 'Sport', 'Event', 'Medal', 'region']]