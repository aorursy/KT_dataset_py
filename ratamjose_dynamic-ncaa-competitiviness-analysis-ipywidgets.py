#data analysis
import pandas as pd
import numpy as np

#visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import ipywidgets as widgets
import plotly.offline as py

py.init_notebook_mode(connected=True)


#statistics
from scipy.stats import ttest_ind

#complementary libraries
import os
import re

#custom settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!jupyter nbextension enable --py --sys-prefix widgetsnbextension
my_title = dict(
    font = dict(
        family = 'verdana',
        size = 25,
        color = 'gray'
    )
)

my_margins = dict(
        t = 70,
        b = 0,
        l = 0,
        r = 0
    )

my_xaxis = dict(
    showline = True,
    linecolor = 'black'
)

my_yaxis = dict(
    showline = True,
    linecolor = 'black'
)

base_layout = dict(
    title = my_title,
    margin = my_margins,
    colorway = px.colors.sequential.Greys,
    paper_bgcolor = 'white',
    plot_bgcolor = 'white',
    xaxis = my_xaxis, 
    yaxis = my_yaxis, 
)
base_fig = go.Figure(
    layout = base_layout
)
template_fig = pio.to_templated(base_fig)
pio.templates['ncaa'] = template_fig.layout.template
pio.templates.default = 'ncaa'
games = pd.read_csv('/kaggle/input/tidy-ncaa-dat/tidy_ncaa.csv', low_memory=False)
games.drop('Unnamed: 0', axis=1, inplace=True)
# date and dayzero to datetime
games['date'] = pd.to_datetime(games['date'])
games['dayzero'] = pd.to_datetime(games['dayzero'])
#loading teams data
wd = '/kaggle/input/ncaa-teams'

mteams = pd.read_csv(os.path.join(wd, 'MTeams.csv'))
mteams.columns = mteams.columns.str.lower()
mteams.drop(['firstd1season', 'lastd1season'], axis=1, inplace=True)
games = games.merge(mteams, on='teamid')
# number of teams per season 
mask = (games.identifier != 'secondary')
teams_per_season = games.loc[mask, :].groupby(['season'])['teamid'].nunique()

ifig = go.Figure()
trace = go.Scatter(
    x = teams_per_season.index,
    y = teams_per_season.values,
    marker_color = 'gray',
    line = dict(width=1)
)
ifig.add_trace(trace)

ifig.update_layout(
    title = dict(
        text = 'TEAMS BY SEASON',
        font = dict(
            family='verdana',
            size = 25,
            color = 'gray'
        )
        
    ),
    paper_bgcolor = 'white',
    plot_bgcolor = 'white',
    xaxis = dict(
        showline = True,
        linecolor = 'black',
    ),
    yaxis= dict(
        range=[teams_per_season.min(), teams_per_season.max() * 1.1],
        showline=True,
        linecolor = 'black'
    ),
    
)

ifig.add_shape(
    dict(
        type = 'line',
        x0 = 2009,
        x1 = 2019,
        y0 = teams_per_season.loc[2009:2019].mean(),
        y1 = teams_per_season.loc[2009:2019].mean(),
        line = dict(
            color = 'steelblue',
            width = 6,
            dash = 'dot'
        )
    )
)
ifig.add_annotation(
    x = 2014,
    y = teams_per_season.loc[2009:2019].mean() * 1.02,
    xref = 'x',
    yref = 'y',
    showarrow = False,
    text = f'Last 10 year Avg: <b>{teams_per_season.loc[2009:2019].mean()}',
    font = dict(
        family = 'verdana',
        size = 16,
        color = 'steelblue'
    )
    
)
ifig.show()
mask = (games.identifier != 'secondary') & (games.identifier == 'tourney')
teams_per_tourney = games.loc[mask, :].groupby('season')['teamid'].nunique()
trace = go.Scatter(
   x = teams_per_tourney.index,
    y = teams_per_tourney.values,
    marker_color = 'steelblue'
)
layout = dict(
    title = dict(
        text = 'TEAMS IN TOURNEY',
        font = dict( family = 'verdana', size = 25, color = 'gray'),
    ),
)
ifig2 = go.Figure(
    data = trace,
    layout = layout
)
annot_years = [1993, 2005, 2015]
teamval_unique = teams_per_tourney.unique()
for year, value in zip(annot_years, teamval_unique):
    ifig2.add_annotation(
        text = f'{value} teams',
        font = dict(color='gray', family='verdana', size=16),
        x = year,
        y = value,
        xanchor = 'center',
        yanchor = 'bottom',
        showarrow=False
    )
ifig2.show()
# dividing teams in two groups
regular_mask = games.final_stage == 'regular'
worst_teams = games[regular_mask].groupby('teamname')['season'].nunique().sort_values(ascending=False)
worst_team_index = worst_teams[worst_teams==35].index

# best teams
bestteams_mask = games.final_stage == 'Nacional Final'
best_teams = games.loc[bestteams_mask].groupby('teamname')['season'].nunique().sort_values().nlargest(5)

best_teams_attrs = games.set_index('teamname').loc[best_teams.index, :]
worst_teams_attrs = games.set_index('teamname').loc[worst_team_index, :]
worst = worst_teams[worst_teams == 35]
best = best_teams.sort_values()
ifig3 = make_subplots(rows=1,cols=2, horizontal_spacing=0.5 )
trace = go.Bar(
    y = best.index,
    x = best.values,
    orientation = 'h',
    marker_color = 'steelblue',
    texttemplate = '%{x} of 35',
    textposition = 'inside',
    showlegend = False
)
ifig3.add_trace(trace, 1,1)

trace1 = go.Bar(
    y = worst.index,
    x = worst.values,
    orientation = 'h',
    marker_color = 'gray',
    texttemplate = '%{x} of 35',
    textposition = 'inside',
    hovertemplate = '%{x} of 35',
    showlegend = False
)
ifig3.add_trace(trace1, 1,2)

ifig3.add_annotation(
    text = 'FINAL STAGE: <b>NATIONAL FINAL</b>',
    font = dict(
        family = 'verdana',
        size = 20,
        color = 'gray'
    ),
    showarrow = False,
    x = 0.225,
    xref = 'paper',
    xanchor = 'center',
    y = 1,
    yref = 'paper',
    yanchor = 'bottom',
)

ifig3.add_annotation(
    text = 'FINAL STAGE: <b>REGULAR</b>',
    font = dict(
        family = 'verdana',
        size = 20,
        color = 'gray'
    ),
    showarrow = False,
    x = 0.775,
    xref = 'paper',
    xanchor = 'center',
    y = 1,
    yref = 'paper',
    yanchor = 'bottom',
    
    
)
ifig3.show()
season_slider = widgets.Dropdown(
    options = games.season.sort_values().unique(),
    value = 2015,
    description = 'Season: '
)
numcols = ['score','score_diff', 'ftm', 'fta', 'or','dr', 'ast',
           'stl', 'blk','fgm', 'fga', 'fgm3', 'fga3', 'to',  'pf', 'cum_loses']
firstvar_slider = widgets.Dropdown(
    options = numcols,
    value = 'score',
    description = 'First variable: '
)
numcols2 = ['score','score_diff', 'ftm', 'fta', 'or','dr', 'ast',
           'stl', 'blk','fgm', 'fga', 'fgm3', 'fga3', 'to',  'pf', 'lose_status']
secondvar_slider = widgets.Dropdown(
    options = numcols2,
    value = 'score',
    description = 'second variable: '
)
bestteams = best_teams_attrs.index.unique()
bestteam_slider = widgets.Dropdown(
    options = bestteams,
    value = 'Duke',
    description = 'Best: '
)
worstteams = worst_teams_attrs.index.unique()
worsteam_slider = widgets.Dropdown(
    options = worstteams,
    value = 'Youngstown St',
    description = 'Worst: '
)
my_specs = [
    [{}],
    [{'rowspan':3}],
    [None],
    [None],
]

fig2 = go.FigureWidget(make_subplots(rows=4, cols=1, specs =my_specs))
trace = go.Box(
    x = best_teams_attrs['score'],
    name = 'Strong - score',
    legendgroup = 'Strong',
    marker_color = 'steelblue'
)
trace1 = go.Box(
    x = worst_teams_attrs['score'],
    name = 'Weak - score',
    legendgroup = 'Weak',
    marker_color = 'gray'
)
fig2.add_trace(trace, 1,1)
fig2.add_trace(trace1, 1,1)

trace3 = go.Histogram(
    x = best_teams_attrs['score'],
    name = 'Strong - score',
    legendgroup = 'Strong',
    marker_color = 'steelblue',
    opacity = 0.55,
    histnorm = 'probability density',
    showlegend = False
)
trace4 = go.Histogram(
    x = worst_teams_attrs['score'],
    name = 'Weak - score',
    legendgroup = 'Weak',
    marker_color = 'gray',
    opacity = 0.55,
    histnorm = 'probability density',
    showlegend = False
)

fig2.add_trace(trace3, 2,1)
fig2.add_trace(trace4, 2,1)
fig2.update_layout(
    barmode = 'overlay'
)
def update_boxplots(a):
    for data in fig2.data:
        if data.marker.color == 'steelblue':
            data.x = best_teams_attrs[firstvar_slider.value]
            data.name = f'Strong - {firstvar_slider.value}'
        else:
            data.x = worst_teams_attrs[firstvar_slider.value]
            data.name = f'Weak - {firstvar_slider.value}'
            
firstvar_slider.observe(update_boxplots, names='value')
widgets.VBox([firstvar_slider, fig2])
def drop_outliers(s):
    q3 = s.quantile(.75)
    q1 = s.quantile(.25)
    iqr = q3 - q1
    upper_fence = q3+(1.5*iqr) 
    lower_fence = q1-(1.5*iqr) 
    return pd.Series(np.where(s.between(lower_fence, upper_fence), s, np.nan))

def ttest_teams(best, worst, col, alpha, sample_size):
    best_sample = best[col].dropna().sample(sample_size)
    worst_sample = worst[col].dropna().sample(sample_size)
    stat, p = ttest_ind(best_sample, worst_sample, equal_var=False)
    if p > alpha:
        status = 'No rejected'
    else:
        status = 'Rejected'
    return p, stat, status, col

best_no_outliers = best_teams_attrs[numcols].apply(drop_outliers)
worst_no_outliers = worst_teams_attrs[numcols].apply(drop_outliers)


df = dict(p_value=[], stat=[], test_status=[], attr=[])

for number_col in numcols:
    test_values = ttest_teams(best_teams_attrs, worst_teams_attrs,number_col ,0.05, 1500)
    df['p_value'].append(test_values[0])
    df['stat'].append(test_values[1])
    df['test_status'].append(test_values[2])
    df['attr'].append(test_values[3])
    
ttest_df = pd.DataFrame(df).set_index('attr').sort_index()
test_status = ttest_df.loc['score', 'test_status']
test_pvalue = ttest_df.loc['score', 'p_value']

fig3 = go.FigureWidget()
trace6 = go.Histogram(
    x = best_no_outliers['score'],
    name = 'Strong - score',
    legendgroup = 'Strong',
    marker_color = 'steelblue',
    opacity = 0.55,
    histnorm = 'probability density',
    showlegend = False
)
trace7 = go.Histogram(
    x = worst_no_outliers['score'],
    name = 'Weak - score',
    legendgroup = 'Weak',
    marker_color = 'gray',
    opacity = 0.55,
    histnorm = 'probability density',
    showlegend = False
)
fig3.add_trace(trace6)
fig3.add_trace(trace7)
fig3.update_layout(
    barmode = 'overlay',
    title = dict(
        text = '<b>STRONGEST</b> VS <b>WEAKEST</b> TEAMS',
        font = dict(size = 25, color = 'gray', family='verdana')
    )
)
fig3.add_shape(
    dict(
        type = 'line',
        line = dict(width=4, color = 'darkblue', dash='dot'),
        x0 = best_no_outliers['score'].mean(),
        x1 = best_no_outliers['score'].mean(),
        yref = 'paper',
        y0 = 0,
        y1 = 1,
    )
)
fig3.add_shape(
    dict(
        type = 'line',
        line = dict(width=4, color = 'black', dash='dot'),
        x0 = worst_no_outliers['score'].mean(),
        x1 = worst_no_outliers['score'].mean(),
        yref = 'paper',
        y0 = 0,
        y1 = 1,
    )
)
fig3.add_annotation(
    text = f'<b>TWO TAILED T TEST</b><br>Confedence interval: 95%<br>P-value: {test_pvalue}<br><b>Ho: {test_status}</b>',
    font = dict(color = 'steelblue', family = 'verdana', size = 20),
    x = 0.85,
    xref = 'paper',
    y = 0.65,
    yref = 'paper',
)
def update_ttest(a):
    for data in fig3.data:
        if data.marker.color == 'steelblue':
            data.x = best_no_outliers[firstvar_slider.value]
            data.name = f'Strong - {firstvar_slider.value}'
        else:
            data.x = worst_no_outliers[firstvar_slider.value]
            data.name = f'Weak - {firstvar_slider.value}'
    
    #updating shapes
    fig3.layout.shapes[0].x0 = best_no_outliers[firstvar_slider.value].mean()
    fig3.layout.shapes[0].x1 = best_no_outliers[firstvar_slider.value].mean()
    
    fig3.layout.shapes[1].x0 = worst_no_outliers[firstvar_slider.value].mean()
    fig3.layout.shapes[1].x1 = worst_no_outliers[firstvar_slider.value].mean()
    
    #updating annotations
    test_status = ttest_df.loc[firstvar_slider.value, 'test_status']
    test_pvalue = ttest_df.loc[firstvar_slider.value, 'p_value']
    fig3.layout.annotations[0].text = f'<b>TWO TAILED T TEST</b><br>Confedence interval: 95%<br>P-value: {test_pvalue}<br><b>Ho: {test_status}</b>'
    
firstvar_slider.observe(update_ttest, names='value')
widgets.VBox([firstvar_slider, fig3])
mask0 = (best_teams_attrs.index == 'Duke') & (best_teams_attrs.season == 2015)
data0 = best_teams_attrs.loc[mask0, ['season', 'games_played', 'score']]\
        .sort_values(['season', 'games_played']).dropna(subset = ['games_played'])

mask1 = (worst_teams_attrs.index == 'Bowling Green') & (worst_teams_attrs.season == 2015)
data1 = worst_teams_attrs.loc[mask1, ['season', 'games_played', 'score']]\
        .sort_values(['season', 'games_played']).dropna(subset=['games_played'])

fig = go.FigureWidget()
trace = go.Scatter(
    x = [data0['season'], data0['games_played']],
    y = data0['score'].cumsum(),
    mode = 'lines',
    marker_color = 'steelblue',
    name = f'Strong - Duke',
    showlegend = False,
    line = dict(width = 4)
)

trace1 = go.Scatter(
    x = [data1['season'], data1['games_played']],
    y = data1['score'].cumsum(),
    mode = 'lines',
    marker_color = 'gray',
    name = f'Weak - Youngstown St',
    showlegend = False,
    line = dict(width = 2)
)
fig.add_shape(
    dict(
        type='line',
        x0 = 0,
        y0 = 0,
        x1 = 33,
        y1 = 0,
        line = dict(
            color = 'gray',
            width = 2,
            dash = 'dot'
        )
    )
)
fig.add_trace(trace)
fig.add_trace(trace1)
fig.add_annotation(
   x = [data0['season'].max(), data0['games_played'].max()],
    xref = 'x',
    xanchor = 'left',
    y = data0['score'].cumsum().iloc[-1],
    yref = 'y',
    yanchor = 'middle',
    showarrow = False,
    text = 'Strong - Duke',
    font = dict(family = 'verdana', size = 16, color = 'steelblue')
)
fig.add_annotation(
   x = [data1['season'].max(), data1['games_played'].max()],
    y = data1['score'].cumsum().iloc[-1],
    xref = 'x',
    yref = 'y',
    showarrow = False,
    text = 'Weak - Youngstown St',
    xanchor = 'left',
    yanchor = 'middle',
    font = dict(family = 'verdana', size = 16, color = 'gray')
)
fig.update_layout(
    title = dict(
        text = '<b>STRONGEST</b> VS <b>WEAKEST</b> TEAMS BY SEASON',
        font = dict(family = 'verdana', size = 25, color = 'gray')
    )
)


def update_bestworst_linechart(a):
    mask0 = (best_teams_attrs.index == bestteam_slider.value ) & (best_teams_attrs.season == season_slider.value)
    data0 = best_teams_attrs.loc[mask0, ['season', 'games_played', secondvar_slider.value]]\
            .sort_values(['season', 'games_played']).dropna(subset=['games_played'])

    mask1 = (worst_teams_attrs.index == worsteam_slider.value) & (worst_teams_attrs.season == season_slider.value)
    data1 = worst_teams_attrs.loc[mask1, ['season', 'games_played', secondvar_slider.value]]\
            .sort_values(['season', 'games_played']).dropna(subset=['games_played'])
    
    #updating plots
    fig.data[0].x = [data0['season'], data0['games_played']]
    fig.data[0].y = data0[secondvar_slider.value].cumsum()
    fig.data[0].name = f'Strong {bestteam_slider.value}'

    fig.data[1].x = [data1['season'], data1['games_played']]
    fig.data[1].y = data1[secondvar_slider.value].cumsum()
    fig.data[1].name = f'Weak {worsteam_slider.value}'
    #updating annotations
    fig.layout.annotations[0].x = [season_slider.value, data0['games_played'].max()]
    fig.layout.annotations[0].y = data0[secondvar_slider.value].cumsum().iloc[-1]
    fig.layout.annotations[0].text = f'Strong {bestteam_slider.value}'

    fig.layout.annotations[1].x = [season_slider.value, data1['games_played'].max()]
    fig.layout.annotations[1].y = data1[secondvar_slider.value].cumsum().iloc[-1]
    fig.layout.annotations[1].text = f'Weak {worsteam_slider.value}'
    
season_slider.observe(update_bestworst_linechart, names='value')
bestteam_slider.observe(update_bestworst_linechart, names='value')
secondvar_slider.observe(update_bestworst_linechart, names='value')
worsteam_slider.observe(update_bestworst_linechart, names='value')
sliders = widgets.HBox([season_slider, bestteam_slider, worsteam_slider, secondvar_slider])
widgets.VBox([sliders, fig])