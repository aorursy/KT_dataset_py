import pandas as pd
import numpy as np
import os

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, FuncTickFormatter, CustomJS, Select
from bokeh.layouts import row, column

output_notebook()

pd.set_option('display.max_rows', 20)
print(os.listdir('../input'))
def load_df(df_path):
    if not os.path.exists(df_path):
        df_path = os.path.join('../input', df_path)
    df = pd.read_csv(df_path)
    if 'Season' in df.columns:
        df = df[df['Season'] >= 2003]
    return df

def make_names(names):
    return [' '.join(name.split('_')).title() for name in names]
tourney_compact_results = load_df('NCAATourneyCompactResults.csv')
team_names = load_df('Teams.csv')
print(tourney_compact_results)
print(team_names)
all_tourney_teams = pd.concat([tourney_compact_results[col] for col in ['WTeamID', 'LTeamID']])
num_unique_teams = len(all_tourney_teams.unique())
num_appearances_by_team = all_tourney_teams.value_counts()
num_appearances_by_team.index = team_names.set_index('TeamID')['TeamName'].loc[num_appearances_by_team.index]

p = figure(plot_width=1100,
           title='Number of Appearances by Team',
           x_range=(-0.5, num_unique_teams+0.5),
           y_range=(0, 53))
p.xaxis.ticker = [i for i in range(num_unique_teams)]
p.xaxis[0].formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
""" % list(num_appearances_by_team.index))
p.xaxis.major_label_orientation = np.pi / 2
p.xaxis.major_label_text_font_size = '4pt'

ds = ColumnDataSource({'x': np.arange(num_unique_teams),
                       'y': num_appearances_by_team.values,
                       'name': num_appearances_by_team.index})
p.vbar(x='x', top='y', bottom=0, width=0.6, source=ds, alpha=0.9, color="#3a94e9", hover_color="#ea5d4d")
p.add_tools(HoverTool(tooltips=[('Team Name', '@name'), ('Games Played', '@y')]))
show(p)
# let's do some analysis on coaches
coaches = load_df('TeamCoaches.csv')
tourney_coaches = coaches[coaches['LastDayNum'] >= 134] # only focus on coaches who played in the tournament
num_plot_coaches = 25 # there's a lot of coaches, so let's focus on the top 25 in terms of average point differential

def get_average_coach_differential(coaches_df, results_df):
    '''
    utility function for computing the mean point differential for the coaches in the "CoachName" column
    of `coaches_df` across the tournament games played in `results_df`. Output is a dataframe where the
    index specifies a coach name, and the only column "diff" specifies the average differential
    '''
    w_team_scores = results_df.set_index(['WTeamID', 'Season'])[['LScore', 'WScore']]
    w_point_differential = w_team_scores.diff(axis=1)['WScore'].to_frame()

    l_team_scores = results_df.set_index(['LTeamID', 'Season'])[['LScore', 'WScore']]
    l_point_differential = -l_team_scores.diff(axis=1)['WScore'].to_frame()

    coach_names = coaches_df.set_index(['TeamID', 'Season'])['CoachName']
    w_point_differential['coach'] = coach_names.loc[w_point_differential.index]
    l_point_differential['coach'] = coach_names.loc[l_point_differential.index]

    all_point_differentials = pd.concat([w_point_differential, l_point_differential], axis=0)
    coach_average_differential = all_point_differentials.groupby('coach').agg('mean')
    coach_average_differential.columns = ['diff']
    return coach_average_differential

coach_average_differential = get_average_coach_differential(tourney_coaches, tourney_compact_results)
coach_average_differential.sort_values(by='diff', ascending=False, inplace=True)
coach_average_differential = coach_average_differential.iloc[:num_plot_coaches]
coach_average_differential.index = make_names(coach_average_differential.index)

p = figure(plot_width=900,
           title='Top {} Average Point Differentials By Coach'.format(num_plot_coaches),
           x_range=(-1, num_plot_coaches),
           y_range=(0, 15))
p.xaxis.ticker = [i for i in range(num_plot_coaches)]
p.xaxis[0].formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
""" % list(coach_average_differential.index))
p.xaxis.major_label_orientation = np.pi / 4
p.xaxis.major_label_text_font_size = '10pt'

ds = ColumnDataSource({'x': np.arange(num_plot_coaches),
                       'y': coach_average_differential.values,
                       'name': coach_average_differential.index})
p.vbar(x='x', top='y', bottom=0,
       width=0.9, alpha=0.9,
       color="#3a94e9", hover_color="#ea5d4d",
       source=ds)
p.add_tools(HoverTool(tooltips=[('Coach Name', '@name'), ('Average Point Differential', '@y')]))
show(p)
# fun experiment: let's see how good of a predictor of victory the coach's past tournament point differential is
past_differentials = []
for season in range(2004, 2018):
    this_tourney_results = tourney_compact_results.set_index('Season').loc[season]
    this_tourney_coaches = tourney_coaches.set_index('Season').loc[season].set_index('TeamID')

    past_tourney_coaches = tourney_coaches[tourney_coaches['Season'] < season]
    past_tourney_results = tourney_compact_results[tourney_compact_results['Season'] < season]
    past_coach_differential = get_average_coach_differential(past_tourney_coaches, past_tourney_results)

    this_tourney_output = pd.DataFrame()
    w_tourney_coaches = this_tourney_coaches.loc[this_tourney_results['WTeamID']]['CoachName']
    this_tourney_output[['Wcoach', 'Wdiff']] = past_coach_differential.loc[w_tourney_coaches].reset_index()

    l_tourney_coaches = this_tourney_coaches.loc[this_tourney_results['LTeamID']]['CoachName']
    this_tourney_output[['Lcoach', 'Ldiff']] = past_coach_differential.loc[l_tourney_coaches].reset_index()

    this_tourney_output[['Wcoach', 'Lcoach']] = this_tourney_output[['Wcoach', 'Lcoach']].apply(make_names)

    this_tourney_team_names = team_names.set_index('TeamID')['TeamName']
    this_tourney_output['Wteam'] = this_tourney_team_names.loc[this_tourney_results['WTeamID']].values
    this_tourney_output['Lteam'] = this_tourney_team_names.loc[this_tourney_results['LTeamID']].values
    past_differentials.append(this_tourney_output)

# when computing win percentage, we'll assume that if you don't have a past tourney point differential, it's lower
# than any coach WITH a past differential
all_differentials = pd.concat(past_differentials)
min_differential = all_differentials[['Wdiff', 'Ldiff']].min() - 1
all_differentials_no_nan = all_differentials[['Wdiff', 'Ldiff']].fillna(min_differential)
percent_winners = (all_differentials_no_nan['Wdiff'] >= all_differentials_no_nan['Ldiff']).mean() * 100
print('Coach with the higher past average point differential won {:0.1f}% of the time'.format(percent_winners))
# let's plot this to get an idea of how these advantages are distributed for winners and losers
# for the purposes of plotting, we'll drop the nans
all_differentials = all_differentials[(~pd.isnull(all_differentials[['Wdiff', 'Ldiff']])).all(axis=1)]
high_diff = all_differentials[['Wdiff', 'Ldiff']].max(axis=1)
low_diff = all_differentials[['Wdiff', 'Ldiff']].min(axis=1)

plot_data_source = pd.DataFrame()
# x-axis will be "advantage" in point differential
plot_data_source['x'] = high_diff - low_diff

# y-axis is 1 if advantaged coach won, 0 otherwise
plot_data_source['y'] = (all_differentials['Wdiff'] >= all_differentials['Ldiff']).astype('int')

# we'll add some vertical jitter for plotting purposes
jitter_std = 0.1
plot_data_source['yj'] = plot_data_source['y'] + jitter_std*np.random.randn(len(plot_data_source))

colors = np.array(["#3a94e9", "#ea5d4d"])
plot_data_source['color'] = colors[plot_data_source['y']]

# we'll bring back in our original data to have access to columns for hovertool
plot_data_source = pd.concat([plot_data_source, all_differentials], axis=1)
ds = ColumnDataSource(plot_data_source)

p = figure(plot_height=400,
           plot_width=700,
           title='Point Differential Comparison',
           tools='')
p.xaxis.axis_label = 'Point Differential Advantage'
p.yaxis.ticker = [0, 1]
p.yaxis[0].formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
""" % ['Underdog Won', 'Favored Team Won'])
p.yaxis.major_label_orientation = np.pi / 2
p.yaxis.major_label_text_font_size = '8pt'

p.scatter(x='x',
          y='yj',
          fill_color='color',
          size=8,
          fill_alpha=0.5,
          line_alpha=0.8,
          line_color='color', 
          line_width=2.5,
          source=ds)
p.add_tools(HoverTool(tooltips=[('Winner', '@Wcoach - @Wteam, @Wdiff'),
                                ('Loser', '@Lcoach - @Lteam, @Ldiff')]))
show(p)
# that's coool enough, but let's make things a little fancier using custom javascript callbacks
# we'll add a selector to filter the scatter plot according to the coach

# start by creating a dummy data source to immutably hold all of our data
dummy_ds = ColumnDataSource(plot_data_source)

# now create custom javascript code for changing the data present in ds,
# which will change the data which is plotted
selector_js_code = """
var coach = cb_obj.value;

var data = source.data;
var dummy_data = dummy_source.data;
var col_names = source.column_names;

var wname = dummy_data['Wcoach'];
var lname = dummy_data['Lcoach'];

if (coach=='') {
    for (i=0; i < col_names.length; i++){
        data[col_names[i]] = dummy_data[col_names[i]];
    }
    source.change.emit()
    return
};

for (i=0; i < col_names.length; i++){
    data[col_names[i]] = [];
};

for (i=0; i < wname.length; i++) {
    if (coach==wname[i]) {
        for (j=0; j < col_names.length; j++){
            data[col_names[j]].push(dummy_data[col_names[j]][i])
        }
    } else if (coach==lname[i]) {
        for (j=0; j < col_names.length; j++){
            data[col_names[j]].push(dummy_data[col_names[j]][i])
        }
    }
};

source.change.emit()
"""
selector_callback = CustomJS(args={'source': ds, 'dummy_source': dummy_ds}, code=selector_js_code)

all_coach_names = list(pd.unique(plot_data_source[['Wcoach', 'Lcoach']].values.ravel('K')))
selector = Select(title='Filter By Coach', value='',
                  options=[''] + all_coach_names,
                  callback=selector_callback)
show(row(p, selector))
# finally, let's move past aggregate stats and take a look at what happens in various regions of advantage
# specifically, we'll make a selector tool which plots a pie chart of win percentages for arbitrary
# advantage regions
p.add_tools(BoxSelectTool(dimensions='width'))
pie_field_names = ['x', 'y', 'start_angle', 'end_angle', 'color', 'pct']
pie_empty_data = dict([(name, []) for name in pie_field_names])
pie_ds = ColumnDataSource(pie_empty_data)

pie_p = figure(title='Selected Region Precision',
               plot_height=300,
               plot_width=300,
               x_range=[-1.5, 1.5],
               y_range=[-1.5, 1.5],
               tools='')
pie_p.xaxis.visible = False
pie_p.yaxis.visible = False

pie_p.wedge(x='x', y='y',
            start_angle='start_angle', end_angle='end_angle',
            fill_color='color', fill_alpha=0.6,
            line_color='color', line_alpha=0.8, line_width=2,
            hover_fill_alpha=0.7, hover_fill_color='#888888', hover_line_color='#888888',
            radius=1, direction='clock',
            source=pie_ds)

# our js code this time will push data to our pie_ds based on the data that is selected
box_select_js = """
var inds = cb_obj.selected['1d'].indices;
var d1 = cb_obj.data;
var d2 = pie_ds.data;

if (inds.length == 0) {{
    for (i=0; i<pie_ds.column_names.length; i++) {{
        d2[pie_ds.column_names[i]] = []
    }};
    pie_ds.change.emit();
    return
}};

var pct1 = 0;
for (i=0; i<inds.length; i++) {{
    pct1 += d1['y'][inds[i]];
}}
pct1 /= inds.length;

d2['start_angle'] = [pct1*2*Math.PI, 2*Math.PI];
d2['end_angle'] = [0, pct1*2*Math.PI];
d2['x'] = [0, 0];
d2['y'] = [0, 0];
d2['color'] = {};
d2['pct'] = [100*(1-pct1), 100*pct1];
d2['team'] = ['Underdog', 'Favored Team'];

pie_ds.change.emit()
""".format(list(colors)[::-1])
box_select_callback = CustomJS(args={'pie_ds': pie_ds}, code=box_select_js)
ds.callback = box_select_callback

pie_p.add_tools(HoverTool(tooltips=[('Win Percentage', '@team: @pct%')], point_policy='follow_mouse'))

show(row(p, column(selector, pie_p)))