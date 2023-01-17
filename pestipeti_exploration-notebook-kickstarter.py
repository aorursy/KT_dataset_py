import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
py.init_notebook_mode(connected=True)

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.options.display.float_format = '{:.2f}'.format

FOLDER_ROOT = './../'
FOLDER_INPUT = FOLDER_ROOT + '/input'
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
kick_df = pd.read_csv(FOLDER_INPUT + '/ks-projects-201801.csv')
kick_df.drop(['name'], axis=1, inplace=True)
kick_df.isnull().sum()
kick_df.drop(['usd pledged'], axis=1, inplace=True)
kick_df.nunique()
kick_df.main_category.value_counts()
kick_df.country.unique()
weird_country = kick_df[kick_df.country == 'N,0"']
weird_country.ID.count()
weird_country.head()
weird_country.currency.value_counts()
# Only the 'real' countries...
kick_df[kick_df.country != 'N,0"'].groupby(['currency'])['country'].unique()
# Replace everything (including EUR -> general 'EU')
cond = (kick_df.country == 'N,0"')
kick_df.loc[cond, 'country'] = kick_df.loc[cond, 'currency'].str.slice(0,2)
co = kick_df.country.value_counts()

trace = go.Bar(x=co.index, y=co)
layout = dict(title = 'Number of projects / country',
              xaxis = dict(title = 'Country'),
              yaxis = dict(title = 'Number of projects'))

py.iplot(dict(data=[trace], layout=layout))
mc = kick_df.state.value_counts()

trace = go.Pie(labels=mc.index, values=mc, textinfo='value')
layout = dict(title = 'Project states')

py.iplot(dict(data=[trace], layout=layout))
kick_df.launched.head()
# Convert the `launched` column to datetime
kick_df['launched'] = pd.to_datetime(kick_df['launched'])
kick_df['launched_month'] = kick_df.launched.dt.to_period("M")

launched_per_month = kick_df[(kick_df.launched.dt.year > 1970) & (kick_df.launched.dt.year < 2018)].groupby(
    'launched_month', sort=True)['ID'].count()
trace = go.Scatter(
    x=launched_per_month.index.strftime('%Y-%m'),
    y=launched_per_month.values
)

layout = dict(title = 'Launched projects per months')

py.iplot(dict(data=[trace], layout=layout))
kick_df.goal.describe([.25, .5, .75, .85, .90, .95, .98, .99])
kick_df['goal_1000'] = kick_df[kick_df.goal < 90000].goal // 1000
trace = go.Histogram(
    x=kick_df['goal_1000']
)

layout = dict(title = 'Distribution of goals',
              xaxis = dict(title = 'Goal ($ thousands)'),
              yaxis = dict(title = 'Number of projects'))

py.iplot(dict(data=[trace], layout=layout))
kick_df.usd_pledged_real.describe([.25, .5, .75, .85, .90, .95, .98, .99])
kick_df['usd_pledged_real_1000'] = kick_df[(kick_df.usd_pledged_real > 1000) & (kick_df.usd_pledged_real < 65000)].usd_pledged_real // 1000
trace = go.Histogram(
    x=kick_df['usd_pledged_real_1000']
)

layout = dict(title = 'Distribution of pledged amount',
              xaxis = dict(title = 'Pledged ($ thousands)'),
              yaxis = dict(title = 'Number of projects'))

py.iplot(dict(data=[trace], layout=layout))
kick_df.backers.describe([.25, .5, .75, .85, .90, .95, .98, .99])
kick_df['backer_100'] = kick_df[kick_df.backers < 1500].backers // 100
trace = go.Histogram(
    x=kick_df['backer_100']
)

layout = dict(title = 'Distribution of backers',
              xaxis = dict(title = 'Backers (hundreds)'),
              yaxis = dict(title = 'Number of projects'))

py.iplot(dict(data=[trace], layout=layout))
mc = kick_df.state.value_counts()

trace = go.Bar(x=mc.index, y=mc)
layout = dict(title = 'Number of projects / state',
              xaxis = dict(title = 'State'),
              yaxis = dict(title = 'Number of projects'))

py.iplot(dict(data=[trace], layout=layout))
psuc_df = kick_df[kick_df.state == 'successful']
punsuc_df = kick_df[(kick_df.state == 'failed') | (kick_df.state == 'canceled')]
trace = go.Pie(
    labels=['Successful', 'Unsuccessful'],
    values=[psuc_df.ID.count(), punsuc_df.ID.count()]
)

layout = dict(title = 'Successful vs. unsuccessful projects')

py.iplot(dict(data=[trace], layout=layout))
# Remove some data for visualisation
psuc_ro_df = psuc_df[(psuc_df.launched.dt.year > 1970) & (psuc_df.launched.dt.year < 2018)]
punsuc_ro_df = punsuc_df[(punsuc_df.launched.dt.year > 1970) & (punsuc_df.launched.dt.year < 2018)]

success_per_month = psuc_ro_df.groupby('launched_month', sort=True)['ID'].count()
unsuccess_per_month = punsuc_ro_df.groupby('launched_month', sort=True)['ID'].count()

trace_success = go.Scatter(
    x=success_per_month.index.strftime('%Y-%m'),
    y=success_per_month.values,
    name='Successful projects'
)

trace_unsuccess = go.Scatter(
    x=unsuccess_per_month.index.strftime('%Y-%m'),
    y=unsuccess_per_month.values,
    name='Unsuccessful projects'
)

layout = dict(title = 'Number of projects (2009 - 2017)')
py.iplot(dict(data=[trace_success, trace_unsuccess], layout=layout))
trace = go.Scatter(
    x=success_per_month.index.strftime('%Y-%m'),
    y=success_per_month.values / (success_per_month.values + unsuccess_per_month.values),
    name='Success rate'
)

layout = dict(title = 'Overall success rate')
py.iplot(dict(data=[trace], layout=layout))
suc_us = psuc_ro_df[psuc_ro_df.country == 'US'].groupby('launched_month', sort=True)['ID'].count()
suc_us.sort_index(inplace=True)

unsuc_us = punsuc_ro_df[punsuc_ro_df.country == 'US'].groupby('launched_month', sort=True)['ID'].count()
unsuc_us = unsuc_us.reindex(suc_us.index, fill_value=0)

suc_non_us = psuc_ro_df[psuc_ro_df.country != 'US'].groupby('launched_month', sort=True)['ID'].count()
suc_non_us = suc_non_us.reindex(suc_us.index, fill_value=0)

unsuc_non_us = punsuc_ro_df[punsuc_ro_df.country != 'US'].groupby('launched_month', sort=True)['ID'].count()
unsuc_non_us = unsuc_non_us.reindex(suc_us.index, fill_value=0)

trace_us = go.Scatter(
    x=suc_us.index.strftime('%Y-%m'),
    y=suc_us.values + unsuc_us.values,
    name='US project'
)

trace_non_us = go.Scatter(
    x=suc_us.index.strftime('%Y-%m'),
    y=suc_non_us.values + unsuc_non_us,
    name='non-US projects'
)

trace_success_rate = go.Scatter(
    x=suc_us.index.strftime('%Y-%m'),
    y=suc_us.values / (suc_us.values + unsuc_us.values),
    name='US success rate'
)

trace_nonus_success_rate = go.Scatter(
    x=suc_us.index.strftime('%Y-%m'),
    y=suc_non_us.divide(suc_non_us + unsuc_non_us).fillna(0),
    name='non-US success rate'
)

layout = dict(title = 'Number of US vs. non-US projects')
py.iplot(dict(data=[trace_us, trace_non_us], layout=layout))
layout = dict(title = 'Success rate: US vs. non-US')
py.iplot(dict(data=[trace_success_rate, trace_nonus_success_rate], layout=layout))
df = pd.DataFrame(columns=['country', 'projects', 'success_p', 'unsuccess_p', 'success_rate'])
df['country'] = kick_df.country.unique()

for index, row in df.iterrows():
    c = row.country
    row.success_p = psuc_ro_df[psuc_ro_df.country == c]['ID'].count()
    row.unsuccess_p = punsuc_ro_df[punsuc_ro_df.country == c]['ID'].count()
    row.projects = row.success_p + row.unsuccess_p
    row.success_rate = row.success_p / row.projects


df.sort_values('success_rate', ascending=False, inplace=True)

sr = df[df.projects > 1000].success_rate;

trace_su = go.Bar(
    x=df[df.projects > 1000].country,
    y=sr,
    text=np.around((sr * 100).astype(np.double), 1),
    textposition = 'auto',
    textfont = dict(
        color='#e8e8e8'
    ),
    name='Successful projects',
)

trace_unsu = go.Bar(
    x=df[df.projects > 1000].country,
    y=(1 - sr),
    text=np.around(((1 - sr) * 100).astype(np.double), 1),
    textposition = 'auto',
    name='Unsuccessful projects'
)

layout = dict(title = 'Success rate / country',
              yaxis = dict(title = 'Success rate of projects'),
              annotations=go.Annotations([
                  go.Annotation(
                      x=0.5004254919715793,
                      y=-0.16191064079952971,
                      showarrow=False,
                      text='(Countries at least 1000 projects)',
                      xref='paper',
                      yref='paper'
                  ),
              ]),              
              barmode='stack')

py.iplot(dict(data=[trace_su, trace_unsu], layout=layout))
mc = kick_df.main_category.value_counts()

trace = go.Bar(x=mc.index, y=mc)
layout = dict(title = 'Number of projects / main category',
              xaxis = dict(title = 'Main Category'),
              yaxis = dict(title = 'Number of projects'))

py.iplot(dict(data=[trace], layout=layout))
df = pd.DataFrame(columns=['main_category', 'projects', 'success_p', 'unsuccess_p', 'success_rate'])
df['main_category'] = kick_df.main_category.unique()

for index, row in df.iterrows():
    c = row.main_category
    row.success_p = psuc_ro_df[psuc_ro_df.main_category == c]['ID'].count()
    row.unsuccess_p = punsuc_ro_df[punsuc_ro_df.main_category == c]['ID'].count()
    row.projects = row.success_p + row.unsuccess_p
    row.success_rate = row.success_p / row.projects

df.sort_values('success_rate', ascending=False, inplace=True)

sr = df.success_rate;

trace_su = go.Bar(
    x=df.main_category,
    y=sr,
    text=np.around((sr * 100).astype(np.double), 1),
    textposition = 'auto',
    textfont = dict(
        color='#e8e8e8'
    ),
    name='Successful projects',
)

trace_unsu = go.Bar(
    x=df.main_category,
    y=(1 - sr),
    text=np.around(((1 - sr) * 100).astype(np.double), 1),
    textposition = 'auto',
    name='Unsuccessful projects'
)

layout = dict(title = 'Success rate / category',
              yaxis = dict(title = 'Success rate of categories'),
              barmode='stack')

py.iplot(dict(data=[trace_su, trace_unsu], layout=layout))
cdf = pd.DataFrame()
traces = []
titles = []

for c in kick_df.main_category.unique():
    cdf[c] = psuc_ro_df[psuc_ro_df.main_category == c].groupby('launched_month', sort=True)['ID'].count()

cdf.fillna(0, inplace=True)

for c in kick_df.main_category.unique():
    titles.append(c)
    trace = go.Scatter(
        x=cdf.index.strftime('%Y-%m'),
        y=cdf[c].values,
        name=c
    )
    traces.append(trace)

fig = tools.make_subplots(rows=5, cols=3, subplot_titles=titles, print_grid=False)

for i, trace in enumerate(traces):
    fig.append_trace(trace, (i // 3) + 1, (i % 3) + 1 )

fig['layout'].update(height=800, title='Number of projects / category')

py.iplot(fig)
cdf = pd.DataFrame()
traces = []
titles = []

for c in kick_df.main_category.unique():
    cat_s = psuc_ro_df[psuc_ro_df.main_category == c].groupby('launched_month', sort=True)['ID'].count()
    cat_us = punsuc_ro_df[punsuc_ro_df.main_category == c].groupby('launched_month', sort=True)['ID'].count()
    cdf[c] = cat_s.divide(cat_s + cat_us)

cdf.fillna(0, inplace=True)


for c in kick_df.main_category.unique():
    titles.append(c)
    trace = go.Scatter(
        x=cdf.index.strftime('%Y-%m'),
        y=cdf[c].values,
        name=c
    )
    traces.append(trace)

fig = tools.make_subplots(rows=5, cols=3, subplot_titles=titles, print_grid=False)

for i, trace in enumerate(traces):
    fig.append_trace(trace, (i // 3) + 1, (i % 3) + 1 )

fig['layout'].update(height=800, title='Success rate of projects / category')

py.iplot(fig)