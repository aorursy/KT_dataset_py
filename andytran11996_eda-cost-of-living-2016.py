%matplotlib inline
import os
import pandas as pd
import pycountry
from plotly import tools
import plotly.graph_objs as go 
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import seaborn as sns
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 


def clean_usa_city_and_country(c):
    if len(c['Country']) == 2:
        c['City'] = c['City'] + ', ' + c['Country']
        c['Country'] = 'United States'
    return c['City'], c['Country']

df = pd.read_csv('../input/cost-of-living-2016.csv')
df.head(3)
rename_rules = {
    'Cost.of.Living.Index' : 'CLI',
    'Rent.Index': 'RI',
    'Cost.of.Living.Plus.Rent.Index': 'CLRI',
    'Groceries.Index' : 'GI',
    'Restaurant.Price.Index': 'RPI',
    'Local.Purchasing.Power.Index': 'LPPI'
}
df.rename(columns=rename_rules, inplace=True)
df['City'], df['Country'] = zip(*df.apply(clean_usa_city_and_country, axis=1).values)
df[df['Country'] == 'Germany'].head(3)
df_groupby_country = df.groupby(['Country']).mean().round(2)
data = dict(type="choropleth",
           locations = df_groupby_country.index.values,
            locationmode = "country names",
           z = df_groupby_country['CLI'],
           text = df_groupby_country.index.values,
           colorbar = {'title':'CLI'})

layout = dict(title="World AVG Cost of Living Index 2016",
             geo = dict(showframe=False,
                      projection = {'type':'mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
df_sort_by_cli = df.sort_values(['CLRI'], ascending=False)
df_cli_top10 = df_sort_by_cli.head(10)

indices = ['CLRI','CLI','RI','GI', 'RPI']
data = []
for idx in indices:
    bar = go.Bar(
        x=df_cli_top10['City'] + ', ' + df_cli_top10['Country'],
        y=df_cli_top10[idx] - 100,
        name=idx,
        opacity=0.6
    )
    data.append(bar)

layout = go.Layout(
    barmode='group',
    title='Top 10 most expensive cities in comparision to New York',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
indices = ['CLI','RI','GI','RPI']

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Cost of Living Index', 'Rent Index',
                                                          'Groceries Index', 'Restaurant Prices Index'))
for i,idx in enumerate(indices):
    row = i // 2 + 1
    col = i % 2 + 1
    fig.append_trace(go.Histogram(x=df[idx], histnorm='probability', opacity=0.6), row, col)
    
fig['layout'].update(title='Histograms', showlegend=False)

iplot(fig)
from bokeh.plotting import figure, output_notebook, show, save, output_file
from bokeh.layouts import row
from bokeh.core.properties import value
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.models import CheckboxGroup, CustomJS, ColumnDataSource, FactorRange
import numpy as np

output_notebook()
df_top_5 = df.sort_values(['CLI'], ascending = False).head(5)

to_select_cols = ['Cappuccino(regular)','Milk(regular)(1 liter)', 'Water(0.33 liter bottle)', 'Eggs(12)','Water(1.5 liter bottle)',
               'Domestic Beer (0.5 liter bottle)','Apples (1kg)', 'Cinema, International Release, 1 Seat', 'One-way Ticket (Local Transport)']

cols_selection = CheckboxGroup(labels=to_select_cols, 
                                  active = [0, 1,2,3,4], width=200)


data = {'cities': df_top_5['City'].values}
plot = {}
max = 0;
for col in to_select_cols:
    data[col] = df_top_5[col].values
    local_max = df_top_5[col].values.max()
    if local_max > max:
        max = local_max
source = ColumnDataSource(data=data)

p = figure(x_range=data['cities'], y_range=[0,int(max*1.7)],
           toolbar_location=None, tools="", sizing_mode='stretch_both')
colors = palettes.Category20[len(to_select_cols)]

for idx,col in enumerate(to_select_cols):
    w = 0.8 / len(to_select_cols)
    x = p.vbar(x=dodge('cities', - (len(to_select_cols)*w/2) + idx*w , range=p.x_range), top=col, width=w, source=source,
       color=colors[idx], legend=value(col), fill_alpha = 0.5)
    plot[col] = x
p.legend.location = "top_left"
#p.legend.click_policy="mute"
p.legend.visible = True
p.legend.border_line_alpha = 0.1
p.legend.background_fill_alpha = 0.1
callback = CustomJS(args=dict(plot = plot, cols = to_select_cols, src=source, fig = p), code="""
        for (var i = 0; i < cols.length; i++){
            var col = cols[i];
            plot[col].visible = cb_obj.active.indexOf(i) > -1;
        }
        var max = 0;
        var data = src.data
        for (var i = 0; i < cols.length; i++){
            if (cb_obj.active.indexOf(i) < 0) continue;
            for (var j = 0; j < data[cols[i]].length; j++){
                max = max > data[cols[i]][j] ? max :  data[cols[i]][j];
            }
        }
        if (cb_obj.active.length == 0) max = 10;
        fig.y_range.end = max*1.7;
        fig.y_range.start = 0;
    """)
cols_selection.callback = callback
show(row(cols_selection, p, sizing_mode = "fixed"))
